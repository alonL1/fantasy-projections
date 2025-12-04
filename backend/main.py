import json
import os
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import nfl_data_py as nfl

from dotenv import load_dotenv
load_dotenv(override=True)

# Prototype CLI that estimates fantasy points from betting lines.
print("API KEY LOADED:", os.environ.get("ODDS_API_KEY"))

FLEX_POSITIONS = {"RB", "WR", "TE"}
SCORING_MULTIPLIERS = {"standard": 0.0, "half-ppr": 0.5, "ppr": 1.0}
THE_ODDS_API_KEY_ENV = "ODDS_API_KEY"
THE_ODDS_API_BASE = os.environ.get("ODDS_API_BASE", "https://api.the-odds-api.com/v4")
THE_ODDS_API_REGIONS = os.environ.get("ODDS_API_REGIONS", "us")
THE_ODDS_API_BOOKS = os.environ.get("ODDS_API_BOOKS")  # optional comma list filter
THE_ODDS_API_MAX_EVENTS = int(os.environ.get("THE_ODDS_API_MAX_EVENTS", "1"))  # limit how many events to scan
USE_LIVE_ODDS = os.environ.get("USE_LIVE_ODDS", "1") != "0"
ANYTIME_TD_DEVIG_FACTOR = 0.92  # rough guess to strip bookmaker hold from single-sided markets

MARKET_ALIASES = {
    "receiving_yards": ["player_reception_yds"],
    "rushing_yards": ["player_rush_yds"],
    "receptions": ["player_receptions"],
    "anytime_td": ["player_anytime_td"],
}
ALL_MARKET_KEYS = sorted({alias for aliases in MARKET_ALIASES.values() for alias in aliases})
_ROSTER_DF_CACHE: Optional[Any] = None


def load_current_roster() -> Optional[Any]:
    """
    Pull the current season roster from nfl_data_py (cached per run).
    Falls back to the prior season if the current year is unavailable.
    """
    global _ROSTER_DF_CACHE
    if _ROSTER_DF_CACHE is not None:
        return _ROSTER_DF_CACHE

    current_year = date.today().year
    desired_columns = ["player_id", "player_name", "position", "team"]
    for season_year in (current_year, current_year - 1):
        roster_df = None
        try:
            roster_df = nfl.import_seasonal_rosters([season_year], columns=desired_columns)
        except KeyError as exc:
            print(f"Roster columns missing {exc}; retrying with full dataset for {season_year}.")
            try:
                roster_df = nfl.import_seasonal_rosters([season_year])
            except Exception as inner_exc:  # noqa: BLE001
                print(f"Full roster fetch failed for {season_year}: {inner_exc}")
                continue
        except Exception as exc:  # noqa: BLE001 - surface any failure
            print(f"Could not fetch roster for {season_year}: {exc}")
            continue

        if roster_df is None or "player_name" not in roster_df or "position" not in roster_df:
            print(f"Roster data missing required columns for {season_year}; skipping.")
            continue

        roster_df["player_name_clean"] = roster_df["player_name"].str.strip().str.lower()
        _ROSTER_DF_CACHE = roster_df
        print(f"Loaded roster data for {season_year}.")
        return _ROSTER_DF_CACHE

    print("Failed to load roster data from nfl_data_py; position lookup unavailable.")
    return None


def lookup_player_position(player_name: str) -> Optional[str]:
    """Return the player's position from the live roster feed."""
    roster_df = load_current_roster()
    if roster_df is None:
        return None

    search_name = normalize_player_name(player_name).lower()
    matches = roster_df[roster_df["player_name_clean"] == search_name]
    if matches.empty:
        return None

    position_value = matches["position"].iloc[0]
    return str(position_value).strip().upper() if position_value else None


def implied_probability(american_odds: float) -> Optional[float]:
    """Convert American odds to implied probability."""
    if american_odds is None:
        return None
    if american_odds > 0:
        return 100.0 / (american_odds + 100.0)
    return -american_odds / (-american_odds + 100.0)


def prob_to_american(prob: float) -> Optional[int]:
    """Convert probability back to a rough American odds number for display."""
    if prob is None or prob <= 0.0 or prob >= 1.0:
        return None
    if prob >= 0.5:
        return -int(round((prob / (1 - prob)) * 100))
    return int(round(((1 - prob) / prob) * 100))


def adjust_line_with_skew(line: float, over_prob: Optional[float], under_prob: Optional[float]):
    """
    If over/under probabilities are far apart, nudge the line toward the favored side.
    Adjustment is intentionally small: 10% of the line scaled by prob difference.
    """
    if over_prob is None or under_prob is None:
        return line, 0.0, None
    skew = over_prob - under_prob
    adjustment = line * 0.1 * skew
    return line + adjustment, adjustment, skew


def blended_market(quotes: List[Dict]):
    """Average multiple book lines/odds into a single blended view."""
    if not quotes:
        return None
    avg_line = sum(q["line"] for q in quotes) / len(quotes)
    over_probs = [implied_probability(q.get("over_odds")) for q in quotes if q.get("over_odds") is not None]
    under_probs = [implied_probability(q.get("under_odds")) for q in quotes if q.get("under_odds") is not None]
    over_prob_raw = sum(over_probs) / len(over_probs) if over_probs else None
    under_prob_raw = sum(under_probs) / len(under_probs) if under_probs else None
    over_prob, under_prob = devig_over_under(over_prob_raw, under_prob_raw)
    return {
        "line": avg_line,
        "over_prob": over_prob,
        "under_prob": under_prob,
        "over_prob_raw": over_prob_raw,
        "under_prob_raw": under_prob_raw,
    }


def devig_over_under(over_prob: Optional[float], under_prob: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    """Normalize over/under implied probabilities so they sum to 1.0."""
    if over_prob is None or under_prob is None:
        return over_prob, under_prob
    total = over_prob + under_prob
    if total == 0:
        return over_prob, under_prob
    return over_prob / total, under_prob / total


def fmt_prob(prob: Optional[float]) -> str:
    return "n/a" if prob is None else f"{prob:.3f}"


def normalize_book_name(book: Dict) -> str:
    return book.get("key") or book.get("title") or "UnknownBook"


def match_player_outcome(player_lower: str, outcome: Dict) -> Tuple[bool, Optional[str]]:
    """
    Try to map an outcome object to the requested player and return the side (over/under/yes/no).
    Works across common shapes for player prop APIs.
    """
    name_lower = str(outcome.get("name") or "").lower()
    desc_lower = str(outcome.get("description") or "").lower()
    participant_lower = str(outcome.get("participant") or "").lower()
    side_candidates = {"over", "under", "yes", "no"}

    if player_lower == name_lower:
        side = desc_lower if desc_lower in side_candidates else None
        return True, side
    if player_lower == desc_lower:
        side = name_lower if name_lower in side_candidates else None
        return True, side
    if player_lower == participant_lower:
        if name_lower in side_candidates:
            return True, name_lower
        if desc_lower in side_candidates:
            return True, desc_lower
        return True, None
    return False, None


def map_the_odds_api_response(player_lower: str, payload: List[Dict]) -> Dict[str, List[Dict]]:
    alias_to_stat = {alias: stat for stat, aliases in MARKET_ALIASES.items() for alias in aliases}
    markets_out: Dict[str, List[Dict]] = {stat: [] for stat in MARKET_ALIASES}

    for game in payload:
        for book in game.get("bookmakers", []):
            book_name = normalize_book_name(book)
            for market in book.get("markets", []):
                stat = alias_to_stat.get(market.get("key"))
                if not stat:
                    continue

                # Anytime TD is a single outcome per player (no over/under split)
                if stat == "anytime_td":
                    for outcome in market.get("outcomes", []):
                        matched, _ = match_player_outcome(player_lower, outcome)
                        if not matched:
                            continue
                        odds = outcome.get("price")
                        if odds is None:
                            continue
                        markets_out[stat].append({"book": book_name, "odds": odds})
                    continue

                # Over/Under markets
                over_odds = None
                under_odds = None
                line = None

                for outcome in market.get("outcomes", []):
                    matched, side = match_player_outcome(player_lower, outcome)
                    if not matched:
                        continue
                    if side not in {"over", "under"}:
                        continue
                    price = outcome.get("price")
                    point = outcome.get("point")
                    if price is None or point is None:
                        continue
                    line = line if line is not None else point
                    if side == "over":
                        over_odds = price
                    elif side == "under":
                        under_odds = price
                    if line is None:
                        line = point

                if over_odds is None and under_odds is None:
                    continue

                markets_out[stat].append(
                    {
                        "book": book_name,
                        "line": line,
                        "over_odds": over_odds,
                        "under_odds": under_odds,
                    }
                )

    # Drop empty stats for clarity
    return {k: v for k, v in markets_out.items() if v}


def fetch_live_markets_from_the_odds_api(player_name: str) -> Optional[Dict[str, List[Dict]]]:
    api_key = os.environ.get(THE_ODDS_API_KEY_ENV)
    if not USE_LIVE_ODDS:
        print("Live odds disabled via USE_LIVE_ODDS=0.")
        return None
    if not api_key:
        print(f"Live odds enabled but {THE_ODDS_API_KEY_ENV} is not set.")
        return None

    player_lower = player_name.lower()
    print(f"[live] Fetching props for '{player_lower}' from The Odds API.")
    print(
        f"[live] Config base={THE_ODDS_API_BASE}, regions={THE_ODDS_API_REGIONS}, books={THE_ODDS_API_BOOKS}, markets={','.join(ALL_MARKET_KEYS)}"
    )

    # 1) Get upcoming NFL events (does not cost usage credits)
    events_url = f"{THE_ODDS_API_BASE}/sports/americanfootball_nfl/events?{urllib.parse.urlencode({'apiKey': api_key})}"
    try:
        with urllib.request.urlopen(events_url, timeout=10) as resp:
            events_body = resp.read()
        events = json.loads(events_body.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001 - surface any failure
        print(f"Failed to fetch NFL events from The Odds API: {exc}")
        return None

    print(f"[live] Retrieved {len(events)} upcoming events.")

    def parse_commence_time(ev: Dict) -> Optional[datetime]:
        raw = ev.get("commence_time")
        if not raw:
            return None
        try:
            # API returns ISO-8601 string; ensure timezone-aware for comparisons
            dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
            return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
        except Exception:
            return None

    now = datetime.now(timezone.utc)
    events_sorted = sorted(
        events, key=lambda ev: parse_commence_time(ev) or datetime.max.replace(tzinfo=timezone.utc)
    )
    upcoming_events = [ev for ev in events_sorted if (parse_commence_time(ev) or now) >= now]
    chosen_events = upcoming_events or events_sorted
    if THE_ODDS_API_MAX_EVENTS > 0:
        chosen_events = chosen_events[:THE_ODDS_API_MAX_EVENTS]

    print(f"[live] Limiting to {len(chosen_events)} event(s) (nearest upcoming first).")

    # 2) Base params for event-odds requests
    params_base = {
        "apiKey": api_key,
        "regions": THE_ODDS_API_REGIONS,
        "markets": ",".join(ALL_MARKET_KEYS),
        "oddsFormat": "american",
    }
    if THE_ODDS_API_BOOKS:
        params_base["bookmakers"] = THE_ODDS_API_BOOKS

    collected: Dict[str, List[Dict]] = {k: [] for k in MARKET_ALIASES.keys()}

    def have_enough() -> bool:
        needed_stats = ("receiving_yards", "rushing_yards", "receptions", "anytime_td")
        return all(len(collected.get(stat, [])) >= 2 for stat in needed_stats)

    # 3) Loop over events and query /events/{id}/odds
    for event in chosen_events:
        event_id = event.get("id")
        if not event_id:
            continue

        params = params_base.copy()
        odds_url = f"{THE_ODDS_API_BASE}/sports/americanfootball_nfl/events/{event_id}/odds?{urllib.parse.urlencode(params)}"
        print(f"[live] Fetching event odds for event {event_id}: {odds_url}")

        try:
            with urllib.request.urlopen(odds_url, timeout=10) as resp:
                body = resp.read()
            event_payload = json.loads(body.decode("utf-8"))
        except urllib.error.URLError as exc:
            print(f"Event odds fetch failed for event {event_id}: {exc}")
            continue
        except json.JSONDecodeError:
            print(f"Event odds for {event_id} returned invalid JSON.")
            continue

        per_event_markets = map_the_odds_api_response(player_lower, [event_payload])
        print(
            f"[live] Event {event_id} yielded markets: { {k: len(v) for k, v in per_event_markets.items()} }"
        )

        for stat, quotes in per_event_markets.items():
            collected.setdefault(stat, []).extend(quotes)

        if have_enough():
            print("[live] Collected enough quotes across events; stopping early.")
            break

    # 4) Strip empty stats and return
    markets = {stat: quotes for stat, quotes in collected.items() if quotes}
    if markets:
        print("Using live player props from TheOddsAPI (event-odds endpoint).")
        return markets

    print("No live player props found for that player on upcoming games.")
    return None


def compute_td_points(odds_quotes: List[Dict], verbose: bool = True) -> float:
    """Average TD odds to a probability, then convert to fantasy points."""
    probs_raw = [implied_probability(q["odds"]) for q in odds_quotes if q.get("odds") is not None]
    probs = [p * ANYTIME_TD_DEVIG_FACTOR for p in probs_raw if p is not None]
    if not probs:
        return 0.0
    avg_prob = sum(probs) / len(probs)
    if verbose:
        print(
            f"  Blended anytime TD probability: {avg_prob:.3f} (devig factor {ANYTIME_TD_DEVIG_FACTOR}, raw avg {sum(probs_raw)/len(probs_raw):.3f})"
        )
        print(f"  Approx implied odds after devig: {prob_to_american(avg_prob)} American")
    points = avg_prob * 6.0
    if verbose:
        print(f"  TD fantasy points: {avg_prob:.3f} * 6 = {points:.2f}")
    return points


def scoring_choice() -> str:
    while True:
        choice = input("Scoring format (standard / half-ppr / ppr): ").strip().lower()
        if choice in SCORING_MULTIPLIERS:
            return choice
        print("Please enter one of: standard, half-ppr, ppr.")


def confirm_position(name: str, guessed_position: str) -> str:
    response = input(f"I think {name} plays {guessed_position}. Is that right? (y/n): ").strip().lower()
    if response in {"y", "yes"}:
        return guessed_position
    corrected = input("Enter the correct position (RB/WR/TE/QB/K/DEF): ").strip().upper()
    return corrected or guessed_position


def fetch_markets(player_name: str, position: str) -> Dict[str, List[Dict]]:
    """
    Try live odds first (TheOddsAPI), then fall back to baked-in mocks so the flow still runs.
    """
    live_markets = fetch_live_markets_from_the_odds_api(player_name)
    if live_markets:
        return live_markets

    key = player_name.lower()
    if key in PLAYERS:
        print("Using mock odds for known player (live fetch unavailable).")
        return PLAYERS[key]["markets"]
    print("No live odds found. Using generic mock lines for testing.")
    return DEFAULT_MARKETS_BY_POSITION.get(position, DEFAULT_MARKETS_BY_POSITION["GENERIC"])


def run_flex_pipeline(
    name: str,
    position: str,
    scoring: str,
    markets: Dict[str, List[Dict]],
    *,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Compute projection details; if verbose, also print a readable breakdown.
    Returns a dict suitable for API responses.
    """
    log = print if verbose else (lambda *args, **kwargs: None)
    log(f"\n--- FLEX projection for {name} ({position}) ---")
    log(f"Scoring: {scoring}\n")

    total_points = 0.0
    components: List[Dict[str, Any]] = []

    def add_component(stat: str, points: float, detail: Dict[str, Any]):
        components.append({"stat": stat, "points": points, **detail})

    # Receiving yards
    if "receiving_yards" in markets:
        log("Receiving yards over/under:")
        for q in markets["receiving_yards"]:
            log(f"  {q['book']}: line {q['line']} (O {q.get('over_odds')}, U {q.get('under_odds')})")
        blended = blended_market(markets["receiving_yards"])
        if blended:
            adjusted, delta, skew = adjust_line_with_skew(
                blended["line"], blended.get("over_prob"), blended.get("under_prob")
            )
            over_prob = blended.get("over_prob")
            under_prob = blended.get("under_prob")
            over_prob_raw = blended.get("over_prob_raw")
            under_prob_raw = blended.get("under_prob_raw")
            if over_prob is not None and under_prob is not None:
                log(
                    f"  Blended line {blended['line']:.2f}, over prob {fmt_prob(over_prob)} (raw {fmt_prob(over_prob_raw)}), under prob {fmt_prob(under_prob)} (raw {fmt_prob(under_prob_raw)}), skew {skew:+.3f}"
                )
                log(f"  Adjusted line: {adjusted:.2f} (delta {delta:+.2f})")
            else:
                log(f"  Using blended line {blended['line']:.2f} (no skew adjustment)")
            rec_points = adjusted / 10.0
            log(f"  Fantasy points from receiving yards: {adjusted:.2f} / 10 = {rec_points:.2f}\n")
            total_points += rec_points
            add_component(
                "receiving_yards",
                rec_points,
                {
                    "blended_line": blended["line"],
                    "adjusted_line": adjusted,
                    "delta": delta,
                    "over_prob": over_prob,
                    "under_prob": under_prob,
                    "over_prob_raw": over_prob_raw,
                    "under_prob_raw": under_prob_raw,
                    "skew": skew,
                },
            )

    # Rushing yards
    if "rushing_yards" in markets:
        log("Rushing yards over/under:")
        for q in markets["rushing_yards"]:
            log(f"  {q['book']}: line {q['line']} (O {q.get('over_odds')}, U {q.get('under_odds')})")
        blended = blended_market(markets["rushing_yards"])
        if blended:
            adjusted, delta, skew = adjust_line_with_skew(
                blended["line"], blended.get("over_prob"), blended.get("under_prob")
            )
            over_prob = blended.get("over_prob")
            under_prob = blended.get("under_prob")
            over_prob_raw = blended.get("over_prob_raw")
            under_prob_raw = blended.get("under_prob_raw")
            if over_prob is not None and under_prob is not None:
                log(
                    f"  Blended line {blended['line']:.2f}, over prob {fmt_prob(over_prob)} (raw {fmt_prob(over_prob_raw)}), under prob {fmt_prob(under_prob)} (raw {fmt_prob(under_prob_raw)}), skew {skew:+.3f}"
                )
                log(f"  Adjusted line: {adjusted:.2f} (delta {delta:+.2f})")
            else:
                log(f"  Using blended line {blended['line']:.2f} (no skew adjustment)")
            rush_points = adjusted / 10.0
            log(f"  Fantasy points from rushing yards: {adjusted:.2f} / 10 = {rush_points:.2f}\n")
            total_points += rush_points
            add_component(
                "rushing_yards",
                rush_points,
                {
                    "blended_line": blended["line"],
                    "adjusted_line": adjusted,
                    "delta": delta,
                    "over_prob": over_prob,
                    "under_prob": under_prob,
                    "over_prob_raw": over_prob_raw,
                    "under_prob_raw": under_prob_raw,
                    "skew": skew,
                },
            )

    # Receptions
    if "receptions" in markets:
        log("Receptions over/under:")
        for q in markets["receptions"]:
            log(f"  {q['book']}: line {q['line']} (O {q.get('over_odds')}, U {q.get('under_odds')})")
        blended = blended_market(markets["receptions"])
        if blended:
            adjusted, delta, skew = adjust_line_with_skew(
                blended["line"], blended.get("over_prob"), blended.get("under_prob")
            )
            over_prob = blended.get("over_prob")
            under_prob = blended.get("under_prob")
            over_prob_raw = blended.get("over_prob_raw")
            under_prob_raw = blended.get("under_prob_raw")
            if over_prob is not None and under_prob is not None:
                log(
                    f"  Blended line {blended['line']:.2f}, over prob {fmt_prob(over_prob)} (raw {fmt_prob(over_prob_raw)}), under prob {fmt_prob(under_prob)} (raw {fmt_prob(under_prob_raw)}), skew {skew:+.3f}"
                )
                log(f"  Adjusted line: {adjusted:.2f} (delta {delta:+.2f})")
            else:
                log(f"  Using blended line {blended['line']:.2f} (no skew adjustment)")
            multiplier = SCORING_MULTIPLIERS[scoring]
            rec_pts = adjusted * multiplier
            log(f"  Reception multiplier for {scoring}: {multiplier}")
            log(f"  Fantasy points from receptions: {adjusted:.2f} * {multiplier} = {rec_pts:.2f}\n")
            total_points += rec_pts
            add_component(
                "receptions",
                rec_pts,
                {
                    "blended_line": blended["line"],
                    "adjusted_line": adjusted,
                    "delta": delta,
                    "over_prob": over_prob,
                    "under_prob": under_prob,
                    "over_prob_raw": over_prob_raw,
                    "under_prob_raw": under_prob_raw,
                    "skew": skew,
                    "multiplier": multiplier,
                },
            )

    # Anytime TD
    if "anytime_td" in markets:
        log("Anytime TD odds:")
        for q in markets["anytime_td"]:
            log(f"  {q['book']}: odds {q.get('odds')}")
        td_points = compute_td_points(markets["anytime_td"], verbose=verbose)
        total_points += td_points
        add_component("anytime_td", td_points, {})
        log()

    log(f"Projected total fantasy points: {total_points:.2f}")
    return {
        "player": name,
        "position": position,
        "scoring": scoring,
        "total_points": total_points,
        "components": components,
    }


def normalize_player_name(name: str) -> str:
    return " ".join(name.strip().split())


def main():
    print("Fantasy Points Predictor (prototype)")
    player_name = normalize_player_name(input("Enter an NFL player: "))
    if not player_name:
        print("No player provided. Exiting.")
        return

    guessed_position = lookup_player_position(player_name)
    if guessed_position:
        print(f"Found position via roster data: {guessed_position}")
    if guessed_position:
        position = confirm_position(player_name, guessed_position)
    else:
        position = input("Position not found. Enter position (RB/WR/TE/QB/K/DEF): ").strip().upper()

    if position not in FLEX_POSITIONS:
        print(f"We cannot handle {position} yet. FLEX positions (RB/WR/TE) only. Coming soon.")
        return

    scoring = scoring_choice()
    markets = fetch_markets(player_name, position)
    run_flex_pipeline(player_name, position, scoring, markets)


PLAYERS: Dict[str, Dict] = {
    "justin jefferson": {
        "position": "WR",
        "markets": {
            "receiving_yards": [
                {"book": "SharpBook", "line": 93.5, "over_odds": -115, "under_odds": -105},
                {"book": "MarketAvg", "line": 94.5, "over_odds": -110, "under_odds": -110},
            ],
            "receptions": [
                {"book": "SharpBook", "line": 7.5, "over_odds": -120, "under_odds": +100},
                {"book": "MarketAvg", "line": 8.0, "over_odds": -110, "under_odds": -105},
            ],
            "rushing_yards": [
                {"book": "SharpBook", "line": 5.5, "over_odds": -105, "under_odds": -115},
            ],
            "anytime_td": [
                {"book": "SharpBook", "odds": -105},
                {"book": "MarketAvg", "odds": -115},
            ],
        },
    },
    "christian mccaffrey": {
        "position": "RB",
        "markets": {
            "receiving_yards": [
                {"book": "SharpBook", "line": 35.5, "over_odds": -110, "under_odds": -110},
                {"book": "MarketAvg", "line": 33.5, "over_odds": -115, "under_odds": -105},
            ],
            "receptions": [
                {"book": "SharpBook", "line": 4.5, "over_odds": -130, "under_odds": +105},
                {"book": "MarketAvg", "line": 4.5, "over_odds": -125, "under_odds": -105},
            ],
            "rushing_yards": [
                {"book": "SharpBook", "line": 72.5, "over_odds": -115, "under_odds": -105},
                {"book": "MarketAvg", "line": 70.5, "over_odds": -110, "under_odds": -110},
            ],
            "anytime_td": [
                {"book": "SharpBook", "odds": -190},
                {"book": "MarketAvg", "odds": -175},
            ],
        },
    },
    "travis kelce": {
        "position": "TE",
        "markets": {
            "receiving_yards": [
                {"book": "SharpBook", "line": 74.5, "over_odds": -120, "under_odds": +100},
                {"book": "MarketAvg", "line": 72.5, "over_odds": -115, "under_odds": -105},
            ],
            "receptions": [
                {"book": "SharpBook", "line": 6.5, "over_odds": -125, "under_odds": +105},
                {"book": "MarketAvg", "line": 6.5, "over_odds": -120, "under_odds": -110},
            ],
            "rushing_yards": [
                {"book": "SharpBook", "line": 0.5, "over_odds": +200, "under_odds": -300},
            ],
            "anytime_td": [
                {"book": "SharpBook", "odds": -105},
                {"book": "MarketAvg", "odds": -110},
            ],
        },
    },
    "josh allen": {
        "position": "QB",
        "markets": {
            "passing_yards": [],
        },
    },
}


DEFAULT_MARKETS_BY_POSITION: Dict[str, Dict[str, List[Dict]]] = {
    "RB": {
        "receiving_yards": [{"book": "Composite", "line": 25.5, "over_odds": -110, "under_odds": -110}],
        "receptions": [{"book": "Composite", "line": 3.0, "over_odds": -110, "under_odds": -110}],
        "rushing_yards": [{"book": "Composite", "line": 55.0, "over_odds": -110, "under_odds": -110}],
        "anytime_td": [{"book": "Composite", "odds": -120}],
    },
    "WR": {
        "receiving_yards": [{"book": "Composite", "line": 62.5, "over_odds": -110, "under_odds": -110}],
        "receptions": [{"book": "Composite", "line": 5.5, "over_odds": -110, "under_odds": -110}],
        "rushing_yards": [{"book": "Composite", "line": 4.0, "over_odds": -110, "under_odds": -110}],
        "anytime_td": [{"book": "Composite", "odds": -105}],
    },
    "TE": {
        "receiving_yards": [{"book": "Composite", "line": 42.5, "over_odds": -110, "under_odds": -110}],
        "receptions": [{"book": "Composite", "line": 4.5, "over_odds": -110, "under_odds": -110}],
        "rushing_yards": [{"book": "Composite", "line": 0.5, "over_odds": +200, "under_odds": -300}],
        "anytime_td": [{"book": "Composite", "odds": +120}],
    },
    "GENERIC": {
        "receiving_yards": [{"book": "Composite", "line": 50.0, "over_odds": -110, "under_odds": -110}],
        "receptions": [{"book": "Composite", "line": 4.0, "over_odds": -110, "under_odds": -110}],
        "rushing_yards": [{"book": "Composite", "line": 25.0, "over_odds": -110, "under_odds": -110}],
        "anytime_td": [{"book": "Composite", "odds": +110}],
    },
}


if __name__ == "__main__":
    main()
