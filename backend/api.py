from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from main import (
    FLEX_POSITIONS,
    SCORING_MULTIPLIERS,
    fetch_markets,
    lookup_player_position,
    normalize_player_name,
    run_flex_pipeline,
)


class PredictionRequest(BaseModel):
    player_name: str
    scoring: str = "ppr"


app = FastAPI(title="Fantasy Points Predictor", version="0.1.0")


@app.post("/predict")
async def predict(req: PredictionRequest):
    player = normalize_player_name(req.player_name)
    if not player:
        raise HTTPException(status_code=400, detail="player_name is required")

    scoring = req.scoring.lower()
    if scoring not in SCORING_MULTIPLIERS:
        raise HTTPException(
            status_code=400,
            detail=f"scoring must be one of: {', '.join(SCORING_MULTIPLIERS.keys())}",
        )

    position = lookup_player_position(player)
    if not position:
        raise HTTPException(status_code=404, detail=f"Position not found for {player}")
    position = position.upper()
    if position not in FLEX_POSITIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported position {position}; supported FLEX positions: {', '.join(sorted(FLEX_POSITIONS))}",
        )

    markets = fetch_markets(player, position)
    result = run_flex_pipeline(player, position, scoring, markets, verbose=False)
    return result
