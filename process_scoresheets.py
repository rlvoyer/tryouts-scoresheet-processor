import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from google import genai
from PIL import Image
from pydantic import BaseModel

from utils import logger


class Row(BaseModel):
    player_name: str
    scores: List[Optional[float]]

EXPECTED_COLUMNS = [
    "Throwing",
    "Catching",
    "Athleticism",
    "Defense",
    "Ultimate IQ / Decision Making",
    "Coachability / Intangibles"
]

def is_valid_score(score: float | None) -> bool:
    if score is None:
        return True

    return 1 <= score <= 5 or score == -1

def _validate_score(score: Any) -> Optional[float]:
    if score == "-1" or score is None:
        return None

    if not isinstance(score, (int, float)) and not is_valid_score(score):
        raise ValueError(f"Invalid score {score}")

    return float(score)

def _validate_scores(evaluator_name: str, player_name: str, row: Any) -> List[Optional[float]]:
    if "scores" not in row:
        raise ValueError(f"No scores in extracted row: {row}")

    scores = row["scores"]
    
    if len(scores) != len(EXPECTED_COLUMNS):
        log_msg = f"Evaluator {evaluator_name} is missing scores for player {player_name} -- please add them manually."
        logger.warning(log_msg)
        scores: List[Optional[float]] = [None] * len(EXPECTED_COLUMNS)
        return scores

    try:
        scores = [_validate_score(score) for score in scores]
    except:
        log_msg = f"Error extracting scores for {player_name} for evaluator {evaluator_name} -- please add them manually."
        logger.warning(log_msg)
        scores: List[Optional[float]] = [None] * len(EXPECTED_COLUMNS)

    return scores

def _validate_player_name(evaluator_name: str, row: Dict[str, Any]) -> str:
    if "player_name" not in row:
        raise ValueError(f"No player name in extracted row: {row}")

    player_name = row["player_name"]

    if not isinstance(player_name, str):
        log_msg = f"Error extracting player name ({type(player_name)}/ {player_name}) for evaluator {evaluator_name}"
        raise ValueError(log_msg)

    return player_name

def _validate_row(evaluator_name: str, row: Dict[str, Any]) -> Row:
    try:
        player_name = _validate_player_name(evaluator_name, row)
    except Exception as e:
        log_msg = f"Error extracting player name from row {row} for evaluator {evaluator_name} -- please add them manually."
        logger.warning(log_msg)
        player_name = ""

    try:
        scores = _validate_scores(evaluator_name, player_name, row)
    except Exception as e:
        log_msg = f"Error extracting scores from row {row} for evaluator {evaluator_name} -- please add them manually."
        logger.warning(log_msg)
        scores: List[Optional[float]] = [None] * len(EXPECTED_COLUMNS)

    return Row(player_name=player_name, scores=scores)

def analyze_scoresheet(image_path: Path | str) -> List[Row]:
    """
    Analyze a scoresheet image using Gemini and return structured player data
    
    Args:
        image_path: Path to the image file
        
    Returns:
        List of tuples containing player name and their scores
        Each score is a list of floats representing comma-separated values
    
    Raises:
        ValueError: If Gemini's response cannot be parsed into the expected format
    """
    image: Image.Image = Image.open(image_path)
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    prompt: str = """
    Your task is to analyze a spreadsheet with manually entered numeric values.

    The spreadsheet contains player names and numeric scores along the following dimensions:
    Throwing, Catching, Athleticism, Defense, Ultimate IQ / Decision Making, Coachability / Intangibles
    All scores should numbers be in the range of 1-5.

    Analyze this spreadsheet and return the data as a JSON array where each element is:
    {
        "player_name": string,
        "scores": number[]  // Array of scores
    }

    Example output:
    [
        {
            "player_name": "John Smith",
            "scores": [4.0, 3.0, 5.0, 4.0, 3.0, 4.0]
        }
    ]

    Every player (row) included in the input should be included in the output.

    If a cell is missing a value, don't hallucinate one. It's better for a cell to be empty than to have an incorrect value.

    Return only valid JSON, no additional text.
    """
    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image],
            config={
                'response_mime_type': 'application/json',
                'response_schema': list[Row],
            }
        )
    except Exception as e:
        raise ValueError(f"Gemini API call failed: {e}")

    evaluator_name = parse_evaluator_name_from_filename(str(image_path))
    
    try:
        data: List[Dict[str, Any]] = json.loads(response.text)
        return [_validate_row(evaluator_name, row) for row in data]

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"Failed to parse Gemini response: {e}\nResponse was: {response.text}")

def is_valid_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate the DataFrame before saving to CSV.
    """
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    duplicates = df[df["Player"].duplicated()]["Player"]
    if not duplicates.empty:
        raise ValueError(f"Found duplicate player names: {duplicates.tolist()}")

    return True
    
def create_dataframe(players: List[Row]) -> pd.DataFrame:
    """
    Convert List[PlayerRow] to a pandas DataFrame.
    """
    rows = [] # type: List[Dict[str, Any]]
    for player in players:
        record = {"Player": player.player_name} # type: Dict[str, Any]
        for idx, score in enumerate(player.scores):
            dimension_name = EXPECTED_COLUMNS[idx]
            record[dimension_name] = score
        rows.append(record)

    return pd.DataFrame(rows)

def parse_evaluator_name_from_filename(filename: Path | str) -> str:
    """
    Extract evaluator name from filename
    
    Args:
        filename: Path or string like "john_smith.scoresheet.jpg"
        
    Returns:
        Evaluator name (e.g. "john_smith")
        
    Raises:
        ValueError: If filename doesn't contain at least one period
    """
    basename = os.path.basename(str(filename))
    parts = basename.split(".", 1)

    if len(parts) < 2:
        raise ValueError(f"Invalid filename format: {filename}. Expected evaluator_name.rest_of_name.extension")

    return parts[0]
    
def process_scoresheets(image_folder: Path | str, output_folder: Path | str) -> None:
    """
    Process all scoresheet images and output as CSVs to folder.
    """    
    for filename in os.listdir(image_folder):
        if filename.endswith(('.jpg', '.png', '.jpeg')):
            image_path = os.path.join(image_folder, filename)
        else:
            logger.warning(f"Input file {filename} has unexpected extension -- skipping")
            continue

        evaluator_name = parse_evaluator_name_from_filename(filename)
        output_path = os.path.join(output_folder, f"{evaluator_name}.csv")

        try:
            players = analyze_scoresheet(image_path)
        except Exception as e:
            log_msg = f"Exception occurred while parsing scoresheet from {evaluator_name}: {e}"
            logger.exception(log_msg)
            logger.error("Skipping -- please enter data manually.")
            continue

        scores_df = create_dataframe(players)

        try:
            if is_valid_dataframe(scores_df):
                scores_df.to_csv(output_path, index=False)
        except Exception as e:
            logger.exception(f"Failed to generate valid CSV: {e}")
            logger.error("Skipping -- please enter data manually.")

if __name__ == "__main__":
    image_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(image_folder):
        logger.error(f"The specified image folder does not exist: {image_folder}")
        sys.exit(1)

    if os.path.exists(output_folder):
        logger.warning(f"The specified output folder {output_folder} already exists; removing it.")

        try:
            shutil.rmtree(output_folder)
        except PermissionError:
            logger.error(f"Unable to delete and recreate output folder {output_folder}: permission denied.")
            sys.exit(1)

    try:
        os.mkdir(output_folder)
    except Exception as e:
        logger.exception(f"Unable to create output folder {output_folder}: {e}")
        sys.exit(1)

    process_scoresheets(image_folder, output_folder)
