import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
from google import genai
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PlayerName = str
PlayerRow = Tuple[PlayerName, List[float]]

EXPECTED_COLUMNS = [
    "Throwing",
    "Catching",
    "Athleticism",
    "Defense",
    "Ultimate IQ / Decision Making",
    "Coachability / Intangibles"
]

COLUMN_NAME_ID_MAP = {column_name: idx for (idx, column_name) in enumerate(EXPECTED_COLUMNS)}

def is_valid_score(score: float | None) -> bool:
    if score is None:
        return True

    return 1 <= score <= 5

def analyze_scoresheet(image_path: Path | str) -> List[PlayerRow]:
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
    client = genai.Client(api_key="GEMINI_API_KEY")
    
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

    Include a null for cells with no values.

    Return only valid JSON, no additional text.
    """

    try:
        response = client.models.generate_content(model="gemini-2.0-flash", contents=[prompt, image])
    except Exception as e:
        raise ValueError(f"Gemini API call failed: {e}")

    try:
        data = json.loads(response.text)
        
        players: List[PlayerRow] = [
            (
                entry["player_name"],
                entry["scores"]
            )
            for entry in data
        ]
        
        for name, scores in players:
            if not isinstance(name, str):
                raise ValueError(f"Invalid player name type: {type(name)}")

            if len(scores) != len(EXPECTED_COLUMNS):
                raise ValueError(f"Expected 6 score columns, got {len(scores)}")

            for score in scores:
                if score is None:
                    continue

                if not isinstance(score, (int, float)) and not is_valid_score(score):
                    raise ValueError(f"Invalid score {score}")

        return players
        
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        raise ValueError(f"Failed to parse Gemini response: {e}\nResponse was: {response.text}")

def is_valid_dataframe(df: pd.DataFrame) -> bool:
    """
    Validate the DataFrame before saving to CSV.
    """
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Check for duplicate player names
    duplicates = df[df["Player"].duplicated()]["Player"]
    if not duplicates.empty:
        raise ValueError(f"Found duplicate player names: {duplicates.tolist()}")
    
def create_dataframe(players: List[PlayerRow]) -> pd.DataFrame:
    """
    Convert List[PlayerRow] to a pandas DataFrame.
    """
    rows = [] # type: List[Dict[str, Any]]
    for name, scores in players:
        row = {"Player": name} # type: Dict[str, Any]
        for idx, score in enumerate(scores):
            dimension_name = EXPECTED_COLUMNS[idx]
            row[dimension_name] = score
        rows.append(row)

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
    name = str(filename)
    parts = name.split(".", 1)

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

        players = analyze_scoresheet(image_path)
        scores_df = create_dataframe(players)
        output_filename = parse_evaluator_name_from_filename(filename)
        output_path = os.path.join(output_folder, output_filename)

        try:
            if is_valid_dataframe(df):
                scores_df.to_csv(output_path, index=False)
        except Exception as e:
            logger.exception(f"Failed to generate valid dataframe: {e}")

if __name__ == "__main__":
    image_folder = sys.argv[1]
    output_folder = sys.argv[2]

    if not os.path.exists(image_folder):
        logger.error(f"The specified image folder does not exist: {image_folder}")
        sys.exit(1)

    if os.path.exists(output_folder):
        logger.error(f"The specified output folder already exists: {output_folder}")
        sys.exit(1)

    os.mkdir(output_folder)

    process_scoresheets(image_folder, output_folder)
