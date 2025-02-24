import argparse
import sys
from pathlib import Path

import pandas as pd

from process_scoresheets import DEFAULT_EVAL_COLUMNS
from utils import logger


def validate_all_players(original_df: pd.DataFrame, aggregated_df: pd.DataFrame) -> bool:
    original_players = set(original_df['Player'].unique())
    final_players = set(aggregated_df['Player'])
    return original_players == final_players

def combine_scores(input_folder: Path, output_file: Path, player_group_lookup_file: Path, evaluation_columns: list[str], include_header: bool) -> None:
    """
    Combine individual evaluator CSVs into a single aggregated CSV
    
    Args:
        input_folder: Directory containing individual evaluator CSVs
        output_file: Path for combined output CSV
        player_group_lookup_file: Path for a 2-column CSV of player evaluation groups
        evaluation_columns: List of column names to process
        include_header: Whether to include a header row in the output
    """
    all_dfs = []
    for csv_file in input_folder.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file)
            logger.info(f"Successfully read {csv_file}")
            all_dfs.append(df)
        except:
            logger.error(f"Failed to read {csv_file}", exc_info=True)
            continue

    if len(all_dfs) == 0:
        raise ValueError(f"No valid CSV files found in {input_folder}")

    combined_df = pd.concat(all_dfs)

    aggregated_df = combined_df.groupby('Player').agg({
        col: lambda x: ','.join(x.dropna().astype(str))
        for col in evaluation_columns
    }).reset_index()

    aggregated_df = combined_df.groupby('Player').agg({
        col: 'mean'
        for col in evaluation_columns
    }).reset_index()

    # NOTE: this should never happen, so this is very defensive...
    if not validate_all_players(combined_df, aggregated_df):
        logger.error(f"Some players were lost during aggregation")
        sys.exit(1)

    player_group_lookup_df = pd.read_csv(player_group_lookup_file)
    merged_df = aggregated_df.merge(player_group_lookup_df, on="Player", how="left")
    sorted_df = merged_df.sort_values(['Group', 'Player']).drop('Group', axis=1)

    sorted_df.to_csv(output_file, index=False, header=include_header)
    logger.info(f"Successfully wrote scores from {len(all_dfs)} scoresheets to {output_file}")
    logger.info(f"Processed {len(all_dfs)} evaluator files with {len(aggregated_df)} unique players")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine individual evaluator scoresheets into a single aggregated CSV"
    )

    parser.add_argument(
        "--input-folder",
        type=Path,
        required=True,
        help="Directory containing individual evaluator CSVs"
    )

    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path for output CSV file"
    )

    parser.add_argument(
        "--player-group-file",
        type=Path,
        required=True,
        help="CSV file containing player evaluation groups"
    )

    parser.add_argument(
        "--evaluation-columns",
        type=str,
        nargs="+",
        default=DEFAULT_EVAL_COLUMNS,
        help=f"Space-separated list of evaluation criteria (default: {' '.join(DEFAULT_EVAL_COLUMNS)})"
    )

    parser.add_argument(
        "--include-header-row",
        action="store_true",
        help="Whether to include a header row in the combined output CSV"
    )

    args = parser.parse_args()

    if not args.input_folder.exists():
        parser.error(f"Input folder does not exist: {args.input_folder}")

    args.output_file.parent.mkdir(parents=True, exist_ok=True)

    if not args.player_group_file.exists():
        parser.error(f"Player group input file does not exist: {args.input_folder}")

    return args

if __name__ == "__main__":
    args = parse_args()

    combine_scores(args.input_folder, args.output_file, args.player_group_file, args.evaluation_columns, args.include_header_row)
