# Tryouts Scoresheet Processor

The intent of this little program is to take images of ultimate tryouts scoresheets and convert them into CSVs.

## How to run

Before you do anything else, you'll want to [acquire a Google Gemini API Key](https://aistudio.google.com/apikey).

First, it's best to create a virtual environment. Here are two (among many) ways to do that.

### Install uv (optional)

```sh
curl -LsSf https://astral.sh/uv/install.sh | less
```

### Create a virtual environment

You'll want to create a virtual environment in the root of this project.

With uv:

```sh
uv venv --python 3.13
```

Or with system Python:

```sh
python -m venv .venv
```

### Activate the virtual environment

```sh
source .venv/bin/activate
```

### Install dependencies

With uv:

```sh
uv sync
```

With pip:

```sh
pip install .
```

### Run it

There are 2 programs here:
1. process_scoresheets.py, and
2. combine_scoresheets.py

The two are intended to run as a pipeline.

process_scoresheets.py takes as input a folder of scoresheet images and outputs a folder of CSVs.

combine_scoresheets.py takes as input a folder of CSVs and a player group CSV and combines them into a single CSV where a player's scores are aggregated.

#### Process scoresheets

To run the pipeline, you must have an input directory with images of tryout scoresheets. Pass this as the first parameter to process_scoresheets.py. The second parameter to the program is the name of a directory where the output will be written. (If this directory exists, it will be overwritten). For example.

Ensure that the Google Gemini API key that you acquired earlier is set in your environment:

```sh
export GEMINI_API_KEY=...
```

Be sure that input images are oriented correctly before loading. Then run it!

```sh
python process_scoresheets.py tryouts_images tryouts_scores
```

#### Combine scoresheets

The combine_scoresheets.py program requires as input an input folder of scoresheet CSVs, a name for the CSV where the combined output should be written, and a player group file which indicates evaluation groups for each of the players. (The evaluation group file is solely as the basis for sorting the output.)

You can run combine_scoresheets.py like so:

```sh
python combine_scoresheets.py \
  --input-folder scoresheet_csvs \
  --output-file combined_scores.csv \
  --player-group-file gx_groups.csv
INFO     2025-02-23 18:13:25 Successfully read scoresheet_csvs/kelly.csv
INFO     2025-02-23 18:13:25 Successfully read scoresheet_csvs/robo.csv
INFO     2025-02-23 18:13:25 Successfully wrote scores from 2 scoresheets to combined_scores.csv
INFO     2025-02-23 18:13:25 Processed 2 evaluator files with 16 unique players
```
