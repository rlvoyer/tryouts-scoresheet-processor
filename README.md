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

### Run the program

It expects that you have input directory with images of tryout scoresheets. Pass this as the first parameter to the program. Additionally, it expects the name of a directory to which it will write the output CSVs. This directory will be removed if one already exists. This is the second parameter to the program. For example:

Ensure that the Google Gemini API key that you acquired earlier is set in your environment:

```sh
export GEMINI_API_KEY=...
```

Then run it!

```sh
python process_scoresheets.py tryouts_images tryouts_scores
```

Be sure that input images are oriented correctly before loading.
