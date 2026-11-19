# next-token-gen-model

A simple LSTM project that predicts the next word in a news article. The work lives in one notebook; the folder layout comes from the `cookiecutter-data-science` template.

## What it does

The notebook downloads a public fake/real news CSV, joins the article text, tokenizes it, maps every unique word to an ID, and builds sliding windows of 10 words. Each window is the input, and the following word is the target. A Keras model with an embedding layer and two LSTM layers learns to predict that target from the vocabulary.

## Folder layout

```text
data/           raw, interim, processed, and external data folders
models/         saved Keras model and training history
notebooks/      Predictor_v1.ipynb
references/     notes or reference material
reports/        figures and write-ups
```

## Setup

```bash
uv sync
```

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. There is no `requirements.txt`.

## Run

Open `notebooks/Predictor_v1.ipynb` and run the cells in order. Training takes about 90 minutes on the original machine, so no cells were executed while creating this repository.

## Saved outputs

After training, the notebook writes `keras_next_word_model.keras` and `history.p` into `models/`.
