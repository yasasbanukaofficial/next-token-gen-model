# next token gen model

A simple next word prediction project built with TensorFlow and Keras. The full workflow lives in `notebooks/Predictor_v1.ipynb`, together with the saved outputs from the original run.

It reads a public fake and real news dataset, turns article text into numbered tokens, learns from 10 word windows, saves the trained model, and then demonstrates prediction and text generation.

## Flow by stage

### Stage 1 Data analysis

```mermaid
%%{init: {"theme": "dark", "themeVariables": {"background": "#0b1220", "primaryColor": "#1e293b", "primaryTextColor": "#ffffff", "primaryBorderColor": "#64748b", "textColor": "#ffffff", "lineColor": "#cbd5e1", "fontSize": "15px", "edgeLabelBackground": "#0b1220"}}}%%
flowchart LR
    A["News CSV URL"] --> B["Download with pandas"]
    B --> C["Load dataset"]
    C --> D["Print shape"]
    D --> E["Inspect first rows"]

    classDef source fill:#172554,stroke:#60a5fa,color:#ffffff;
    classDef process fill:#052e2b,stroke:#2dd4bf,color:#ffffff;
    classDef result fill:#064e3b,stroke:#34d399,color:#ffffff;

    class A source;
    class B,C,D process;
    class E result;
```

Recorded result: 6,335 rows and 3 columns. The columns are title, text, and label.

### Stage 2 Data processing

```mermaid
%%{init: {"theme": "dark", "themeVariables": {"background": "#0b1220", "primaryColor": "#1e293b", "primaryTextColor": "#ffffff", "primaryBorderColor": "#64748b", "textColor": "#ffffff", "lineColor": "#cbd5e1", "fontSize": "15px", "edgeLabelBackground": "#0b1220"}}}%%
flowchart LR
    A["Text column"] --> B["Join articles"]
    B --> C["Keep first 1000000 characters"]
    C --> D["Lowercase text"]
    D --> E["Extract word tokens"]
    E --> F["Find unique tokens"]
    F --> G["Map tokens to IDs"]

    classDef source fill:#172554,stroke:#60a5fa,color:#ffffff;
    classDef process fill:#052e2b,stroke:#2dd4bf,color:#ffffff;
    classDef result fill:#064e3b,stroke:#34d399,color:#ffffff;

    class A source;
    class B,C,D,E,F process;
    class G result;
```

This stage turns raw article text into a numbered vocabulary.

### Stage 3 Training data

```mermaid
%%{init: {"theme": "dark", "themeVariables": {"background": "#0b1220", "primaryColor": "#1e293b", "primaryTextColor": "#ffffff", "primaryBorderColor": "#64748b", "textColor": "#ffffff", "lineColor": "#cbd5e1", "fontSize": "15px", "edgeLabelBackground": "#0b1220"}}}%%
flowchart LR
    A["Token list"] --> B["Take 10 words"]
    B --> C["Next word as target"]
    C --> D["Slide window forward"]
    D --> E["Input words"]
    E --> F["X array"]
    C --> G["Next words"]
    G --> H["y array"]

    classDef source fill:#172554,stroke:#60a5fa,color:#ffffff;
    classDef process fill:#052e2b,stroke:#2dd4bf,color:#ffffff;
    classDef result fill:#064e3b,stroke:#34d399,color:#ffffff;

    class A source;
    class B,C,D,E,G process;
    class F,H result;
```

Recorded result: X shape is 169677 by 10 and y shape is 169677.

### Stage 4 Model building

```mermaid
%%{init: {"theme": "dark", "themeVariables": {"background": "#0b1220", "primaryColor": "#1e293b", "primaryTextColor": "#ffffff", "primaryBorderColor": "#64748b", "textColor": "#ffffff", "lineColor": "#cbd5e1", "fontSize": "15px", "edgeLabelBackground": "#0b1220"}}}%%
flowchart LR
    A["Vocabulary size"] --> B["Input for 10 words"]
    B --> C["Token embedding"]
    C --> D["First recurrent layer"]
    D --> E["Second recurrent layer"]
    E --> F["Output layer"]
    F --> G["Probability scores"]

    classDef source fill:#172554,stroke:#60a5fa,color:#ffffff;
    classDef process fill:#2e1065,stroke:#c084fc,color:#ffffff;
    classDef result fill:#064e3b,stroke:#34d399,color:#ffffff;

    class A source;
    class B,C,D,E,F process;
    class G result;
```

The model ends with a probability score for every known word.

### Stage 5 Training

```mermaid
%%{init: {"theme": "dark", "themeVariables": {"background": "#0b1220", "primaryColor": "#1e293b", "primaryTextColor": "#ffffff", "primaryBorderColor": "#64748b", "textColor": "#ffffff", "lineColor": "#cbd5e1", "fontSize": "15px", "edgeLabelBackground": "#0b1220"}}}%%
flowchart LR
    A["X and y arrays"] --> B["Compile model"]
    B --> C["Set batch size 128"]
    C --> D["Train 30 epochs"]
    D --> E["Record accuracy"]
    D --> F["Record loss"]

    classDef source fill:#172554,stroke:#60a5fa,color:#ffffff;
    classDef process fill:#2e1065,stroke:#c084fc,color:#ffffff;
    classDef result fill:#064e3b,stroke:#34d399,color:#ffffff;

    class A source;
    class B,C,D process;
    class E,F result;
```

Recorded result: first epoch accuracy 0.0747 and loss 7.0795. Final epoch accuracy 0.1592 and loss 5.9869.

### Stage 6 Saving

```mermaid
%%{init: {"theme": "dark", "themeVariables": {"background": "#0b1220", "primaryColor": "#1e293b", "primaryTextColor": "#ffffff", "primaryBorderColor": "#64748b", "textColor": "#ffffff", "lineColor": "#cbd5e1", "fontSize": "15px", "edgeLabelBackground": "#0b1220"}}}%%
flowchart LR
    A["Trained model"] --> B["Save Keras file"]
    A --> C["Save history pickle"]
    B --> D["keras_next_word_model.keras"]
    C --> E["history.p"]

    classDef source fill:#172554,stroke:#60a5fa,color:#ffffff;
    classDef process fill:#052e2b,stroke:#2dd4bf,color:#ffffff;
    classDef result fill:#064e3b,stroke:#34d399,color:#ffffff;

    class A source;
    class B,C process;
    class D,E result;
```

Artifacts are kept so training does not need to run again.

### Stage 7 Inference

```mermaid
%%{init: {"theme": "dark", "themeVariables": {"background": "#0b1220", "primaryColor": "#1e293b", "primaryTextColor": "#ffffff", "primaryBorderColor": "#64748b", "textColor": "#ffffff", "lineColor": "#cbd5e1", "fontSize": "15px", "edgeLabelBackground": "#0b1220"}}}%%
flowchart LR
    A["User phrase"] --> B["Tokenize text"]
    B --> C["Map words to IDs"]
    C --> D["Pad to 10 words"]
    D --> E["Model predicts"]
    E --> F["Sort probabilities"]
    F --> G["Top words"]
    G --> H["Append and repeat"]

    classDef source fill:#172554,stroke:#60a5fa,color:#ffffff;
    classDef process fill:#052e2b,stroke:#2dd4bf,color:#ffffff;
    classDef result fill:#064e3b,stroke:#34d399,color:#ffffff;

    class A source;
    class B,C,D,E,F,G process;
    class H result;
```

### How one prediction travels

```mermaid
%%{init: {"theme": "dark", "themeVariables": {"background": "#0b1220", "primaryTextColor": "#ffffff", "textColor": "#ffffff", "actorBkg": "#1e293b", "actorBorder": "#64748b", "actorTextColor": "#ffffff", "actorLineColor": "#cbd5e1", "signalColor": "#cbd5e1", "signalTextColor": "#ffffff", "labelBoxBkgColor": "#1e293b", "labelBoxBorderColor": "#64748b", "labelTextColor": "#ffffff"}}}%%
sequenceDiagram
    participant U as You
    participant N as Notebook
    participant M as Trained model

    U->>N: Type a phrase
    N->>N: Tokenize and map words to IDs
    N->>M: Send 10 word input
    M-->>N: Probabilities over vocabulary
    N-->>U: Show ranked next words
```

## Example output

**Top five predictions after "I will"**

1. **be**
2. **have**
3. **help**
4. **only**
5. **take**

**Generated continuation**

> **I will have killed a syrian turkish fighters forces and the united new minister old line on**

The generated sentence is rough, but it demonstrates the full loop: predict one word, append it, and predict again. The creativity setting controls how many top candidates are considered, so higher values give more varied output.

## Tech Stack

**Python** 3.12

**TensorFlow and Keras** model building and training

**NumPy and pandas** data handling

**NLTK** tokenization

**uv** environment and dependency management

## Project Structure

```text
README.md                          this documentation
pyproject.toml                     project metadata and dependencies
uv.lock                            locked dependency versions
Python version pin                 Python version used by uv
notebooks/Predictor_v1.ipynb       full notebook workflow
models/keras_next_word_model.keras trained Keras model
models/history.p                   pickled training history
```

The dataset is downloaded from a URL inside the notebook and is not stored in this repository.

## Setup

```bash
uv sync
```

Dependencies are declared in `pyproject.toml` and locked in `uv.lock`. There is no `requirements.txt`.

## Run the notebook

```bash
uv run jupyter notebook notebooks/Predictor_v1.ipynb
```

Run the cells in order. Data analysis and preparation are quick. The training cell takes about 90 minutes on the original machine.

## Dataset credit

The news dataset used by the notebook is the Fake and Real News Dataset shared by lutzhamel on GitHub. Thanks to the provider for making the data publicly available.

Source: https://github.com/lutzhamel/fake-news/blob/master/data/fake_or_real_news.csv

Copyright © 2026 Yasas Banu · [GitHub](https://github.com/yasasbanukaofficial) · [LinkedIn](https://www.linkedin.com/in/yasasbanukagunasena/) · [Website](https://www.yasasbanuka.tech/)
