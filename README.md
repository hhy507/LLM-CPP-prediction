# LLM-CPP-prediction

Code and data for the manuscript **"Artificial-Intelligence-Driven Prediction and Design of Cell-Penetrating Peptides for Advanced Drug-Delivery Systems"** (Manuscript ID 1845913).

The framework derives interpretable biochemical rules for cell-penetrating peptides (CPPs) from two large language models (GPT-4o and DeepSeek), encodes each rule as a deterministic binary feature, combines these with conventional sequence descriptors (amino-acid frequency, k-mer, physicochemical properties, molecular fingerprints), and trains classical machine-learning classifiers (Random Forest, XGBoost, SVM, Logistic Regression) for CPP/non-CPP prediction. The best models are then used for de novo CPP generation, with downstream evaluation by established computational tools.

## Repository structure

```
.
├── data/
│   ├── origin/        # raw CPP/non-CPP sequences and distribution analysis
│   ├── seq/           # sequence + physicochemical feature analysis and figures
│   └── llm/           # LLM-derived rules and their 0/1 encodings
│       ├── 0424gpt_01/   # GPT-4o rule operationalisation (0424_gpt_01.py)
│       └── 0424ds_01/    # DeepSeek rule operationalisation (ds_01.py)
├── ML/                # classifier training/evaluation across feature sets
│   ├── spilt/         # train/test split scripts
│   ├── RF/ SVM/ XGB/ LR/   # per-algorithm × feature-set models and outputs
│   └── metrics/       # aggregated metrics, confusion matrices, figures
└── deno/              # de novo generation and re-evaluation
```

Feature-set folder names follow the convention used in the paper: `gpt` / `ds` (LLM rules),
`seq` (amino-acid frequency), `seq-kmer`, `seq-chem` (physicochemical), `seq-hot` (one-hot),
`seq-sm` (molecular fingerprints), and `gpt-seq` / `ds-seq` (integrated rule + frequency).

## Rule operationalisation

Each natural-language rule is implemented as an explicit, deterministic Python function, so the
conversion from an English rule to its 0/1 feature is unambiguous and reproducible:

- GPT-4o rules: `data/llm/0424gpt_01/0424_gpt_01.py`
- DeepSeek rules: `data/llm/0424ds_01/ds_01.py`

The complete 40-rule table (20 GPT-4o + 20 DeepSeek) is provided as Supplementary Table S1 in the manuscript.

## Dataset

The models are developed on the public **CPP924** benchmark (462 CPP / 462 non-CPP; all peptide
pairs share < 80% identity), derived from CPPsite 2.0. The data files needed to reproduce the
analyses are included under `data/` and the per-experiment `data/` subfolders.

## Requirements

```
python >= 3.9
numpy < 2
scikit-learn >= 1.3, < 1.6
xgboost
scipy < 1.14
pandas
openpyxl >= 3.1
matplotlib
```

Install with:

```bash
pip install "numpy<2" "scikit-learn>=1.3,<1.6" xgboost "scipy<1.14" pandas "openpyxl>=3.1" matplotlib
```

## Reproducing the analyses

The scripts read and write relative paths from the repository root. A typical workflow is:

1. `data/llm/0424gpt_01/0424_gpt_01.py` / `data/llm/0424ds_01/ds_01.py` — encode each peptide as a rule-compliance vector.
2. `ML/spilt/*.py` — create the stratified train/test splits.
3. `ML/<RF|SVM|XGB|LR>/<feature-set>/train_rf.py` then `evaluate.py` — train and evaluate each classifier × feature-set combination.
4. `deno/.../` — generate and re-evaluate de novo candidates.

> Note: some helper scripts contain relative input/output paths that may need to be adjusted to
> your local directory layout.

## Direct LLM classification baseline

`llm_direct_classification.py` reproduces the direct-LLM baseline requested in review: each of
the 185 held-out test peptides is given directly to a current-version LLM, which returns a binary
CPP/non-CPP label; the accuracy is compared with the ML models on the same split. API keys are read
**only** from environment variables.

```bash
export DEEPSEEK_API_KEY=...      # DeepSeek official API
export OPENROUTER_API_KEY=...    # OpenRouter (for openai/gpt-4o)
python llm_direct_classification.py          # full 185-sequence run
```

Outputs `llm_direct_classification_results.csv` (per-sequence predictions) and
`llm_direct_classification_summary.json` (metrics). On the held-out test set, direct LLM
classification (GPT-4o 0.854, DeepSeek-V3 0.773 accuracy) is less accurate than the ML models
(RF-Fre 0.946, RF-GPT-Fre 0.903), confirming the value of operationalising the LLM-derived rules
within a classical classifier.

## Citation

If you use this code, please cite the manuscript (Manuscript ID 1845913). Full citation details
will be added upon publication.

## License

Released under the MIT License (see `LICENSE`).
