# RuleCard: Learning Operational Scorecards via Boosted Rule Extraction

Interpretable predictive models are crucial in high-impact and regulated settings, where accuracy must coexist with transparency and operational usability. 
Traditional scorecards are easy to deploy but limited in modeling nonlinearities and interactions. 
Supersparse linear scorecards improve performance and sparsity, yet rely on less intuitive feature-importance explanations. 
Meanwhile, flexible additive and tree-based interpretable models rarely translate into standardized integer scorecards for operational use.
We introduce RuleCard, a natively interpretable classifier that unifies these strengths. 
RuleCard learns an additive model on the log-odds scale through stagewise boosting of shallow univariate and selected pairwise trees. 
A deterministic extraction pipeline converts the trained ensemble into human-readable rules over numeric intervals or categorical sets, assigning each an integer point value. 
Empirical results demonstrate that RuleCard outperforms existing scorecard-based approaches and is competitive with interpretable state-of-the-art alternatives, 
producing compact and transparent scorecards that effectively balance robustness and practical deployability.

## Repository Structure

- `RuleCard/`: core implementation of `RuleCardGAM`, `ScoreCard`, feature-importance wrappers, and FAST utilities.
- `main.py`: minimal executable example on the Titanic dataset.
- `datasets/CLF/`: tabular classification datasets used by examples and experiments.
- `Experiments/`: benchmark runner, dataset readers, competitor wrappers, and result-analysis notebooks.
- `requirements.txt`: Python packages required by the implementation and experiments.

## Installation

RuleCard requires Python 3.10 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The code can be used directly from the repository root. For notebooks and documentation-related tooling, also install:

```bash
python -m pip install -r requirements.docs.txt
```

Some experiment competitors have additional build/runtime requirements, especially `pyfim`, FasterRisk/RiskSLIM-related code, and the external `benchmark.evaluation_utils` module imported by `Experiments/experiments.py`. 
Make sure these are available on `PYTHONPATH` before running the full benchmark grid.

## Quick Start

Run the minimal example:

```bash
python main.py
```

The script trains a `RuleCardGAM` classifier on `datasets/CLF/titanic.csv`, converts it into a `ScoreCard`, prints classification metrics, and displays the first scorecard rules.

The essential usage pattern is:

```python
from RuleCard.RuleCardGAM import RuleCardGAM
from RuleCard.ScoreCard import ScoreCard

model = RuleCardGAM(
    max_n_iter=10,
    patience=3,
    learning_rate=1.0,
    feature_order="feature_importance",
    top_k_features="sqrt",
    n_jobs=1,
    random_state=42,
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

scorecard = ScoreCard.from_rulecard(model, X_train)
scorecard_pred = scorecard.predict(X_test)
```

`RuleCardGAM` currently supports binary classification. Input features should be numeric; categorical variables should be encoded before fitting.

## Reproducing the Experiments

The experimental pipeline is implemented in `Experiments/experiments.py`.

1. Install the dependencies listed above.
2. Ensure the datasets expected by `Experiments/readers.py` are present under `datasets/CLF/`.
3. Review `dict_hyper` in `Experiments/experiments.py` to select the models and hyperparameter grids.
4. Review `all_datasets` in `Experiments/readers.py` to select the datasets.
5. Run the benchmark from the `Experiments` directory so that dataset paths resolve correctly:

```bash
cd Experiments
PYTHONPATH=.. python experiments.py
```

Results are written under:

```text
Experiments/res/<dataset>/<model>/<hyperparameter_hash>.csv
Experiments/res/<dataset>/<model>/<hyperparameter_hash>.pkl
Experiments/res/<dataset>/<model>/<hyperparameter_hash>.h5
```

For RuleCard runs, extracted rules are also saved as:

```text
Experiments/res/<dataset>/<model>/<hyperparameter_hash>.rules
```

The `__main__` block in `Experiments/experiments.py` contains temporary model-family filters used to run selected subsets of the grid. To reproduce the full paper grid, remove or adapt those `continue` filters and run all desired model names from `dict_hyper`.

## Analyzing Results

The notebooks in `Experiments/` aggregate and visualize benchmark outputs:

- `results_performance.ipynb`: predictive performance, model-complexity, and timing summaries.
- `results_calibration.ipynb`: calibration-oriented analysis.
- `datasets_info.ipynb`: dataset statistics.
- `Export ScoreCard viz.ipynb`: scorecard visualization/export utilities.

Open the notebooks after generating `Experiments/res/`:

```bash
jupyter notebook Experiments/results_performance.ipynb
```

## License

This repository is distributed under the license included in `LICENSE`.

## Citation

```bibtex
@inproceedings{placeholder_rulecard_2026,
  title     = {RuleCard: Learning Operational Scorecards via Boosted Rule Extraction},
  author    = {PLACEHOLDER AUTHOR LIST},
  booktitle = {PLACEHOLDER},
  year      = {2026},
  pages     = {PLACEHOLDER},
  doi       = {PLACEHOLDER},
  url       = {PLACEHOLDER}
}
```
