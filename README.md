# RuleCard: Learning Operational Scorecards via Boosted Rule Extraction

Interpretable predictive models are crucial in high-impact and regulated settings, where accuracy must coexist with transparency and operational usability. 
Traditional scorecards are easy to deploy but limited in modeling nonlinearities and interactions. 
Supersparse linear scorecards improve performance and sparsity, yet rely on less intuitive feature-importance explanations. 
Meanwhile, flexible additive and tree-based interpretable models rarely translate into standardized integer scorecards for operational use.
We introduce RuleCard, a natively interpretable classifier that unifies these strengths. 
RuleCard learns an additive model on the log-odds scale through stagewise boosting of shallow univariate and selected pairwise trees. 
A deterministic extraction pipeline converts the trained ensemble into human-readable rules over numeric intervals or categorical sets, assigning each an integer point value. 
Empirical results demonstrate that RuleCard outperforms existing scorecard-based approaches and is competitive with interpretable state-of-the-art alternatives, producing compact and transparent scorecards that effectively balance robustness and practical deployability.


## Setup

### Using PyPI

```bash
  pip install RuleCard #TODO
```

### Manual Setup

```bash
git clone https://github.com/___/RuleCard
cd RuleCard
pip install -e .
```

Dependencies are listed in `requirements.txt`.


## Running the code

```python
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Experiments.readers import read_telco
from RuleCard.RuleCardGAM import RuleCardGAM

df = read_telco()
X = df.iloc[:, :-1].values
y = LabelEncoder().fit_transform(df.iloc[:, -1].values)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


rcgam = RuleCardGAM()
y_pred = rcgam.fit(X_train, y_train).predict(X_test)
print('RuleCardGAM:', classification_report(y_test, y_pred), end='\t')
```

Jupyter notebooks with examples on real datasets can be found in the `examples/` directory.


## Docs and reference


You can find the software documentation in the `/docs/` folder and 
a powerpoint presentation on Geolet can be found [here](http://example.org).
You can cite this work with
```
TODO
```


## Extending the algorithm

The original Geolet code, i.e., the code used for the experiments in the paper, is available in the /original_code branch.

The code in the main branch is a reimplementation that speeds up the execution time by about 7%.
 
