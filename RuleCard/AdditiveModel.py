import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Union, Sequence, Callable, Dict
from joblib import Parallel, delayed
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss, precision_recall_curve
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

# Assuming FAST is available in the same directory or package
from FAST import FAST


# --- Helper Functions ---

def _sigmoid(x):
    return expit(x)


def _calc_loss(y, F, metric="logloss", sample_weight=None, epsilon=1e-12):
    """Unified loss calculation for scalar or vector F."""
    if metric == "logloss":
        p = np.clip(expit(F), epsilon, 1 - epsilon)
        loss = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    else:  # mse
        p = expit(F)
        loss = (y - p) ** 2

    if sample_weight is not None:
        return np.average(loss, weights=sample_weight, axis=0)
    return np.mean(loss, axis=0)


def _fit_tree_job(X_subset, residuals, max_depth, random_state, sample_weight=None):
    """Worker function for parallel tree fitting."""
    t = DecisionTreeRegressor(max_depth=max_depth, random_state=random_state)
    t.fit(X_subset, residuals, sample_weight=sample_weight)
    return t, t.predict(X_subset)


@dataclass
class Stage:
    features: Tuple[int, ...]
    tree: DecisionTreeRegressor
    kind: str
    score: float


# --- Training Context Manager ---

class _BoostingContext:
    """
    Manages transient state during fit() to avoid polluting the estimator's self.
    Handles caching, candidate generation, and finding the best split.
    """

    def __init__(self, model: 'AdditiveModel', X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray):
        self.model = model
        self.X = X
        self.y = y
        self.sw = sample_weight
        self.n_samples, self.n_features = X.shape

        # Caches
        self.uni_trees: List[DecisionTreeRegressor] = []
        self.uni_preds: Optional[np.ndarray] = None
        self.pair_trees: List[DecisionTreeRegressor] = []
        self.pair_preds: Optional[np.ndarray] = None
        self.pair_meta: List[Tuple[int, int]] = []  # List of (f1, f2)

        # State tracking
        self.used_uni = set()
        self.used_pairs = set()
        self.feats_in_pairs = set()

        # Feature Mapping for cache
        self.uni_map = {}
        self.pair_map = {}  # Maps (f1, f2) -> index in cache

    def init_fast_pairs(self, residuals, F):
        """Run FAST to identify candidate pairs."""
        if self.model.group_mode not in ("pairwise", "mixed"):
            return

        feat_names = getattr(self.model, "feature_names_in_", [f"f{i}" for i in range(self.n_features)])
        pairs, _ = FAST.run(
            self.X, residuals,
            feature_names=list(feat_names),
            bins=self.model.fast_bins,
            num_threads=self.model.fast_num_threads,
            min_support=self.model.fast_min_support,
            max_cells=self.model.fast_max_cells
        )
        # Filter duplicates and update metadata
        current_set = set(self.pair_meta)
        for p in pairs:
            p_tuple = tuple(sorted((p[0], p[1])))  # Store indices, not names from FAST wrapper if using numpy
            # Note: FAST.run output format in the simplified version depends on implementation.
            # Assuming it returns indices (i, j) based on input matrix X columns.
            if p_tuple not in current_set:
                self.pair_meta.append(p_tuple)

    def refresh_cache(self, residuals):
        """Re-fits trees on current residuals for all candidates."""

        # 1. Univariate
        candidates_uni = []
        if self.model.group_mode in ("univariate", "mixed"):
            # Filter valid features
            for i in range(self.n_features):
                if i in self.used_uni: continue
                if self.model.block_univariate_if_in_pair and i in self.feats_in_pairs: continue
                candidates_uni.append(i)

        if candidates_uni and self.model.precompute_univariate:
            results = Parallel(n_jobs=-1)(
                delayed(_fit_tree_job)(self.X[:, [i]], residuals, self.model.max_depth, self.model.random_state,
                                       self.sw)
                for i in candidates_uni
            )
            self.uni_trees, preds = zip(*results)
            self.uni_preds = np.column_stack(preds)
            self.uni_map = {idx: i for i, idx in enumerate(candidates_uni)}
        else:
            self.uni_preds = None  # Fallback to on-the-fly or no candidates

        # 2. Pairwise
        candidates_pair = []
        if self.model.group_mode in ("pairwise", "mixed"):
            for p in self.pair_meta:
                if not self.model.allow_pair_reuse and p in self.used_pairs: continue
                candidates_pair.append(p)

        if candidates_pair and self.model.precompute_pairs:
            results = Parallel(n_jobs=-1)(
                delayed(_fit_tree_job)(self.X[:, list(p)], residuals, self.model.max_depth, self.model.random_state,
                                       self.sw)
                for p in candidates_pair
            )
            self.pair_trees, preds = zip(*results)
            self.pair_preds = np.column_stack(preds)
            self.pair_map = {p: i for i, p in enumerate(candidates_pair)}
        else:
            self.pair_preds = None

    def find_best(self, F_current, residuals):
        """Finds the single best stage among all cached/non-cached candidates."""
        best_loss = np.inf
        best_stage = None

        # --- Check Univariate ---
        if self.uni_preds is not None:
            # Vectorized loss check
            # F_new = F + lr * pred. Calculate loss for all columns at once if possible or loop
            # For simplicity and memory, loop over columns but use precomputed preds
            for feat_idx, cache_idx in self.uni_map.items():
                loss = _calc_loss(self.y, F_current + self.model.lr * self.uni_preds[:, cache_idx],
                                  self.model.greedy_metric, self.sw)
                if loss < best_loss:
                    best_loss = loss
                    best_stage = Stage((feat_idx,), self.uni_trees[cache_idx], "uni", 0.0)

        # --- Check Pairwise ---
        if self.pair_preds is not None:
            for pair, cache_idx in self.pair_map.items():
                loss = _calc_loss(self.y, F_current + self.model.lr * self.pair_preds[:, cache_idx],
                                  self.model.greedy_metric, self.sw)
                if loss < best_loss:
                    best_loss = loss
                    # Note: We pass a clone or the tree itself.
                    best_stage = Stage(pair, self.pair_trees[cache_idx], "pair", 0.0)

        # Calculate gain score relative to base
        if best_stage:
            base_loss = _calc_loss(self.y, F_current, self.model.greedy_metric, self.sw)
            best_stage.score = base_loss - best_loss

            # Update usage state
            if best_stage.kind == 'uni':
                self.used_uni.add(best_stage.features[0])
            else:
                self.used_pairs.add(best_stage.features)
                self.feats_in_pairs.update(best_stage.features)

        return best_stage


# --- Main Estimator ---

class AdditiveModel(ClassifierMixin, BaseEstimator):
    def __init__(
            self,
            max_depth: int = 2,
            lr: float = 1.0,
            group_mode: str = "mixed",  # 'univariate', 'pairwise', 'mixed'
            n_stages: Optional[int] = None,
            greedy_metric: str = "logloss",
            threshold: float = 0.5,
            random_state: Optional[int] = None,
            epsilon: float = 1e-12,
            # Early Stopping
            early_stopping: bool = False,
            validation_fraction: float = 0.1,
            patience: int = 10,
            validation_X=None, validation_y=None,
            # Constraints
            allow_pair_reuse: bool = False,
            block_univariate_if_in_pair: bool = True,
            # FAST
            fast_bins: int = 16,
            fast_num_threads: int = 1,
            fast_min_support: int = 50,
            fast_max_cells: int = 1024,
            # Caching strategy
            cache_refresh_every: int = 5,
            fast_refresh_every: int = 10,
            precompute_univariate: bool = True,
            precompute_pairs: bool = True,
            verbose: int = 0,
            # Placeholders for compatibility
            selection: str = "greedy",
            feature_order=None,
            feature_importance_fn=None,
    ):
        self.max_depth = max_depth
        self.lr = lr
        self.group_mode = group_mode
        self.n_stages = n_stages
        self.greedy_metric = greedy_metric
        self.threshold = threshold
        self.random_state = random_state
        self.epsilon = epsilon
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.patience = patience
        self.validation_X = validation_X
        self.validation_y = validation_y
        self.allow_pair_reuse = allow_pair_reuse
        self.block_univariate_if_in_pair = block_univariate_if_in_pair
        self.fast_bins = fast_bins
        self.fast_num_threads = fast_num_threads
        self.fast_min_support = fast_min_support
        self.fast_max_cells = fast_max_cells
        self.cache_refresh_every = cache_refresh_every
        self.fast_refresh_every = fast_refresh_every
        self.precompute_univariate = precompute_univariate
        self.precompute_pairs = precompute_pairs
        self.verbose = verbose

        # State
        self.stages_: List[Stage] = []
        self.p0_: float = 0.5
        self.feature_names_in_ = None

    def fit(self, X, y, sample_weight=None):
        # 1. Data Prep (Minimal checks)
        X = np.asarray(X)
        y = np.asarray(y)
        if sample_weight is not None: sample_weight = np.asarray(sample_weight)

        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
        elif self.feature_names_in_ is None:
            self.feature_names_in_ = np.array([f"f{i}" for i in range(X.shape[1])])

        # 2. Validation Split
        if self.early_stopping:
            if self.validation_X is not None:
                X_tr, y_tr, sw_tr = X, y, sample_weight
                X_val, y_val = np.asarray(self.validation_X), np.asarray(self.validation_y)
            else:
                out = train_test_split(X, y, sample_weight, test_size=self.validation_fraction,
                                       random_state=self.random_state, stratify=y)
                X_tr, X_val, y_tr, y_val = out[0], out[1], out[2], out[3]
                sw_tr = out[4] if sample_weight is not None else None
        else:
            X_tr, y_tr, sw_tr = X, y, sample_weight
            X_val, y_val = None, None

        # 3. Initialization
        self.stages_ = []
        self.p0_ = np.clip(np.mean(y_tr), self.epsilon, 1 - self.epsilon)
        F_tr = np.full(len(y_tr), np.log(self.p0_ / (1 - self.p0_)))
        F_val = np.full(len(y_val), np.log(self.p0_ / (1 - self.p0_))) if X_val is not None else None

        # 4. Context Setup (Local state manager)
        ctx = _BoostingContext(self, X_tr, y_tr, sw_tr)

        # Initial FAST run
        ctx.init_fast_pairs(y_tr - expit(F_tr), F_tr)
        ctx.refresh_cache(y_tr - expit(F_tr))  # Initial cache build

        # 5. Boosting Loop
        max_stages = self.n_stages if self.n_stages else (X_tr.shape[1] + 50)  # Default cap if None
        best_val_score = -np.inf
        no_imp = 0

        for t in range(max_stages):
            residuals = y_tr - expit(F_tr)

            # Refresh strategies
            if t > 0:
                do_fast = self.fast_refresh_every and (t % self.fast_refresh_every == 0)
                do_cache = (t % self.cache_refresh_every == 0) or do_fast

                if do_fast: ctx.init_fast_pairs(residuals, F_tr)
                if do_cache: ctx.refresh_cache(residuals)

            # Find best
            stage = ctx.find_best(F_tr, residuals)
            if not stage:
                if self.verbose: print(f"No valid candidate found at stage {t}.")
                break

            # Update Model
            pred_tr = stage.tree.predict(X_tr[:, list(stage.features)])
            F_tr += self.lr * pred_tr
            self.stages_.append(stage)

            if self.verbose:
                print(f"Stage {t}: {stage.kind} {stage.features} Gain={stage.score:.4f}")

            # Early Stopping
            if self.early_stopping and X_val is not None:
                pred_val = stage.tree.predict(X_val[:, list(stage.features)])
                F_val += self.lr * pred_val

                # Check metric (AUC for simplicity here)
                curr_score = roc_auc_score(y_val, expit(F_val))
                if curr_score > best_val_score + 1e-5:
                    best_val_score = curr_score
                    no_imp = 0
                else:
                    no_imp += 1

                if no_imp >= self.patience:
                    print(f"Early stopping at stage {t}")
                    # Revert excess stages? Usually yes.
                    self.stages_ = self.stages_[:-no_imp]
                    break

        return self

    def predict_proba(self, X):
        X = np.asarray(X)
        F = np.full(X.shape[0], np.log(self.p0_ / (1 - self.p0_)))
        for stage in self.stages_:
            # Note: Explicit tuple cast for indices
            F += self.lr * stage.tree.predict(X[:, list(stage.features)])
        p = expit(F)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)


# --- Scorecard Class ---

@dataclass
class Scorecard:
    rules_df: pd.DataFrame
    base_points: int
    factor: float
    offset: float

    @staticmethod
    def from_model(model, X_train, y_train, min_support=10, PDO=50, score0=0, odds0=None):
        """Extracts rules and converts weights to points."""
        from sklearn.tree import _tree

        feature_names = model.feature_names_in_
        rules = []
        X_train = np.asarray(X_train)
        n = len(X_train)

        for i, stage in enumerate(model.stages_):
            tree = stage.tree.tree_

            def recurse(node, bounds):
                if tree.feature[node] != _tree.TREE_UNDEFINED:
                    # Feature index in the subset used by tree
                    local_feat = tree.feature[node]
                    # Map to global index
                    global_feat = stage.features[local_feat]
                    thr = tree.threshold[node]

                    # Branch Left (<=)
                    b_left = bounds.copy()
                    b_left.setdefault(global_feat, []).append(("<=", thr))
                    recurse(tree.children_left[node], b_left)

                    # Branch Right (>)
                    b_right = bounds.copy()
                    b_right.setdefault(global_feat, []).append((">", thr))
                    recurse(tree.children_right[node], b_right)
                else:
                    # Leaf
                    mask = np.ones(n, dtype=bool)
                    for feat_idx, conds in bounds.items():
                        col = X_train[:, feat_idx]
                        for op, val in conds:
                            if op == "<=":
                                mask &= (col <= val)
                            else:
                                mask &= (col > val)

                    support = np.sum(mask)
                    if support < min_support: return

                    # Get value
                    val = tree.value[node][0][0] * model.lr

                    # Format rule string
                    desc_parts = []
                    for feat_idx, conds in bounds.items():
                        fname = feature_names[feat_idx]
                        # Simplify bounds (min of <= and max of >)
                        ub = min([v for op, v in conds if op == "<="], default=np.inf)
                        lb = max([v for op, v in conds if op == ">"], default=-np.inf)

                        if lb == -np.inf:
                            desc_parts.append(f"{fname} <= {ub:.4f}")
                        elif ub == np.inf:
                            desc_parts.append(f"{fname} > {lb:.4f}")
                        else:
                            desc_parts.append(f"{lb:.4f} < {fname} <= {ub:.4f}")

                    rules.append({
                        "stage": i,
                        "description": " AND ".join(desc_parts) or "TRUE",
                        "bounds": bounds,
                        "weight": val,
                        "support": support
                    })

            recurse(0, {})

        df = pd.DataFrame(rules)

        # Scaling
        p0 = np.mean(y_train)
        odds0 = odds0 if odds0 else p0 / (1 - p0)
        factor = PDO / np.log(2)
        offset = score0 - factor * np.log(odds0)
        base_points = int(offset + factor * np.log(p0 / (1 - p0)))

        df['points'] = (df['weight'] * factor).astype(int)
        df = df[df['points'] != 0].copy()

        return Scorecard(df, base_points, factor, offset)

    def predict_scores(self, X):
        X = np.asarray(X)
        scores = np.full(X.shape[0], self.base_points)

        # Naive implementation (Vectorization possible via sparse matrix)
        for _, rule in self.rules_df.iterrows():
            mask = np.ones(X.shape[0], dtype=bool)
            for feat_idx, conds in rule['bounds'].items():
                col = X[:, feat_idx]
                for op, val in conds:
                    if op == "<=":
                        mask &= (col <= val)
                    else:
                        mask &= (col > val)
            scores[mask] += rule['points']

        return scores

    def predict_proba(self, X):
        S = self.predict_scores(X)
        logit = (S - self.offset) / self.factor
        p = expit(logit)
        return np.column_stack([1 - p, p])