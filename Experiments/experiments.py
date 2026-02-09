random_state = 42
n_jobs = 1

dict_hyper = {
    "log_reg": {
        "penalty": [None, 'l1', 'l2', 'elasticnet'],
        "C": [1.0, .5],
        "l1_ratio": [.0],
        "dual": [False],
        "fit_intercept": [True, False],
        "class_weight": [None, 'balanced'],
        "normalize": [True, False],
        "max_iter": [100],
        "random_state": [random_state],
        "n_jobs": [n_jobs],
    },
    "ridge": {
        "alpha": [1.],
        "fit_intercept": [True, False],
        "class_weight": [None, 'balanced'],
        "random_state": [random_state],
    },
    "EBM": {
        "max_bins": [1024],
        "max_interaction_bins": [64],
        "interactions": [0, "3x"],
        "validation_size": [0.15],
        "learning_rate": [0.015],
        "n_jobs": [n_jobs],
        "random_state": [random_state],
    },
    "FIGS": {
        "max_rules_mult": [1, .5],
        "max_trees": [None],
        "min_impurity_decrease": [.0],
        "random_state": [random_state],
        "max_features": [None],
        "max_depth": [None],
    },
    "LogisticGAM": {
        "terms": ['auto'],
        "max_iter": [100],
        "tol": [.0001],
        "fit_intercept": [True],
    },
    "TreeGAMCl":  {
        "n_boosting_rounds": [100],
        "max_leaf_nodes": [3],
        "reg_param": [0.0],
        "learning_rate": [0.01],
        "n_boosting_rounds_marginal": [0],
        "max_leaf_nodes_marginal": [2],
        "reg_param_marginal": [0.0],
        "fit_linear_marginal": [None],
        "boosting_strategy": ["cyclic"],
        "validation_frac": [0.15],
        "random_state": [random_state]
    },
    "FasterRisk": {

    }
}