"""
Compare monotonicity (Kendall tau + PDP) for a binary classification task
between an original dataset and the predictions of an MLP on a test set.

Usage:
    python monotonicity_binary.py original.csv test_results.csv
"""

import argparse
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import contextlib
import traceback
import os
from scipy.stats import kendalltau
from sklearn.inspection import partial_dependence
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from train import prepare_data


ROBOT_PROFILES = ["comfort-blind", "comfort-caring", "complacent", "time-efficient"]
MODELS = ["normal", "hidden_5", "hidden_5_fc", "hidden_10", "hidden_10_fc"]

# EDIT: path to the original CSV of each profile
ORIGINAL_CSVS = {
    "comfort-blind":  "data/argumentation_QE_cb.csv",
    "comfort-caring": "data/argumentation_QE_cc.csv",
    "complacent":     "data/argumentation_QE_complacent.csv",
    "time-efficient": "data/argumentation_QE_te.csv",
}

DROP_COLUMNS = ["run", "iteration", "robot_profile", "users_before_robot",
                "elevator_capacity", "case", "model", "timeout", "inside_comfort",
                "breaking_queue", "robot_urgency", "number_users_before_robot",
                "number_people_in_elevator", "robot_time_waiting", "people_with_priority"]
TARGET = "target"

# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def positive_class_proba(model, X):
    """Return P(class = 1) for any sklearn-like classifier."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    # fall back to decision function squashed by sigmoid
    d = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-d))


def kendall_per_feature(df, feature_cols, score_col):
    """Kendall tau between each feature and a continuous score column."""
    results = {}
    y = df[score_col].values
    for f in feature_cols:
        x = df[f].values
        mask = ~(np.isnan(x) | np.isnan(y))
        if mask.sum() < 3:
            results[f] = (np.nan, np.nan)
            continue
        tau, p = kendalltau(x[mask], y[mask])
        results[f] = (tau, p)
    return results


# def print_kendall_table(name, results):
#     print(f"\n=== Kendall tau — {name} ===")
#     print(f"{'feature':<30}{'tau':>10}{'p-value':>12}{'direction':>14}")
#     for f, (tau, p) in results.items():
#         if np.isnan(tau):
#             direction = "n/a"
#         elif tau > 0:
#             direction = "increasing"
#         elif tau < 0:
#             direction = "decreasing"
#         else:
#             direction = "flat"
#         print(f"{f:<30}{tau:>10.4f}{p:>12.4g}{direction:>14}")


# def compare(k_orig, k_test):
#     print("\n=== Kendall sign agreement (original vs MLP) ===")
#     print(f"{'feature':<30}{'orig':>10}{'test':>10}{'agree':>10}")
#     for f in k_orig:
#         t1, t2 = k_orig[f][0], k_test[f][0]
#         agree = ("yes" if (not np.isnan(t1) and not np.isnan(t2)
#                            and np.sign(t1) == np.sign(t2)) else "NO")
#         print(f"{f:<30}{t1:>10.4f}{t2:>10.4f}{agree:>10}")

def print_kendall_table(name, results):
    print(f"\n=== Kendall tau — {name} ===")
    print(f"{'feature':<30}{'tau':>10}{'p-value':>12}{'direction':>14}")
    for f, (tau, p) in results.items():
        if np.isnan(tau):
            direction = "n/a"
        elif tau > 0:
            direction = "increasing"
        elif tau < 0:
            direction = "decreasing"
        else:
            direction = "flat"
        print(f"{f:<30}{tau:>10.4f}{p:>12.4g}{direction:>14}")


def compare(k_orig, k_test):
    print("\n=== Kendall sign agreement (original vs MLP) ===")
    print(f"{'feature':<30}{'orig':>10}{'test':>10}{'agree':>10}")
    for f in k_orig:
        t1, t2 = k_orig[f][0], k_test[f][0]
        agree = ("yes" if (not np.isnan(t1) and not np.isnan(t2)
                           and np.sign(t1) == np.sign(t2)) else "NO")
        print(f"{f:<30}{t1:>10.4f}{t2:>10.4f}{agree:>10}")

# ----------------------------------------------------------------------
# PDP wrapper that always returns P(class=1)
# ----------------------------------------------------------------------
# class ProbaWrapper:
#     """Sklearn-compatible regressor that outputs P(class=1).

#     This lets us reuse sklearn.inspection.partial_dependence directly.
#     """
#     def __init__(self, clf):
#         self.clf = clf
#         self._estimator_type = "regressor"

#     def fit(self, X, y):
#         self.clf.fit(X, y)
#         return self

#     def predict(self, X):
#         return positive_class_proba(self.clf, X)

class ProbaWrapper(ClassifierMixin, BaseEstimator):
    def __init__(self, model=None):
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        self.classes_ = self.model.classes_
        return self

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

def compute_pdp(model, X, feature_cols, grid_resolution=50):
    pdp = {}
    for i, f in enumerate(feature_cols):
        r = partial_dependence(
            model, X, features=[i],
            grid_resolution=grid_resolution,
            kind="average",
        )
        pdp[f] = (r["grid_values"][0], r["average"][0])
    return pdp


def plot_pdp(pdp_orig, pdp_test, feature_cols, out_path="pdp_comparison.png"):
    n = len(feature_cols)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, feature_cols):
        g1, a1 = pdp_orig[f]
        g2, a2 = pdp_test[f]
        ax.plot(g1, a1, label="Original (surrogate)", linewidth=2)
        ax.plot(g2, a2, label="MLP (test)", linewidth=2, linestyle="--")
        ax.set_title(f)
        ax.set_xlabel(f)
        ax.set_ylabel("P(class=1)")
        ax.grid(alpha=0.3)
        ax.set_ylim(0, 1)
    axes[0].legend()
    for ax in axes[len(feature_cols):]:
        ax.axis("off")
    fig.suptitle("PDP of P(class=1): original vs MLP", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved PDP figure to {out_path}")
    plt.show()


def fix_dataframe(df):
    # --- Fix types if needed ---
    if isinstance(df["users_before_robot"].iloc[0], str):
        df["users_before_robot"] = df["users_before_robot"].apply(ast.literal_eval)

    bool_cols = ["next_time_over_timeout", "people_with_impairments", "waiting_users_did_not_enter"]
    for col in bool_cols:
        if df[col].dtype == object:
            df[col] = (
                df[col].astype(str).str.strip().str.lower()
                .map({"true": True, "false": False, "1": True, "0": False})
            )
    return df

# --- Apply prepare_data and build flat rows ---
def run_prepare(row, dataset_name, original=False):
    context, base_scores, target = prepare_data(row, dataset_name, original=original)
    return pd.Series({**context, **base_scores, "target": target})



# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
# def main():

#     robot_profile = "comfort-blind"
#     original_csv = "data/argumentation_QE_cb.csv"
#     model_path = "results/" + robot_profile + "/hidden_5"
#     test_csv = model_path+"/combined_decision_log.csv"
#     drop_columns = ["run","iteration","robot_profile","users_before_robot", 
#                     "elevator_capacity", "case", "model", "timeout", "inside_comfort", "breaking_queue", "robot_urgency", 
#                     "number_users_before_robot", "number_people_in_elevator", "robot_time_waiting", "people_with_priority"]

#     output_path = model_path + "/plots/"

#     output_model = output_path + "attributions.png"

#     target_test = "target"

#     df_orig = pd.read_csv(original_csv)
#     df_test = pd.read_csv(test_csv)

#     df_test = df_test[df_test["model"]== "comfort-blind trained"]

#     # df_orig["decision"] = df_orig["decision"].map({"enter": 1, "no enter": 0})
#     # df_test["decision"] = df_test["decision"].map({"enter": 1, "no enter": 0})

#     # Convert values to base scores as inputs. 

#     df_orig = fix_dataframe(df_orig)
#     df_test = fix_dataframe(df_test)

#     df_orig = df_orig.apply(
#         lambda r: run_prepare(r, dataset_name=robot_profile, original=False),
#         axis=1,
#     ).reset_index(drop=True)

#     df_test = df_test.apply(
#         lambda r: run_prepare(r, dataset_name=robot_profile, original=False),
#         axis=1,
#     ).reset_index(drop=True)



#     feature_cols = [c for c in df_orig.columns
#                     if c not in drop_columns
#                     and c != target_test
#                     and c in df_test.columns]

#     # ---- Original: fit a surrogate classifier, use P(class=1) ----
#     X_orig = df_orig[feature_cols].values
#     y_orig = df_orig[target_test].values

#     # Gradient boosting is a good, well-calibrated default surrogate
#     base = GradientBoostingClassifier(random_state=0)
#     surrogate_orig = ProbaWrapper(base).fit(X_orig, y_orig)

#     df_orig_score = df_orig.copy()
#     df_orig_score["_p1"] = surrogate_orig.predict(X_orig)

#     # ---- Test: assume target_test already holds P(class=1) from the MLP ----
#     #     If it instead holds 0/1 labels, we fit a surrogate on the test
#     #     predictions too (see note below).
#     X_test = df_test[feature_cols].values
#     y_test = df_test[target_test].values

#     if np.array_equal(np.unique(y_test), [0, 1]) or set(np.unique(y_test)) <= {0, 1}:
#         # It's a hard label — fit a surrogate classifier for PDP to work
#         print("Test column looks like hard 0/1 labels; fitting a surrogate "
#               "classifier on it for PDP/Kendall.")
#         surrogate_test = ProbaWrapper(
#             GradientBoostingClassifier(random_state=0)
#         ).fit(X_test, y_test)
#         df_test_score = df_test.copy()
#         df_test_score["_p1"] = surrogate_test.predict(X_test)
#     else:
#         # Already a probability — use directly for Kendall, but wrap the
#         # original features + probability in a 1-feature regressor so PDP
#         # can be computed. Easiest: fit a tiny surrogate on (X, proba).
#         print("Test column looks like a probability; using it directly.")
#         surrogate_test = ProbaWrapper(
#             GradientBoostingClassifier(random_state=0)
#         ).fit(X_test, (y_test > 0.5).astype(int))
#         df_test_score = df_test.copy()
#         df_test_score["_p1"] = y_test
#         # For PDP we want to model the probability surface; fit a regressor
#         # on (X, probability) so partial_dependence works.
#         from sklearn.ensemble import GradientBoostingRegressor
#         reg = GradientBoostingRegressor(random_state=0).fit(X_test, y_test)

#         class _RegWrapper:
#             _estimator_type = "regressor"
#             def predict(self, X): return reg.predict(X)
#         surrogate_test = _RegWrapper()

#     # ---- Kendall tau on P(class=1) ----
#     k_orig = kendall_per_feature(df_orig_score, feature_cols, "_p1")
#     k_test = kendall_per_feature(df_test_score, feature_cols, "_p1")
#     print_kendall_table("original (surrogate P(class=1))", k_orig)
#     print_kendall_table("test / MLP P(class=1)", k_test)
#     compare(k_orig, k_test)

#     # ---- PDP on P(class=1) ----
#     pdp_orig = compute_pdp(surrogate_orig, X_orig, feature_cols)
#     pdp_test = compute_pdp(surrogate_test, X_test, feature_cols)
#     plot_pdp(pdp_orig, pdp_test, feature_cols, out_path=output_model)


# if __name__ == "__main__":
#     main()

def run_analysis(robot_profile, model_name):
    original_csv = ORIGINAL_CSVS[robot_profile]
    model_path = f"results/{robot_profile}/{model_name}"
    test_csv = f"{model_path}/combined_decision_log.csv"
    output_path = f"{model_path}/plots/"
    output_model = output_path + "attributions.png"

    df_orig = pd.read_csv(original_csv)
    df_test = pd.read_csv(test_csv)
    df_test = df_test[df_test["model"] == f"{robot_profile} trained"]

    df_orig = fix_dataframe(df_orig)
    df_test = fix_dataframe(df_test)

    df_orig = df_orig.apply(
        lambda r: run_prepare(r, dataset_name=robot_profile, original=False), axis=1
    ).reset_index(drop=True)
    df_test = df_test.apply(
        lambda r: run_prepare(r, dataset_name=robot_profile, original=False), axis=1
    ).reset_index(drop=True)

    feature_cols = [c for c in df_orig.columns
                    if c not in DROP_COLUMNS and c != TARGET and c in df_test.columns]

    # ---- Original: surrogate classifier ----
    X_orig = df_orig[feature_cols].values
    y_orig = df_orig[TARGET].values
    surrogate_orig = ProbaWrapper(GradientBoostingClassifier(random_state=0)).fit(X_orig, y_orig)
    df_orig_score = df_orig.copy()
    df_orig_score["_p1"] = surrogate_orig.predict(X_orig)

    # ---- Test ----
    X_test = df_test[feature_cols].values
    y_test = df_test[TARGET].values

    if set(np.unique(y_test)) <= {0, 1}:
        print("Test column looks like hard 0/1 labels; fitting a surrogate "
              "classifier on it for PDP/Kendall.")
        surrogate_test = ProbaWrapper(GradientBoostingClassifier(random_state=0)).fit(X_test, y_test)
        df_test_score = df_test.copy()
        df_test_score["_p1"] = surrogate_test.predict(X_test)
    else:
        print("Test column looks like a probability; using it directly.")
        df_test_score = df_test.copy()
        df_test_score["_p1"] = y_test
        reg = GradientBoostingRegressor(random_state=0).fit(X_test, y_test)

        class _RegWrapper:
            _estimator_type = "regressor"
            def predict(self, X): return reg.predict(X)
        surrogate_test = _RegWrapper()

    # ---- Kendall tau ----
    k_orig = kendall_per_feature(df_orig_score, feature_cols, "_p1")
    k_test = kendall_per_feature(df_test_score, feature_cols, "_p1")
    print_kendall_table("original (surrogate P(class=1))", k_orig)
    print_kendall_table("test / MLP P(class=1)", k_test)
    compare(k_orig, k_test)

    # ---- PDP ----
    pdp_orig = compute_pdp(surrogate_orig, X_orig, feature_cols)
    pdp_test = compute_pdp(surrogate_test, X_test, feature_cols)
    plot_pdp(pdp_orig, pdp_test, feature_cols, out_path=output_model)
    plt.close("all")

    return output_path

def main():
    for robot_profile in ROBOT_PROFILES:
        for model_name in MODELS:
            print(f"Running {robot_profile} / {model_name} ...")
            buf = io.StringIO()
            output_path = f"results/{robot_profile}/{model_name}/plots/"
            try:
                with contextlib.redirect_stdout(buf):
                    print(f"Robot profile: {robot_profile}\nModel: {model_name}")
                    output_path = run_analysis(robot_profile, model_name)
            except Exception:
                buf.write("\nERROR:\n" + traceback.format_exc())
                print(f"  -> failed, see txt file for the traceback")

            os.makedirs(output_path, exist_ok=True)
            txt_path = os.path.join(output_path, "attributions.txt")
            with open(txt_path, "w") as f:
                f.write(buf.getvalue())
            print(f"  -> saved {txt_path}")


if __name__ == "__main__":
    main()