# automl_agent.py
import pandas as pd
import numpy as np
import warnings
import traceback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, accuracy_score, mean_squared_error
from sklearn.inspection import permutation_importance

# shap optional import
try:
    import shap
except Exception:
    shap = None

warnings.filterwarnings("ignore")


class AutoMLAgent:
    def __init__(self, df: pd.DataFrame, target_column: str = None, llm_api_key: str = None):
        """
        df: input dataframe
        target_column: string name of target column (if None, agent will auto-detect)
        llm_api_key: optional (not used heavily here)
        """
        self.raw_df = df.copy()
        self.target_column = target_column
        self.llm_api_key = llm_api_key

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.dropna(axis=1, how="all")
        df = df.drop_duplicates()
        # trim column names
        df.columns = df.columns.astype(str).str.strip().str.replace("\ufeff", "", regex=True)
        return df

    def _drop_text_heavy(self, df: pd.DataFrame, high_card_threshold: int = 60):
        to_drop = []
        for c in df.select_dtypes(include=["object"]).columns:
            if df[c].nunique() > high_card_threshold:
                to_drop.append(c)
        return df.drop(columns=to_drop, errors="ignore"), to_drop

    def _auto_detect_target(self, df: pd.DataFrame):
        # prefer exact common names
        for c in df.columns:
            if str(c).strip().lower() in ("target", "label", "y", "outcome", "survived"):
                return c
        # prefer low-cardinality categorical
        for c in df.columns:
            if df[c].dtype == "object" and 2 <= df[c].nunique() <= 3:
                return c
        # prefer small-integer columns
        for c in df.columns:
            if df[c].dtype.kind in "iu" and 2 <= df[c].nunique() <= 10:
                return c
        # else last numeric column
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            return num_cols[-1]
        # fallback first column
        return df.columns[0]

    def _prepare_features(self, df: pd.DataFrame):
        # Ensure target exists or auto-detect
        if self.target_column is None or self.target_column not in df.columns:
            self.target_column = self._auto_detect_target(df)

        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in dataframe after normalization.")

        # drop columns that are entirely empty
        df = df.dropna(axis=1, how="all")

        # preserve target even if it looks like an id
        id_like_cols = [c for c in df.columns if ("id" in str(c).lower() or "identifier" in str(c).lower()) and c != self.target_column]

        # Remove high-cardinality text columns (but never the target)
        df, dropped_text_cols = self._drop_text_heavy(df)

        # Now drop id-like cols from X (but not the target)
        df = df.drop(columns=id_like_cols, errors="ignore")

        # drop rows where target is missing
        df = df.dropna(subset=[self.target_column]).copy()

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        # try to coerce numeric-like strings to numeric
        for c in X.columns:
            if X[c].dtype == object:
                try:
                    X[c] = pd.to_numeric(X[c], errors="ignore")
                except Exception:
                    pass

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

        # Build transformers with imputers
        num_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])
        cat_transformer = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ])

        preprocessor = ColumnTransformer(transformers=[
            ("num", num_transformer, num_cols),
            ("cat", cat_transformer, cat_cols)
        ], remainder="drop")

        return X, y, preprocessor, num_cols, cat_cols, id_like_cols + dropped_text_cols

    def _detect_problem_type(self, y: pd.Series):
        if y.dtype == "object" or (y.nunique() <= 20 and y.dtype.kind in "iufb"):
            return "classification"
        return "regression"

    def _train_and_evaluate(self, X_train, X_test, y_train, y_test, problem_type, preprocessor):
        models = {}
        performance = {}
        feature_imps = {}

        if problem_type == "classification":
            candidates = {
                "LogisticRegression": Pipeline([("pre", preprocessor), ("clf", LogisticRegression(max_iter=500))]),
                "RandomForest": Pipeline([("pre", preprocessor), ("clf", RandomForestClassifier(n_estimators=200))]),
                "GradientBoosting": Pipeline([("pre", preprocessor), ("clf", GradientBoostingClassifier())])
            }
        else:
            candidates = {
                "LinearRegression": Pipeline([("pre", preprocessor), ("reg", LinearRegression())]),
                "RandomForestRegressor": Pipeline([("pre", preprocessor), ("reg", RandomForestRegressor(n_estimators=200))]),
                "GradientBoostingRegressor": Pipeline([("pre", preprocessor), ("reg", GradientBoostingRegressor())])
            }

        for name, model in candidates.items():
            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                if problem_type == "classification":
                    try:
                        score = float(f1_score(y_test, preds, average="weighted"))
                    except Exception:
                        score = float(accuracy_score(y_test, preds))
                else:
                    score = -float(mean_squared_error(y_test, preds, squared=False))
                performance[name] = score
                models[name] = model

                # permutation importance (best-effort)
                try:
                    r = permutation_importance(model, X_test, y_test, n_repeats=6, random_state=42, n_jobs=1)
                    try:
                        feat_names = model.named_steps["pre"].get_feature_names_out()
                        if hasattr(feat_names, "tolist"):
                            feat_names = list(feat_names)
                    except Exception:
                        feat_names = X_test.columns.tolist()
                    imp_series = dict(zip(feat_names, r.importances_mean))
                    feature_imps[name] = imp_series
                except Exception:
                    feature_imps[name] = {}
            except Exception:
                traceback.print_exc()
                performance[name] = float("-inf")
                feature_imps[name] = {}

        return models, performance, feature_imps

    def run(self):
        df = self._basic_clean(self.raw_df.copy())
        X, y, preprocessor, num_cols, cat_cols, dropped = self._prepare_features(df)
        problem_type = self._detect_problem_type(y)

        # split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # train candidates
        models, perf, feat_imps = self._train_and_evaluate(X_train, X_test, y_train, y_test, problem_type, preprocessor)

        # pick best
        try:
            best_name = max(perf, key=perf.get)
        except Exception:
            best_name = list(perf.keys())[0] if perf else None
        best_score = perf.get(best_name, None)
        best_feat_imp = feat_imps.get(best_name, {})

        # training details
        training_details = {
            "num_rows": len(df),
            "num_features": X.shape[1],
            "num_categorical": len(cat_cols),
            "num_numerical": len(num_cols),
            "dropped_text_cols": dropped
        }

        # build result dict
        result = {
            "processed_df": df.fillna(df.median(numeric_only=True)).fillna("missing"),
            "problem_type": problem_type,
            "target_column": self.target_column,
            "models": models,
            "model_performance": perf,
            "best_model_name": best_name,
            "best_score": best_score,
            "feature_importances": best_feat_imp,
            "training_details": training_details,
            "shap": None  # will try to compute below
        }

        # attach trained pipeline for best model (if available)
        if best_name and best_name in models:
            best_pipeline = models[best_name]

            # save for downstream use
            result["trained_pipeline"] = best_pipeline

            # Compute SHAP values (best-effort)
            if shap is not None:
                try:
                    # get preprocessor and raw model from pipeline if possible
                    if hasattr(best_pipeline, "named_steps") and "model" in best_pipeline.named_steps:
                        model_for_shap = best_pipeline.named_steps["model"]
                        preproc = best_pipeline.named_steps.get("pre", None)
                    else:
                        model_for_shap = best_pipeline
                        preproc = None

                    # get a sample (use train split from earlier via transform)
                    # We'll transform a subset of the processed dataframe
                    X_all = X.copy()
                    # fit preprocessor on X_all if needed (safe)
                    if preproc is not None:
                        try:
                            X_proc = preproc.transform(X_all)
                            # get feature names for visualization
                            try:
                                feature_names = preproc.get_feature_names_out(X_all.columns)
                            except Exception:
                                # fallback: use numeric + categorical columns
                                feature_names = list(X_all.columns)
                        except Exception:
                            X_proc = None
                            feature_names = list(X_all.columns)
                    else:
                        X_proc = X_all.values
                        feature_names = list(X_all.columns)

                    # Build explainer and compute shap values
                    explainer = None
                    shap_values = None
                    try:
                        explainer = shap.TreeExplainer(model_for_shap)
                        shap_values = explainer.shap_values(X_proc)
                    except Exception:
                        # fallback to KernelExplainer (slower)
                        try:
                            explainer = shap.KernelExplainer(model_for_shap.predict, shap.sample(X_proc, min(50, X_proc.shape[0])))
                            shap_values = explainer.shap_values(shap.sample(X_proc, min(200, X_proc.shape[0])))
                        except Exception:
                            explainer = None
                            shap_values = None

                    result["shap"] = {
                        "explainer": explainer,
                        "shap_values": shap_values,
                        "feature_names": feature_names
                    }
                except Exception:
                    # shap failed; leave as None
                    result["shap"] = None
            else:
                result["shap"] = None
        else:
            result["trained_pipeline"] = None

        return result
