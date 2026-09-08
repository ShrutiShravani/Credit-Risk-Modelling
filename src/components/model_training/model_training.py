import os, sys
import math
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from xgboost import XGBClassifier, XGBRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import load_numpy_array_data, save_object
from src.entity.artifact_entity import ModelTrainerArtifact, DataTransformationArtifact
from src.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact
            mlflow.set_experiment("credit_risk_pd_lgd_ead")
        except Exception as e:
            raise CustomException(e, sys)

    # ---------- PD model ----------
    def train_pd_model(self):
        train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_pd_train_file_path)
        test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_pd_test_file_path)

        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        with mlflow.start_run(run_name="pd_model"):
            # Baseline: Logistic Regression
            log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
            log_reg.fit(X_train, y_train)
            log_reg_auc = roc_auc_score(y_test, log_reg.predict_proba(X_test)[:, 1])
            mlflow.log_metric("log_reg_auc", log_reg_auc)

            # Challenger: XGBoost
            xgb_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                       eval_metric='auc', random_state=42)
            xgb_model.fit(X_train, y_train)
            xgb_auc = roc_auc_score(y_test, xgb_model.predict_proba(X_test)[:, 1])
            mlflow.log_metric("xgb_auc", xgb_auc)

            logging.info(f"Logistic Regression AUC: {log_reg_auc:.4f} | XGBoost AUC: {xgb_auc:.4f}")

            if xgb_auc >= log_reg_auc:
                best_model, best_auc, best_name = xgb_model, xgb_auc, "XGBClassifier"
            else:
                best_model, best_auc, best_name = log_reg, log_reg_auc, "LogisticRegression"

            gini = 2 * best_auc - 1

            mlflow.log_param("selected_model", best_name)
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 4)
            mlflow.log_param("learning_rate", 0.05)
            mlflow.log_metric("selected_auc", best_auc)
            mlflow.log_metric("gini", gini)

            if best_name == "XGBClassifier":
                mlflow.xgboost.log_model(best_model, "model")
            else:
                mlflow.sklearn.log_model(best_model, "model")

            if best_auc < self.model_trainer_config.expected_pd_auc:
                mlflow.log_param("status", "FAILED_THRESHOLD")
                raise Exception(f"PD model AUC {best_auc:.4f} below acceptable threshold "
                                 f"{self.model_trainer_config.expected_pd_auc}")

            mlflow.log_param("status", "PASSED")
            save_object(self.model_trainer_config.pd_model_file_path, best_model)

        return best_model, best_auc, gini

    # ---------- LGD model ----------
    def train_lgd_model(self):
        train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_lgd_train_file_path)
        test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_lgd_test_file_path)

        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        with mlflow.start_run(run_name="lgd_model"):
            model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred = np.clip(y_pred, 0, 1)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            logging.info(f"LGD model — RMSE: {rmse:.4f}")

            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 4)
            mlflow.log_param("learning_rate", 0.05)
            mlflow.log_metric("rmse", rmse)
            mlflow.xgboost.log_model(model, "model")

            if rmse > self.model_trainer_config.expected_lgd_rmse:
                mlflow.log_param("status", "FAILED_THRESHOLD")
                raise Exception(f"LGD model RMSE {rmse:.4f} above acceptable threshold "
                                 f"{self.model_trainer_config.expected_lgd_rmse}")

            mlflow.log_param("status", "PASSED")
            save_object(self.model_trainer_config.lgd_model_file_path, model)

        return model, rmse

    # ---------- EAD model ----------
    def train_ead_model(self):
        train_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_ead_train_file_path)
        test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_ead_test_file_path)

        X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        with mlflow.start_run(run_name="ead_model"):
            model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_pred = np.clip(y_pred, 0, None)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))

            logging.info(f"EAD model — RMSE: {rmse:.4f}")

            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 4)
            mlflow.log_param("learning_rate", 0.05)
            mlflow.log_metric("rmse", rmse)
            mlflow.xgboost.log_model(model, "model")
            mlflow.log_param("status", "PASSED")

            save_object(self.model_trainer_config.ead_model_file_path, model)

        return model, rmse

    # ---------- Orchestrator ----------
    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            os.makedirs(self.model_trainer_config.model_trainer_dir, exist_ok=True)

            pd_model, pd_auc, pd_gini = self.train_pd_model()
            lgd_model, lgd_rmse = self.train_lgd_model()
            ead_model, ead_rmse = self.train_ead_model()

            artifact = ModelTrainerArtifact(
                pd_model_file_path=self.model_trainer_config.pd_model_file_path,
                lgd_model_file_path=self.model_trainer_config.lgd_model_file_path,
                ead_model_file_path=self.model_trainer_config.ead_model_file_path,
                pd_auc_score=pd_auc,
                pd_gini_score=pd_gini,
                lgd_rmse=lgd_rmse,
                ead_rmse=ead_rmse,
            )
            logging.info(f"Model trainer artifact: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e, sys) from e