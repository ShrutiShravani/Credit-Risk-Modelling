import os, sys
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.entity.artifact_entity import ModelTrainerArtifact, ModelEvaluationArtifact, DataTransformationArtifact
from src.entity.config_entity import ModelEvaluationConfig
from src.utils.main_utils import load_object, load_numpy_array_data, write_yaml_file
from sklearn.metrics import roc_auc_score, mean_squared_error


class ModelResolver:
    """Finds the currently deployed (production) model, if one exists."""
    def __init__(self, model_dir="saved_models"):
        self.model_dir = model_dir

    def is_model_exists(self, model_name: str) -> bool:
        path = os.path.join(self.model_dir, model_name)
        return os.path.exists(path)

    def get_best_model_path(self, model_name: str) -> str:
        return os.path.join(self.model_dir, model_name)


class ModelEvaluation:
    def __init__(self, model_eval_config: ModelEvaluationConfig,
                 data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_artifact: ModelTrainerArtifact):
        try:
            self.model_eval_config = model_eval_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()
        except Exception as e:
            raise CustomException(e, sys)

    def evaluate_pd(self):
        test_arr = load_numpy_array_data(self.data_transformation_artifact.transformed_pd_test_file_path)
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        new_model = load_object(self.model_trainer_artifact.pd_model_file_path)
        new_auc = roc_auc_score(y_test, new_model.predict_proba(X_test)[:, 1])

        if not self.model_resolver.is_model_exists("pd_model.pkl"):
            return True, new_auc, None  # no existing model, auto-accept

        old_model = load_object(self.model_resolver.get_best_model_path("pd_model.pkl"))
        old_auc = roc_auc_score(y_test, old_model.predict_proba(X_test)[:, 1])

        improved = new_auc - old_auc
        accepted = improved > self.model_eval_config.pd_change_threshold
        logging.info(f"PD — new AUC: {new_auc:.4f}, old AUC: {old_auc:.4f}, improved: {improved:.4f}")
        return accepted, new_auc, old_auc

    def evaluate_regression(self, model_path_key, test_file_key, model_filename):
        test_arr = load_numpy_array_data(getattr(self.data_transformation_artifact, test_file_key))
        X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

        new_model = load_object(getattr(self.model_trainer_artifact, model_path_key))
        new_rmse = np.sqrt(mean_squared_error(y_test, new_model.predict(X_test)))

        if not self.model_resolver.is_model_exists(model_filename):
            return True, new_rmse, None

        old_model = load_object(self.model_resolver.get_best_model_path(model_filename))
        old_rmse = np.sqrt(mean_squared_error(y_test, old_model.predict(X_test)))

        # for RMSE, lower is better — so "improvement" is old_rmse - new_rmse
        improved = old_rmse - new_rmse
        accepted = improved > self.model_eval_config.regression_change_threshold
        logging.info(f"{model_filename} — new RMSE: {new_rmse:.4f}, old RMSE: {old_rmse:.4f}, improved: {improved:.4f}")
        return accepted, new_rmse, old_rmse

    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            pd_accepted, pd_new, pd_old = self.evaluate_pd()
            lgd_accepted, lgd_new, lgd_old = self.evaluate_regression(
                "lgd_model_file_path", "transformed_lgd_test_file_path", "lgd_model.pkl")
            ead_accepted, ead_new, ead_old = self.evaluate_regression(
                "ead_model_file_path", "transformed_ead_test_file_path", "ead_model.pkl")

            artifact = ModelEvaluationArtifact(
                pd_is_accepted=pd_accepted, pd_new_score=pd_new, pd_old_score=pd_old,
                lgd_is_accepted=lgd_accepted, lgd_new_score=lgd_new, lgd_old_score=lgd_old,
                ead_is_accepted=ead_accepted, ead_new_score=ead_new, ead_old_score=ead_old,
            )
            write_yaml_file(self.model_eval_config.report_file_path, artifact.__dict__)
            logging.info(f"Model evaluation artifact: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e, sys) from e