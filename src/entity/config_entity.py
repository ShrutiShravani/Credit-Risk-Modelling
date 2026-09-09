import datetime
import os
from datetime import datetime
from src.constant  import training_pipeline

class TrainingPipelineConfig:
    def __init__(self,timestamp=datetime.now()):
        timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.artifact_dir: str = os.path.join(os.getcwd(),"artifacts",timestamp)
        self.timestamp: str = timestamp

class DataIngestionConfig:
        def __init__(self,training_pipeline_config:TrainingPipelineConfig):
            self.data_ingestion_dir: str = os.path.join(
                training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME
            )
            self.feature_store_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_FEATURE_STORE_DIR, training_pipeline.FILE_NAME
            )
            self.training_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR,training_pipeline.TRAIN_FILE_NAME
            )
            self.testing_file_path: str = os.path.join(
                self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DIR, training_pipeline.TEST_FILE_NAME
            )
            self.train_test_split_ratio: float = training_pipeline.DATA_INGESTION_TRAIN_TEST_SPLIT_RATION
            self.collection_name: str = training_pipeline.DATA_INGESTION_COLLECTION_NAME


class DataValidationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_validation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir,
            training_pipeline.DATA_VALIDATION_DIR_NAME
        )
        self.valid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_VALID_DIR
        )
        self.invalid_data_dir: str = os.path.join(
            self.data_validation_dir,
            training_pipeline.DATA_VALIDATION_INVALID_DIR
        )
        self.valid_train_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline.TRAIN_FILE_NAME
        )
        self.valid_test_file_path: str = os.path.join(
            self.valid_data_dir, training_pipeline.TEST_FILE_NAME
        )
        self.invalid_train_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline.TRAIN_FILE_NAME
        )
        self.invalid_test_file_path: str = os.path.join(
            self.invalid_data_dir, training_pipeline.TEST_FILE_NAME
        )
        self.report_file_path: str = os.path.join(
            self.data_validation_dir, training_pipeline.DATA_VALIDATION_REPORT_FILE_NAME
        )

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir: str = os.path.join(
            training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME)

        transformed_dir = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR)
        obj_dir = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR)

        os.makedirs(transformed_dir, exist_ok=True)
        os.makedirs(obj_dir, exist_ok=True)

        self.transformed_pd_train_file_path: str = os.path.join(
            transformed_dir, training_pipeline.PD_TRAIN_FILE_NAME.replace("csv", "npy"))
        self.transformed_pd_test_file_path: str = os.path.join(
            transformed_dir, training_pipeline.PD_TEST_FILE_NAME.replace("csv", "npy"))

        self.transformed_ead_train_file_path: str = os.path.join(
            transformed_dir, training_pipeline.EAD_TRAIN_FILE_NAME.replace("csv", "npy"))
        self.transformed_ead_test_file_path: str = os.path.join(
            transformed_dir, training_pipeline.EAD_TEST_FILE_NAME.replace("csv", "npy"))

        self.transformed_lgd_train_file_path: str = os.path.join(
            transformed_dir, training_pipeline.LGD_TRAIN_FILE_NAME.replace("csv", "npy"))
        self.transformed_lgd_test_file_path: str = os.path.join(
            transformed_dir, training_pipeline.LGD_TEST_FILE_NAME.replace("csv", "npy"))
        self.transformed_full_test_features_file_path:str=os.path.join(transformed_dir,training_pipeline.FULL_TEST_FILE.replace("csv","npy"))
        self.raw_fund_amt_test_file_path:str=os.path.join(transformed_dir,training_pipeline.RAW_TEST_FUNDAMT)
       
        self.preprocessor_woe_file_path: str = os.path.join(obj_dir, training_pipeline.WOE_PREPROCESSOR_OBJECT_FILE_NAME)
        self.preprocessor_scaler_file_path: str = os.path.join(obj_dir, training_pipeline.SCALER_PREPROCESSOR_OBJECT_FILE_NAME)


class ModelTrainerConfig:
    def __init__(self, training_pipeline_config):
        self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir, "model_trainer")
        self.pd_model_file_path = os.path.join(self.model_trainer_dir, "pd_model.pkl")
        self.lgd_model_file_path = os.path.join(self.model_trainer_dir, "lgd_model.pkl")
        self.ead_model_file_path = os.path.join(self.model_trainer_dir, "ead_model.pkl")
        self.expected_pd_auc = 0.70   # minimum acceptable AUC, adjust as needed
        self.expected_lgd_rmse = 0.25 # maximum acceptable RMSE, adjust as needed

class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config):
        self.model_evaluation_dir = os.path.join(
            training_pipeline_config.artifact_dir, "model_evaluation")
        self.report_file_path = os.path.join(self.model_evaluation_dir, "report.yaml")

        # thresholds — how much better must the new model be to replace the old one
        self.pd_change_threshold = 0.01              # AUC must improve by at least 0.01
        self.regression_change_threshold = 0.01      # RMSE must improve by at least 0.01