import os, sys
from src.exception import CustomException
from src.logger import logging
from src.entity.config_entity import (
    TrainingPipelineConfig, DataIngestionConfig,
    DataTransformationConfig, ModelTrainerConfig,DataValidationConfig
)
from src.components.data_preprocessing.data_ingestion import DataIngestion
from src.components.data_preprocessing.data_validation import DataValidation
from src.components.data_preprocessing.data_transformation import DataTransformation
from src.components.model_training.model_training import ModelTrainer
from src.entity.artifact_entity import (
    DataIngestionArtifact, DataValidationArtifact,
    DataTransformationArtifact, ModelTrainerArtifact
)


class TrainPipeline:
    is_pipeline_running = False

    def __init__(self):
        self.training_pipeline_config = TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            self.data_ingestion_config = DataIngestionConfig(
                training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data ingestion")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config)
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            logging.info(f"Data ingestion completed and artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_validation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(
                training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data validation")
            data_validation = DataValidation(
                data_validation_config=data_validation_config,
                data_ingestion_artifact=data_ingestion_artifact)
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Data validation completed and artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact) -> DataTransformationArtifact:
        try:
            data_transformation_config = DataTransformationConfig(
                training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting data transformation")
            data_transformation = DataTransformation(
                data_transformation_config=data_transformation_config,
                data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            logging.info(f"Data transformation completed and artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def start_model_training(self, data_transformation_artifact: DataTransformationArtifact) -> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(
                training_pipeline_config=self.training_pipeline_config)
            logging.info("Starting model training")
            model_trainer = ModelTrainer(
                model_trainer_config=model_trainer_config,
                data_transformation_artifact=data_transformation_artifact)
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Model training completed and artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e, sys)

    def run_pipeline(self):
        try:
            TrainPipeline.is_pipeline_running = True

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
            data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
            model_trainer_artifact = self.start_model_training(data_transformation_artifact)

            TrainPipeline.is_pipeline_running = False
            logging.info(f"Pipeline completed. Final artifact: {model_trainer_artifact}")
            return data_transformation_artifact,model_trainer_artifact

        except Exception as e:
            TrainPipeline.is_pipeline_running = False
            raise CustomException(e, sys)