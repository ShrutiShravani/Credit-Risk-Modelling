import os
import sys
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from src.entity.artifact_entity import DataIngestionArtifact
from src.entity.config_entity import DataIngestionConfig
from src.exception import CustomException
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.logger import logging
from src.data_access.credit_loan_data import LoanData  # Ensures LoanData utilit,y is imported


class DataIngestion:
  def __init__(self, data_ingestion_config: DataIngestionConfig):
    """Initializes DataIngestion component with its dataclass configuration."""
    try:
      self.data_ingestion_config = data_ingestion_config
    except Exception as e:
      raise CustomException(e, sys)

  def export_data_into_feature_store(self) -> DataFrame:
    """Exports raw data from MongoDB and saves it to the feature store directory."""
    try:
      dataframe=None
      try:
          logging.info("Exporting data from MongoDB to feature store")
          loan_data = LoanData()
          dataframe = loan_data.export_collection_as_dataframe(
              collection_name=self.data_ingestion_config.collection_name
          )

          raw_data_file_path = r"data/loan_data.csv"
          logging.info(f"Reading raw CSV dataset from: {raw_data_file_path}")

          # Read raw CSV using selected columns to reduce memory overhead
          dataframe = pd.read_csv(raw_data_file_path)
          logging.info(
              f"Successfully loaded dataset with shape: {dataframe.shape}"
          )

    
          logging.info("Successfully exported data from MongoDB.")
        
      except Exception as mongo_err:
          # 2. Fallback Path: MongoDB failed or isn't running -> Read Local CSV
          logging.warning(
              f"MongoDB connection/export failed ({mongo_err}). Falling back to"
              " local raw CSV."
          )

          raw_csv_path = r"data/loan_data.csv"

          if not os.path.exists(raw_csv_path):
            raise FileNotFoundError(
                f"Fallback failed: Local CSV not found at {raw_csv_path}"
            )

          logging.info(f"Reading raw CSV dataset from: {raw_csv_path}")
          dataframe = pd.read_csv(raw_csv_path)
          logging.info(
              f"Successfully loaded local CSV with shape: {dataframe.shape}"
          )

      feature_store_file_path = (
          self.data_ingestion_config.feature_store_file_path
      )

      # Create feature store directory if it doesn't exist
      dir_path = os.path.dirname(feature_store_file_path)
      os.makedirs(dir_path, exist_ok=True)

      dataframe.to_csv(feature_store_file_path, index=False, header=True)
      logging.info(f"Saved feature store file at: {feature_store_file_path}")
      return dataframe

    except Exception as e:
      raise CustomException(e, sys)

  def split_data_as_train_test(self, dataframe: DataFrame) -> None:
    """Splits feature store dataset into train and test CSV files."""
    try:
      train_set, test_set = train_test_split(
          dataframe,
          test_size=self.data_ingestion_config.train_test_split_ratio,
          random_state=42,  # Fixed seed for pipeline reproducibility
      )

      logging.info("Performed train-test split on the dataframe")

      dir_path = os.path.dirname(
          self.data_ingestion_config.training_file_path
      )
      os.makedirs(dir_path, exist_ok=True)

      logging.info("Exporting train and test files to ingested directory.")

      train_set.to_csv(
          self.data_ingestion_config.training_file_path,
          index=False,
          header=True,
      )

      test_set.to_csv(
          self.data_ingestion_config.testing_file_path, index=False, header=True
      )

      logging.info("Successfully exported train and test files.")

    except Exception as e:
      # Fixed: Changed LoanData(e, sys) to CustomException(e, sys)
      raise CustomException(e, sys)

  def initiate_data_ingestion(self) -> DataIngestionArtifact:
    """Orchestrates ingestion, feature storage, split, and returns artifact."""
    try:
      dataframe = self.export_data_into_feature_store()
      self.split_data_as_train_test(dataframe=dataframe)

      # Fixed artifact initialization with correct config attributes
      data_ingestion_artifact = DataIngestionArtifact(
          trained_file_path=self.data_ingestion_config.training_file_path,
          test_file_path=self.data_ingestion_config.testing_file_path,
      )

      logging.info(
          f"Data Ingestion completed successfully. Artifact:"
          f" {data_ingestion_artifact}"
      )
      return data_ingestion_artifact

    except Exception as e:
      raise CustomException(e, sys)


