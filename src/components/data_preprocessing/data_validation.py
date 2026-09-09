import sys,os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.entity.artifact_entity import DataValidationArtifact, DataIngestionArtifact
from src.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig,
                 data_ingestion_artifact: DataIngestionArtifact):
        try:
            self.data_validation_config = data_validation_config
            self.ingestion_artifact = data_ingestion_artifact
            # the columns you expect in the raw ingested data (your usecols list)
            self.expected_columns = [
                'issue_d', 'earliest_cr_line', 'loan_amnt', 'funded_amnt', 'term',
                'int_rate', 'grade', 'sub_grade', 'purpose', 'home_ownership',
                'annual_inc', 'emp_length', 'dti', 'verification_status',
                'fico_range_low', 'fico_range_high', 'open_acc', 'pub_rec',
                'revol_bal', 'revol_util', 'total_acc', 'delinq_2yrs',
                'inq_last_6mths', 'mort_acc', 'loan_status', 'total_rec_prncp',
                'recoveries', 'collection_recovery_fee'
            ]
        except Exception as e:
            raise CustomException(e, sys)

    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)

    def validate_number_of_columns(self, df: pd.DataFrame) -> bool:
        return len(df.columns) == len(self.expected_columns)

    def validate_column_names(self, df: pd.DataFrame) -> bool:
        missing_columns = set(self.expected_columns) - set(df.columns)
        if missing_columns:
            logging.info(f"Missing columns: {missing_columns}")
            return False
        return True

    def validate_train_test_column_match(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> bool:
        return set(train_df.columns) == set(test_df.columns)

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            train_df = self.read_data(self.ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.ingestion_artifact.test_file_path)

            validation_error_msg = ""
            status = True

            for df_name, df in [("train", train_df), ("test", test_df)]:
                if not self.validate_number_of_columns(df):
                    status = False
                    validation_error_msg += f"{df_name}: column count mismatch (got {len(df.columns)}, expected {len(self.expected_columns)}). "
                if not self.validate_column_names(df):
                    status = False
                    validation_error_msg += f"{df_name}: missing expected columns. "

            if not self.validate_train_test_column_match(train_df, test_df):
                status = False
                validation_error_msg += "train and test columns don't match each other. "

            os.makedirs(self.data_validation_config.valid_data_dir, exist_ok=True)
            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                message=validation_error_msg,
                valid_train_file_path=self.ingestion_artifact.trained_file_path if status else None,
                valid_test_file_path=self.ingestion_artifact.test_file_path if status else None,
            )

            logging.info(f"Data validation artifact: {data_validation_artifact}")
            return data_validation_artifact
        except Exception as e:
            raise CustomException(e, sys) from e