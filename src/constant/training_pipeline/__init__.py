import os

ARTIFACT_DIR: str = "artifact"
FILE_NAME: str = "loan_data.csv"

TRAIN_FILE_NAME:str="train.csv"
TEST_FILE_NAME:str="test.csv"

PD_TRAIN_FILE_NAME: str = "PD_train.csv"
PD_TEST_FILE_NAME: str = "PD_test.csv"
EAD_TRAIN_FILE_NAME: str = "EAD_train.csv"      # was "EDA_train.csv" — typo
EAD_TEST_FILE_NAME: str = "EAD_test.csv"        # was "EDA_test_csv" — missing dot too
LGD_TRAIN_FILE_NAME: str = "LGD_train.csv"
LGD_TEST_FILE_NAME: str = "LGD_test.csv"        # was "LGD_test_csv" — missing dot

FULL_TEST_FILE:str= "full_test.csv"
RAW_TEST_FUNDAMT:str="fund_amt_test.csv"


PREPROCSSING_OBJECT_FILE_NAME: str = "preprocessing.pkl"

"""
Data Ingestion related constant start with DATA_INGESTION VAR NAME
"""
DATA_INGESTION_COLLECTION_NAME: str = "loan_data"
DATA_INGESTION_DIR_NAME: str = "data_ingestion"
DATA_INGESTION_FEATURE_STORE_DIR: str = "feature_store"
DATA_INGESTION_INGESTED_DIR: str = "ingested"
DATA_INGESTION_TRAIN_TEST_SPLIT_RATION: float = 0.2

"""
Data Transformation related constant start with DATA_TRANSFORMATION_VAR_NAME 
"""

DATA_TRANSFORMATION_DIR_NAME: str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR: str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR: str = "transformed_object"
WOE_PREPROCESSOR_OBJECT_FILE_NAME: str = "woe_preprocessor.pkl"
SCALER_PREPROCESSOR_OBJECT_FILE_NAME: str = "scaler_preprocessor.pkl"

DATA_VALIDATION_DIR_NAME: str = "data_validation"
DATA_VALIDATION_VALID_DIR: str = "validated"
DATA_VALIDATION_INVALID_DIR: str = "invalid"
DATA_VALIDATION_REPORT_FILE_NAME: str = "report.yaml"



COLS_TO_USE = [
    # dates
    'issue_d', 'earliest_cr_line',
    # loan characteristics
    'loan_amnt', 'funded_amnt', 'term', 'int_rate', 'grade', 'sub_grade',
    'purpose', 'home_ownership',
    # borrower profile
    'annual_inc', 'emp_length', 'dti', 'verification_status',
    # credit history
    'fico_range_low', 'fico_range_high', 'open_acc', 'pub_rec',
    'revol_bal', 'revol_util', 'total_acc', 'delinq_2yrs',
    'inq_last_6mths', 'mort_acc',
    # outcome / label-building fields (needed to construct default_flag, lgd, ead)
    'loan_status', 'total_rec_prncp', 'recoveries', 'collection_recovery_fee'
]