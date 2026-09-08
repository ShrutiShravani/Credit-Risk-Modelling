from dataclasses import dataclass

from typing import Optional

@dataclass
class DataIngestionArtifact:
    trained_file_path: str
    test_file_path: str


@dataclass
class DataTransformationArtifact:
    preprocessor_woe_file_path: str
    preprocessor_scaler_file_path: str

    transformed_pd_train_file_path: str
    transformed_pd_test_file_path: str

    transformed_ead_train_file_path: str
    transformed_ead_test_file_path: str

    transformed_lgd_train_file_path: str
    transformed_lgd_test_file_path: str
    transformed_full_test_features_file_path:str
    raw_fund_amt_test_file_path: str
    

@dataclass
class DataValidationArtifact:
    validation_status: bool
    message: str
    valid_train_file_path: str
    valid_test_file_path: str


@dataclass
class ModelTrainerArtifact:
    pd_model_file_path: str
    lgd_model_file_path: str
    ead_model_file_path: str
    pd_auc_score: float
    pd_gini_score: float
    lgd_rmse: float
    ead_rmse: float


@dataclass
class ModelEvaluationArtifact:
    pd_is_accepted: bool
    pd_new_score: float
    pd_old_score: Optional[float]

    lgd_is_accepted: bool
    lgd_new_score: float
    lgd_old_score: Optional[float]

    ead_is_accepted: bool
    ead_new_score: float
    ead_old_score: Optional[float]