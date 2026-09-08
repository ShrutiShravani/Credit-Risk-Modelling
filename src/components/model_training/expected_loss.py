import numpy as np
from src.utils.main_utils import load_object, load_numpy_array_data
import json
from  src.entity.artifact_entity  import DataTransformationArtifact,ModelTrainerArtifact
import pandas as pd

def calculate_expected_loss(model_trainer_artifact: ModelTrainerArtifact,
    data_transformation_artifact: DataTransformationArtifact,):
    pd_model = load_object(model_trainer_artifact.pd_model_file_path)
    lgd_model = load_object(model_trainer_artifact.lgd_model_file_path)
    ead_model = load_object(model_trainer_artifact.ead_model_file_path)

    # PD: full test set (already has all loans)
    pd_test_arr = load_numpy_array_data(data_transformation_artifact.transformed_pd_test_file_path)
    X_test_pd = pd_test_arr[:, :-1]
    pd_pred = pd_model.predict_proba(X_test_pd)[:, 1]

    # LGD/EAD: the NEW full-test-set features file 
    X_test_full = load_numpy_array_data(data_transformation_artifact.transformed_full_test_features_file_path)
    lgd_pred = np.clip(lgd_model.predict(X_test_full), 0, 1)
    ead_pred = np.clip(ead_model.predict(X_test_full), 0, None)

    expected_loss_per_loan = pd_pred * lgd_pred * ead_pred

    risk_adjusted_reserve = expected_loss_per_loan.sum()

   
    print(f"Average EL per loan: ${expected_loss_per_loan.mean():,.2f}")

    funded_amnt_df = pd.read_csv(data_transformation_artifact.raw_fund_amt_test_file_path)
    total_funded_amount = funded_amnt_df['funded_amnt'].sum()
    naive_reserve = total_funded_amount * 0.05

    efficiency_gain_pct = (naive_reserve - risk_adjusted_reserve) / naive_reserve * 100

    results = {
        "pd_auc": model_trainer_artifact.pd_auc_score,
        "lgd_rmse": model_trainer_artifact.lgd_rmse,
        "ead_rmse": model_trainer_artifact.ead_rmse,
        "total_expected_loss": float(risk_adjusted_reserve),
        "naive_reserve": float(naive_reserve),
        "risk_adjusted_reserve": float(risk_adjusted_reserve),
        "efficiency_gain_pct": float(efficiency_gain_pct)
    }

    with open("artifacts/model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"Total Expected Loss: ${risk_adjusted_reserve:,.2f}")
    print(f"Naive Reserve (5% flat): ${naive_reserve:,.2f}")
    print(f"Efficiency Gain: {efficiency_gain_pct:.1f}%")

    return results

   