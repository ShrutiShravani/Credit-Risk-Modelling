import sys
import numpy as np
import pandas as pd
import os
from src.exception import CustomException
from src.entity.artifact_entity import DataTransformationArtifact,DataIngestionArtifact,DataValidationArtifact
from src.entity.config_entity import DataTransformationConfig
from imblearn.combine import SMOTETomek
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.utils.main_utils import save_numpy_array_data, save_object
import scorecardpy as sc
from src.logger import logging

class DataTransformation:
    def __init__(self,data_transformation_config: DataTransformationConfig,data_validation_artifact:DataValidationArtifact):
        try:
            self.data_transformation_config= data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self._dti_median = None
            self._revol_util_median = None
            self._income_cap = None
        except Exception as e:
            raise CustomException(e, sys)
     
    @staticmethod
    def read_data(file_path) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e, sys)
    

    def clean_data(self,df:pd.DataFrame,is_train: bool)->pd.DataFrame:
        df=df.copy()

        bad_status = ['Charged Off', 'Default']
        good_status = ['Fully Paid']
        df = df[df['loan_status'].isin(bad_status + good_status)].copy()
        df['default_flag'] = df['loan_status'].apply(lambda x: 1 if x in bad_status else 0)
        

        # Fix dti sentinel/placeholder values
        df.loc[df['dti'] == 999.00, 'dti'] = np.nan
        if is_train:
            self._dti_median = df['dti'].median()
            self._revol_util_median = df['revol_util'].median()
            self._income_cap = df['annual_inc'].quantile(0.99)

        # these run every time, for both train AND test
        df['dti'] = df['dti'].fillna(self._dti_median)
        df['revol_util'] = df['revol_util'].fillna(self._revol_util_median)
        df['annual_inc'] = df['annual_inc'].clip(upper=self._income_cap)
        

        # emp_length — ordinal encoding, -1 for not disclosed
        emp_length_map = {
            '< 1 year': 0, '1 year': 1, '2 years': 2, '3 years': 3, '4 years': 4,
            '5 years': 5, '6 years': 6, '7 years': 7, '8 years': 8, '9 years': 9,
            '10+ years': 10
        }
        df['emp_length'] = df['emp_length'].map(emp_length_map)
        df['emp_length'] = df['emp_length'].fillna(-1)

        
        df['fico_avg'] = (df['fico_range_low'] + df['fico_range_high']) / 2
        df = df.drop(columns=['fico_range_low', 'fico_range_high'])
        

        df['term'] = df['term'].str.extract(r'(\d+)').astype(float)
        logging.info(f"BEFORE PARSE:, {df['issue_d'].head(5).tolist()}")
        logging.info(f"BEFORE PARSE:, {df['earliest_cr_line'].head(5).tolist()}")
        logging.info(f"before:{df['earliest_cr_line'].min(), df['earliest_cr_line'].max()}")
        logging.info(f"issue_d max is {df['issue_d'].max()}")
        df['issue_d']= pd.to_datetime(df['issue_d'],format='%b-%y', errors='coerce')

        df['earliest_cr_line'] = pd.to_datetime(df['earliest_cr_line'], format='%b-%y', errors='coerce')
      
        cutoff = df['issue_d'].max()
        mask = df['earliest_cr_line'] > cutoff
        df.loc[mask, 'earliest_cr_line'] = df.loc[mask, 'earliest_cr_line'] - pd.DateOffset(years=100)
        logging.info(f"after:{df['earliest_cr_line'].min(), df['earliest_cr_line'].max()}")


        logging.info(f"AFTER PARSE:, {df['issue_d'].head(5).tolist()}")

        return df
    
    @staticmethod
    def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['credit_history_length_yrs'] = (df['issue_d'] - df['earliest_cr_line']).dt.days / 365.25
        return df

    # ---------- PD branch (WoE) ----------
    def transform_pd(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
        y_col = 'default_flag'

        bins = sc.woebin(train_df[feature_cols + [y_col]], y=y_col)
        train_woe = sc.woebin_ply(train_df[feature_cols + [y_col]], bins)
        test_woe = sc.woebin_ply(test_df[feature_cols + [y_col]], bins)

        # balance train only, never test
        smt = SMOTETomek(sampling_strategy="minority")
        X_train, y_train = smt.fit_resample(
            train_woe.drop(columns=[y_col]), train_woe[y_col]
        )

        train_arr = np.c_[X_train.values, y_train.values]
        test_arr = np.c_[test_woe.drop(columns=[y_col]).values, test_woe[y_col].values]

        
        print("X_train columns:", X_train.columns.tolist())
        print("Last column of train_arr matches y_train?", 
        np.array_equal(train_arr[:, -1], y_train.values))

        save_object(self.data_transformation_config.preprocessor_woe_file_path, bins)
        save_numpy_array_data(self.data_transformation_config.transformed_pd_train_file_path, array=train_arr)
        save_numpy_array_data(self.data_transformation_config.transformed_pd_test_file_path, array=test_arr)

    # ---------- LGD / EAD branch (scaler + encoder) ----------
    def transform_lgd_ead(self, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list):
        train_def = train_df[train_df['default_flag'] == 1].copy()
        test_def = test_df[test_df['default_flag'] == 1].copy()

        net_recov_train = train_def['recoveries'] - train_def['collection_recovery_fee']
        train_def['lgd'] = (1.0 - ((train_def['total_rec_prncp'] + net_recov_train) / train_def['funded_amnt'])).clip(0, 1)
        train_def['ead'] = train_def['funded_amnt'] - train_def['total_rec_prncp']

        net_recov_test = test_def['recoveries'] - test_def['collection_recovery_fee']
        test_def['lgd'] = (1.0 - ((test_def['total_rec_prncp'] + net_recov_test) / test_def['funded_amnt'])).clip(0, 1)
        test_def['ead'] = test_def['funded_amnt'] - test_def['total_rec_prncp']

        num_cols = ['loan_amnt', 'funded_amnt', 'term', 'annual_inc', 'dti', 'int_rate', 'fico_avg',
            'revol_util', 'open_acc', 'total_acc', 'delinq_2yrs', 'inq_last_6mths',
            'mort_acc', 'emp_length', 'credit_history_length_yrs']
        cat_cols = [c for c in feature_cols if c not in num_cols]

        scaler = StandardScaler()
        train_num = scaler.fit_transform(train_def[num_cols])
        test_num = scaler.transform(test_def[num_cols])

        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        train_cat = encoder.fit_transform(train_def[cat_cols])
        test_cat = encoder.transform(test_def[cat_cols])

        train_X = np.c_[train_num, train_cat]
        test_X = np.c_[test_num, test_cat]

        #transform whoel tets data for expected loss calcualtion
        full_test_num= scaler.transform(test_df[num_cols])
        full_test_cat= encoder.transform(test_df[cat_cols])
        full_test_X = np.c_[full_test_num, full_test_cat]

        lgd_train_arr = np.c_[train_X, train_def['lgd'].values]
        lgd_test_arr = np.c_[test_X, test_def['lgd'].values]
        ead_train_arr = np.c_[train_X, train_def['ead'].values]
        ead_test_arr = np.c_[test_X, test_def['ead'].values]


        save_object(self.data_transformation_config.preprocessor_scaler_file_path, {'scaler': scaler, 'encoder': encoder})
        save_numpy_array_data(self.data_transformation_config.transformed_lgd_train_file_path, array=lgd_train_arr)
        save_numpy_array_data(self.data_transformation_config.transformed_lgd_test_file_path, array=lgd_test_arr)
        save_numpy_array_data(self.data_transformation_config.transformed_ead_train_file_path, array=ead_train_arr)
        save_numpy_array_data(self.data_transformation_config.transformed_ead_test_file_path, array=ead_test_arr)
        save_numpy_array_data(self.data_transformation_config.transformed_full_test_features_file_path,array=full_test_X)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(f"Data validation failed: {self.data_validation_artifact.message}")

            train_raw = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_raw = self.read_data(self.data_validation_artifact.valid_test_file_path)

            train_null_counts = train_raw.isna().sum()
            train_nulls = train_null_counts[train_null_counts > 0]
            
            test_null_counts = test_raw.isna().sum()
            test_nulls = test_null_counts[test_null_counts > 0]

            if not train_nulls.empty:
                logging.warning(f"Train dataset has null values after FE:\n{train_nulls}")
            else:
                logging.info("Train dataset has 0 null values after FE.")

            if not test_nulls.empty:
                logging.warning(f"Test dataset has null values after FE:\n{test_nulls}")
            else:
                logging.info("Test dataset has 0 null values after FE.")


            # clean train FIRST (fits stats), then test (reuses train's stats)
            train_clean = self.clean_data(train_raw, is_train=True)
            test_clean = self.clean_data(test_raw, is_train=False)
            logging.info("cleaning_done")
            train_null_counts = train_clean.isna().sum()
            train_nulls = train_null_counts[train_null_counts > 0]
            
            test_null_counts = test_clean.isna().sum()
            test_nulls = test_null_counts[test_null_counts > 0]

            if not train_nulls.empty:
                logging.warning(f"Train dataset has null values after FE:\n{train_nulls}")
            else:
                logging.info("Train dataset has 0 null values after FE.")

            if not test_nulls.empty:
                logging.warning(f"Test dataset has null values after FE:\n{test_nulls}")
            else:
                logging.info("Test dataset has 0 null values after FE.")

            test_clean[['funded_amnt']].to_csv(
                self.data_transformation_config.raw_fund_amt_test_file_path, index=False
            )

            train_clean = self.feature_engineering(train_clean)
            test_clean = self.feature_engineering(test_clean)
            logging.info("fe_done")


            train_null_counts = train_clean.isna().sum()
            train_nulls = train_null_counts[train_null_counts > 0]
            
            test_null_counts = test_clean.isna().sum()
            test_nulls = test_null_counts[test_null_counts > 0]

            if not train_nulls.empty:
                logging.warning(f"Train dataset has null values after FE:\n{train_nulls}")
            else:
                logging.info("Train dataset has 0 null values after FE.")

            if not test_nulls.empty:
                logging.warning(f"Test dataset has null values after FE:\n{test_nulls}")
            else:
                logging.info("Test dataset has 0 null values after FE.")


            feature_cols = ['loan_amnt', 'funded_amnt', 'term', 'int_rate', 'grade', 'annual_inc',
                             'dti', 'fico_avg', 'revol_util', 'open_acc', 'total_acc', 'delinq_2yrs',
                             'inq_last_6mths', 'mort_acc', 'emp_length', 'home_ownership', 'purpose',
                             'credit_history_length_yrs']

            self.transform_pd(train_clean, test_clean, feature_cols)
            self.transform_lgd_ead(train_clean, test_clean, feature_cols)


            artifact = DataTransformationArtifact(
                preprocessor_woe_file_path=self.data_transformation_config.preprocessor_woe_file_path,
                preprocessor_scaler_file_path=self.data_transformation_config.preprocessor_scaler_file_path,
                transformed_pd_train_file_path=self.data_transformation_config.transformed_pd_train_file_path,
                transformed_pd_test_file_path=self.data_transformation_config.transformed_pd_test_file_path,
                transformed_lgd_train_file_path=self.data_transformation_config.transformed_lgd_train_file_path,
                transformed_lgd_test_file_path=self.data_transformation_config.transformed_lgd_test_file_path,
                transformed_ead_train_file_path=self.data_transformation_config.transformed_ead_train_file_path,
                transformed_ead_test_file_path=self.data_transformation_config.transformed_ead_test_file_path,
                transformed_full_test_features_file_path=self.data_transformation_config.transformed_full_test_features_file_path,
                raw_fund_amt_test_file_path=self.data_transformation_config.raw_fund_amt_test_file_path
            )
            logging.info(f"Data transformation artifact: {artifact}")
            return artifact
        except Exception as e:
            raise CustomException(e, sys) from e
        



