from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity  import  DataValidationConfig
import sys,os
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.constant.training_pipeline import RAW_SCHEMA_FILE_PATH
import pandas as pd
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file
from scipy.stats import ks_2samp


class DataValidation:
    def __init__(self,data_ingestion_artifact : DataIngestionArtifact, data_validation_config : DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._raw_schema_config = read_yaml_file(RAW_SCHEMA_FILE_PATH)
        except Exception as e:
            raise CustomException (e,sys)
    
    @staticmethod
    def read_data(file_path: str)->pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException (e,sys)
        
    def validate_no_of_columns(self, dataframe: pd.DataFrame)-> bool:
        try:
            number_of_columns = len(self._raw_schema_config['columns'])
            logging.info(f"Required Number of Columns in Data: {number_of_columns}")
            logging.info(f"Number of Columns in Data Frame: {len(dataframe.columns)}")
            if number_of_columns == len(dataframe.columns):
                return True
            return False
        except Exception as e:
            raise CustomException (e,sys)
        
    def is_numerical_column_exits(self, dataframe: pd.DataFrame)->bool:
        try:
            missing_columns = []
            for col in self._raw_schema_config['numerical_columns']:
                if col not in dataframe.columns:
                    missing_columns.append(col)
            if missing_columns:
                logging.info(f"Missing Numerical Columns: {missing_columns}")
                return False
            else:
                return True

        except Exception as e:
            raise CustomException(e,sys)
    
    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df:pd.DataFrame, threshold: float = 0.05)-> bool:
        try:
            status = True
            report = {}
            for col in self._raw_schema_config["numerical_columns"]:
                d1 = base_df[col]
                d2 = current_df[col]
                is_same_dist = ks_2samp(d1,d2)
                if threshold <= is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({col:{
                    "p_value":float(is_same_dist.pvalue),
                    "drift_status": is_found
                }})
            
            data_drift_report_path = self.data_validation_config.data_drift_report_path
            dir_path = os.path.dirname(data_drift_report_path)
            os.makedirs(dir_path, exist_ok = True)
            write_yaml_file(file_path =data_drift_report_path, content =  report)
            return status                

        except Exception as e:
            raise CustomException(e,sys)

    

    def initiate_data_validation(self)-> DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.train_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            logging.info(f"Reading the train.csv and test.csv  from DataIngestionArtifact")
            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)
            
            status = True

            valid_train_file_path = None
            valid_test_file_path = None
            invalid_train_file_path = None
            invalid_test_file_path = None 

            error_message = []           

            logging.info(f"Checking the Number of Columns in train.csv") 
            if not self.validate_no_of_columns(train_df):            
                error_message.append(f"Train DataFrame Does Not have all the columns.")
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path), exist_ok = True)
                train_df.to_csv(self.data_validation_config.invalid_train_file_path,  index=False, header=True)
                invalid_train_file_path = self.data_validation_config.invalid_train_file_path
                status = False

            logging.info(f"Checking the Number of Columns in test.csv")
            if not self.validate_no_of_columns(test_df):
                error_message.append(f"Test DataFrame Does Not have all the columns.")
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_test_file_path),exist_ok = True)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path,  index=False, header=True)
                invalid_test_file_path = self.data_validation_config.invalid_test_file_path
                status = False

            logging.info(f"Checking the Numerical Columns in train.csv") 
            if not self.is_numerical_column_exits(train_df):
                error_message.append(f"Missing Columns in Train DataFrame")
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_train_file_path), exist_ok= True)
                train_df.to_csv(self.data_validation_config.invalid_train_file_path,  index=False, header=True)
                invalid_train_file_path = self.data_validation_config.invalid_train_file_path
                status = False
            
            logging.info(f"Checking the Numerical Columns in test.csv")
            if not self.is_numerical_column_exits(test_df):
                error_message.append(f"Missing Columns in Test DataFrame")
                os.makedirs(os.path.dirname(self.data_validation_config.invalid_test_file_path), exist_ok= True)
                test_df.to_csv(self.data_validation_config.invalid_test_file_path,  index=False, header=True)
                invalid_test_file_path = self.data_validation_config.invalid_test_file_path
                status = False
            

            logging.info(f"Checking the data drift in the data set")
            drift_status = self.detect_dataset_drift(train_df, test_df)

            if not drift_status:
               logging.warning(
                    f"Dataset drift detected. Report saved at: "
                    f"{self.data_validation_config.data_drift_report_path}"
                )

            if error_message:
                for msg in error_message:
                    logging.error(msg)    

            valid_train_file_path = self.data_validation_config.valid_train_file_path
            dir_name = os.path.dirname(valid_train_file_path)
            os.makedirs(dir_name, exist_ok = True)

            if status:
                valid_train_file_path = self.data_validation_config.valid_train_file_path
                dir_name = os.path.dirname(valid_train_file_path)
                os.makedirs(dir_name, exist_ok = True)
                train_df.to_csv(
                    self.data_validation_config.valid_train_file_path,
                    index=False,
                    header=True
                )

                test_df.to_csv(
                    self.data_validation_config.valid_test_file_path,
                    index=False,
                    header=True
                )

                valid_train_file_path = self.data_validation_config.valid_train_file_path
                valid_test_file_path = self.data_validation_config.valid_test_file_path

            data_validation_artifact = DataValidationArtifact(
                validation_status = status,
                valid_train_file_path=valid_train_file_path,
                valid_test_file_path=valid_test_file_path,
                invalid_train_file_path=invalid_train_file_path,
                invalid_test_file_path=invalid_test_file_path,
                drift_report_file_path = self.data_validation_config.data_drift_report_path
            )

            return data_validation_artifact
        
        except Exception as e:
            raise CustomException (e,sys)
        








