import numpy as np
import pandas as pd
from  src.exception.exception import CustomException
from src.logging.logger import logging
from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import DataValidationArtifact, DataTransformationArtifact
import sys,os
from src.constant.training_pipeline import (PROCESSED_SCHEMA_FILE_PATH, DATA_TRANSFORMATION_DROPPING_COLUMNS, 
                                        DATA_TRANSFORMATION_IMPUTATION_COLUMNS, TARGET_COLUMN,DATA_TRANSFORMATION_ONE_HOT_ENCODING_COLUMNS,
                                        DATA_TRANSFORMATION_BINARAY_ENCODING_COLUMNS, DATA_TRANSFORMATION_ORDINAL_ENCODING_COLUMNS,
                                        DATA_TRANSFORMATION_NUMERICAL_SCALING_COLUMNS, DATA_TRANSFORMATION_EDUCATION_ORDER, 
                                        DATA_TRANSAFORMATION_CATEGORICAL_COLUMNS)
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file, save_numpy_array_data,save_object
from scipy.stats import ks_2samp
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline

class DataTransformation:
  
    def __init__(self, data_transformation_config:DataTransformationConfig, data_validation_artifact: DataValidationArtifact):
        try:
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            self.preprocessed_schema_config = read_yaml_file(PROCESSED_SCHEMA_FILE_PATH)

        except Exception as e:
            raise CustomException(e,sys)
    

    @staticmethod
    def read_data (file_path:str)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise CustomException(e,sys)       

    
    def dropping_unwanted_columns(self, dataframe: pd.DataFrame, columns_list: list)-> pd.DataFrame:        
        try:
            logging.info("Extracted policy bind date features.")
            dataframe["policy_bind_date"] = pd.to_datetime(dataframe["policy_bind_date"])
            dataframe["policy_bind_year"] = dataframe["policy_bind_date"].dt.year
            dataframe["policy_bind_month"] = dataframe["policy_bind_date"].dt.month
            dataframe["policy_bind_day"] = dataframe["policy_bind_date"].dt.day

            logging.info("Extracted incident date features.")
            dataframe["incident_date"] = pd.to_datetime(dataframe["incident_date"])
            dataframe["incident_month"] = dataframe["incident_date"].dt.month
            dataframe["incident_day"] = dataframe["incident_date"].dt.day

            logging.info("Dropped unnecessary columns.")
            dataframe.drop(
                columns=columns_list,
                errors="ignore",
                inplace=True
            )

            return dataframe
            
        except Exception as e:
            raise CustomException(e,sys)    
    
    def imputing_missing_values (self, dataframe: pd.DataFrame, column_list: list)-> pd.DataFrame:
        try:
            logging.info("Missing values imputed.")
            for col in column_list:
                if col == 'authorities_contacted':
                    dataframe[col] = dataframe[col].fillna( 'unknown')
                else:
                    dataframe[col] = dataframe[col].replace('?', 'unknown')
            return dataframe
                    
        except Exception as e:
            raise CustomException(e,sys)
    
    def validate_no_of_columns(self, dataframe: pd.DataFrame)-> bool:
        try:
            if len(dataframe.columns) == len(self.preprocessed_schema_config['columns']):
                logging.info(f"Required Number of Columns in Data after Preprocessing: {len(self.preprocessed_schema_config['columns'])}")
                logging.info(f"Number of Columns in Data Frame after Preprocessing: {len(dataframe.columns)}")
                return True
            else:
                return False
        except Exception as e:
            raise CustomException(e,sys) 

    def is_numerical_column_exits(self, dataframe: pd.DataFrame)->bool:
        try:
            missing_columns = []
            for col in self.preprocessed_schema_config['numerical_columns']:
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
            for col in self.preprocessed_schema_config["numerical_columns"]:
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
            
            transformed_data_drift_report_path = self.data_transformation_config.transformed_data_drift_report_path
            dir_path = os.path.dirname(transformed_data_drift_report_path)
            os.makedirs(dir_path, exist_ok = True)
            write_yaml_file(file_path =transformed_data_drift_report_path, content =  report)
            return status                

        except Exception as e:
            raise CustomException(e,sys)
    
    def get_data_transformation_obj(self):
        try:
            logging.info("Entered get_data_transformer_object()")
            preprocessor = ColumnTransformer(transformers=[
                ("one_hot_encoder",OneHotEncoder(handle_unknown="ignore",drop="first",sparse_output=False),DATA_TRANSFORMATION_ONE_HOT_ENCODING_COLUMNS
                ),

                ("binary_encoder",OrdinalEncoder(),DATA_TRANSFORMATION_BINARAY_ENCODING_COLUMNS
                ),

                ("ordinal_encoder",OrdinalEncoder(categories=DATA_TRANSFORMATION_EDUCATION_ORDER),DATA_TRANSFORMATION_ORDINAL_ENCODING_COLUMNS
                ),

                ("standard_scaler",StandardScaler(),DATA_TRANSFORMATION_NUMERICAL_SCALING_COLUMNS
                )
            ],
            remainder="drop"
        )

            logging.info("Created preprocessing object successfully.")

            return preprocessor

            
        except Exception as e:
            raise CustomException(e,sys)
            
    
    def initiate_data_transformation(self)-> DataTransformationArtifact:
        try:
            train_file_path = self.data_validation_artifact.valid_train_file_path
            test_file_path = self.data_validation_artifact.valid_test_file_path

            logging.info(f"Reading Training and Testing Data Sets from Data Validation Artifact")
            train_df = DataTransformation.read_data(train_file_path)
            test_df = DataTransformation.read_data(test_file_path)

            logging.info("Doing Feature Engineering and Dropping Unnecessary Columns in Train and Test")
            train_df = self.dropping_unwanted_columns(train_df, DATA_TRANSFORMATION_DROPPING_COLUMNS)
            test_df = self.dropping_unwanted_columns(test_df, DATA_TRANSFORMATION_DROPPING_COLUMNS)

            logging.info("Imputing the missing values in Train and Test")
            train_df = self.imputing_missing_values(train_df,DATA_TRANSFORMATION_IMPUTATION_COLUMNS)
            test_df = self.imputing_missing_values(test_df,DATA_TRANSFORMATION_IMPUTATION_COLUMNS)
            logging.info("Missing values successfully imputed.")

            status = True
            transformed_train_file_path = None
            transformed_test_file_path = None
            transformed_invalid_train_file_path = None
            transformed_invalid_test_file_path = None 
            error_message = [] 

            logging.info(f"Checking the Number of Columns in train.csv after Data Transformation") 
            if not self.validate_no_of_columns(train_df):            
                error_message.append(f"Train DataFrame Does Not have all the columns after Transformation.")
                os.makedirs(os.path.dirname(self.data_transformation_config.transformed_invalid_train_file_path), exist_ok = True)
                train_df.to_csv(self.data_transformation_config.transformed_invalid_train_file_path,  index=False, header=True)
                transformed_invalid_train_file_path = self.data_transformation_config.transformed_invalid_train_file_path
                status = False

            logging.info(f"Checking the Number of Columns in test.csv after Data Transformation") 
            if not self.validate_no_of_columns(test_df):            
                error_message.append(f"Train DataFrame Does Not have all the columns after Transformation.")
                os.makedirs(os.path.dirname(self.data_transformation_config.transformed_invalid_test_file_path), exist_ok = True)
                test_df.to_csv(self.data_transformation_config.transformed_invalid_test_file_path,  index=False, header=True)
                transformed_invalid_test_file_path = self.data_transformation_config.transformed_invalid_test_file_path
                status = False

            logging.info(f"Checking the Numerical Columns in train.csv after Data Transformation") 
            if not self.is_numerical_column_exits(train_df):
                error_message.append(f"Missing Columns in Train DataFrame after Transformation")
                os.makedirs(os.path.dirname(self.data_transformation_config.transformed_invalid_train_file_path), exist_ok= True)
                train_df.to_csv(self.data_transformation_config.transformed_invalid_train_file_path,  index=False, header=True)
                transformed_invalid_train_file_path = self.data_transformation_config.transformed_invalid_train_file_path
                status = False
            
            logging.info(f"Checking the Numerical Columns in test.csv after Data Transformation")
            if not self.is_numerical_column_exits(test_df):
                error_message.append(f"Missing Columns in Test DataFrame after Transformation")
                os.makedirs(os.path.dirname(self.data_transformation_config.transformed_invalid_test_file_path), exist_ok= True)
                test_df.to_csv(self.data_transformation_config.transformed_invalid_test_file_path,  index=False, header=True)
                transformed_invalid_test_file_path = self.data_transformation_config.transformed_invalid_test_file_path
                status = False

            logging.info(f"Checking the data drift in the data set after Data Transformation")
            drift_status = self.detect_dataset_drift(train_df, test_df)

            if not drift_status:
               logging.warning(
                    f"Dataset drift detected. Report saved at: "
                    f"{self.data_transformation_config.transformed_data_drift_report_path}"
                )

            if error_message:
                for msg in error_message:
                    logging.error(msg)    
            
            # If validation fails, stop the transformation process
            if not status:
                logging.error("Data Transformation validation failed. Stopping further processing.")

                data_transformation_artifact = DataTransformationArtifact(
                    transformed_validation_status=False,
                    transformed_train_file_path=None,
                    transformed_test_file_path=None,
                    transformed_object_file_path=None,
                    transformed_train_file_path_csv = None,
                    transformed_test_file_path_csv = None,
                    transformed_invalid_train_file_path=transformed_invalid_train_file_path,
                    transformed_invalid_test_file_path=transformed_invalid_test_file_path,
                    transformed_drift_report_file_path=self.data_transformation_config.transformed_data_drift_report_path
                )

                return data_transformation_artifact
            
            transformed_train_file_path = self.data_transformation_config.transformed_train_file_path
            dir_name = os.path.dirname(transformed_train_file_path)
            os.makedirs(dir_name, exist_ok = True)


            input_feature_train_df = train_df.drop(TARGET_COLUMN, axis = 1)            
            target_feature_train_df = (train_df[TARGET_COLUMN].map({"Y":1, "N":0}))

            input_feature_test_df = test_df.drop(TARGET_COLUMN, axis = 1)            
            target_feature_test_df = (test_df[TARGET_COLUMN].map({"Y":1, "N":0}))

            preprocessor = self.get_data_transformation_obj()

            input_feature_train_df[DATA_TRANSAFORMATION_CATEGORICAL_COLUMNS] = input_feature_train_df[DATA_TRANSAFORMATION_CATEGORICAL_COLUMNS].astype(str)
            input_feature_test_df[DATA_TRANSAFORMATION_CATEGORICAL_COLUMNS] = input_feature_test_df[DATA_TRANSAFORMATION_CATEGORICAL_COLUMNS].astype(str)
            
            
            
            transformed_input_train_feature = preprocessor.fit_transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor.transform(input_feature_test_df)  

            feature_names = preprocessor.get_feature_names_out()

            train_transformed_df = pd.DataFrame(
                transformed_input_train_feature,
                columns=feature_names
            )

            test_transformed_df = pd.DataFrame(
                transformed_input_test_feature,
                columns=feature_names
            )

            train_transformed_df.to_csv(
                self.data_transformation_config.transformed_train_file_path_csv,
                index=False
            )

            test_transformed_df.to_csv(
                self.data_transformation_config.transformed_test_file_path_csv,
                index=False
            )

       
            transformed_train_file_path_csv = self.data_transformation_config.transformed_train_file_path_csv
            transformed_test_file_path_csv = self.data_transformation_config.transformed_test_file_path_csv




            preprocessor_object = preprocessor
            
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]
            
            # Save numpy array data
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
                


            save_object(self.data_transformation_config.transformed_object_file_path, preprocessor_object)
            
            save_object("final_model/preprocessor.pkl", preprocessor_object)
            data_transformation_artifact = DataTransformationArtifact(
                    transformed_validation_status=True,
                    transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                    transformed_test_file_path=self.data_transformation_config.transformed_test_file_path,
                    transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                    transformed_invalid_train_file_path=None,
                    transformed_invalid_test_file_path=None,
                    transformed_drift_report_file_path=self.data_transformation_config.transformed_data_drift_report_path,
                    transformed_train_file_path_csv = transformed_train_file_path_csv,
                    transformed_test_file_path_csv = transformed_test_file_path_csv
                )
            logging.info("Saving transformed train array.")
            logging.info("Saving transformed test array.")
            logging.info("Saving preprocessing object.")

            return data_transformation_artifact
          
        except Exception as e:
            raise CustomException (e,sys)

        
        
