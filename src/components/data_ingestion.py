from src.exception.exception import CustomException
from src.logging.logger import logging
import os, sys
import pandas as pd
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from sklearn.model_selection import train_test_split
from src.constant import training_pipeline


class DataIngestion:
    def __init__(self, data_ingestion_config:DataIngestionConfig):        
        self.data_ingestion_config = data_ingestion_config

    
    def read_data(self) -> pd.DataFrame:
        try:
            logging.info(f"Reading dataset from source.")
            df = pd.read_excel(self.data_ingestion_config.data_file_path)
            return df
        except Exception as e:
            raise CustomException(e,sys)
    
    def export_data_into_feature_store(self, dataframe: pd.DataFrame)-> None:
        try:
            logging.info(f"Exporting raw dataset to feature store.")
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path, exist_ok = True)
            dataframe.to_csv(feature_store_file_path, index= False, header = True)
            
        except Exception as e:
            raise CustomException (e,sys)
    
    def split_data_as_train_test(self, dataframe: pd.DataFrame)-> None:
        try:
            train_test_split_ratio = self.data_ingestion_config.train_test_split_ratio
            train_set, test_set = train_test_split(dataframe, test_size = train_test_split_ratio,random_state=42,
            stratify=dataframe[training_pipeline.TARGET_COLUMN])
            logging.info(f"Splitting the dataset into train set and test set")
            os.makedirs(
                self.data_ingestion_config.data_ingestion_ingested_file_path,
                exist_ok=True
            )
            train_file_path = self.data_ingestion_config.training_file_path
            test_file_path = self.data_ingestion_config.test_file_path

            os.makedirs(os.path.dirname(train_file_path), exist_ok=True)

            train_set.to_csv(train_file_path, index=False)
            test_set.to_csv(test_file_path, index=False)
            logging.info("Train and Test datasets saved successfully.")
        except Exception as e:
            raise CustomException (e,sys)
    

    

    def initiate_data_ingestion(self)-> DataIngestionArtifact:
        try:
            dataframe = self.read_data()
            self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)
            data_ingestion_artifact = DataIngestionArtifact(train_file_path=self.data_ingestion_config.training_file_path, 
            test_file_path = self.data_ingestion_config.test_file_path)
            return data_ingestion_artifact
        
        except Exception as e:
            raise CustomException(e,sys)

    







 



    