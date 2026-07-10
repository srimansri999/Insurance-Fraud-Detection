import os
import sys
from src.constant import training_pipeline
from datetime import datetime


class TrainingPipelineConfig:
    def __init__(self, timestamp = None):
        if timestamp is None:
            timestamp = datetime.now()
        self.timestamp = timestamp.strftime("%m_%d_%Y_%H_%M_%S")
        self.pipeline_name = training_pipeline.PIPELINE_NAME
        self.artifact_name = training_pipeline.ARTIFACT_DIR
        self.artifact_dir = os.path.join(self.artifact_name, self.timestamp)

class DataIngestionConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_ingestion_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_INGESTION_DIR_NAME)
        self.feature_store_file_path = os.path.join(self.data_ingestion_dir,training_pipeline.DATA_INGESTION_FEATURE_STORE_FILE_PATH,
        training_pipeline.FILE_NAME)
        self.data_ingestion_ingested_file_path = os.path.join(self.data_ingestion_dir, training_pipeline.DATA_INGESTION_INGESTED_DATA)
        self.training_file_path = os.path.join(self.data_ingestion_ingested_file_path, training_pipeline.TRAIN_FILE_NAME)
        self.test_file_path = os.path.join(self.data_ingestion_ingested_file_path, training_pipeline.TEST_FILE_NAME)
        self.train_test_split_ratio = training_pipeline.DATA_INGESTIOON_TRAIN_TEST_SPLIT_RATION
        self.data_file_path = training_pipeline.DATA_INGESTION_FILE_PATH

class DataValidationConfig:
    def __init__(self,training_pipeline_config:TrainingPipelineConfig):
        self.data_validation_dir = os.path.join (training_pipeline_config.artifact_dir,training_pipeline.DATA_VALIDATION_DIR_NAME )
        self.valid_data_dir_name= os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_VALID_DIR_NAME)
        self.invalid_data_dir_name = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_INVALID_DIR_NAME)
        self.valid_train_file_path = os.path.join(self.valid_data_dir_name,training_pipeline.TRAIN_FILE_NAME )
        self.valid_test_file_path = os.path.join(self.valid_data_dir_name,training_pipeline.TEST_FILE_NAME )
        self.invalid_train_file_path = os.path.join(self.invalid_data_dir_name, training_pipeline.TRAIN_FILE_NAME)
        self.invalid_test_file_path = os.path.join(self.invalid_data_dir_name,training_pipeline.TEST_FILE_NAME )
        self.data_drift_report_path = os.path.join(self.data_validation_dir, training_pipeline.DATA_VALIDATION_DIRFT_REPORT_FILE_PATH,
                     training_pipeline.DATA_VALIDATION_DRIFT_REPORT_FILE_NAME)

class DataTransformationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir, training_pipeline.DATA_TRANSFORMATION_DIR_NAME)
        self.transformed_valid_dir_name = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_VALID_DIR_NAME)
        self.transformed_invalid_dir_name = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_INVALID_DIR_NAME)
        self.transformed_train_file_path = os.path.join(self.transformed_valid_dir_name, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR
                                                        ,training_pipeline.TRAIN_FILE_NAME.replace("csv","npy"))
        self.transformed_test_file_path = os.path.join(self.transformed_valid_dir_name, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,
                                                       training_pipeline.TEST_FILE_NAME.replace("csv","npy"))
        self.transformed_invalid_train_file_path = os.path.join(self.transformed_invalid_dir_name,training_pipeline.TRAIN_FILE_NAME )
        self.transformed_invalid_test_file_path = os.path.join(self.transformed_invalid_dir_name, training_pipeline.TEST_FILE_NAME)
        self.transformed_data_drift_report_path = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_DRIFT_REPORT_FILE_PATH,
                                                               training_pipeline.DATA_TRANSFORMATION_DRIFT_REPORT_FILE_NAME)
        self.transformed_object_file_path = os.path.join(self.data_transformation_dir, training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,
                                                         training_pipeline.DATA_TRANSFORMATION_TRANSFORMED_OBJECT_FILE_NAME)
        




        



