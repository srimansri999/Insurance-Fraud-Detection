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




        



