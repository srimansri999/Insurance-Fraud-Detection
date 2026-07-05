from src.entity.config_entity import (TrainingPipelineConfig,DataIngestionConfig)
from src.components.data_ingestion import DataIngestion
from src.exception.exception import CustomException
from src.logging.logger import logging
import sys

def main():
    try:
        training_pipeline_config = TrainingPipelineConfig()
        data_ingestion_config = DataIngestionConfig(training_pipeline_config)
        data_ingestion = DataIngestion(data_ingestion_config)
        logging.info(f"Intitiating Data ingestion")
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        logging.info(f"Data Ingestion Completed.\n"
         f"Artifact: {data_ingestion_artifact}")
        print(data_ingestion_artifact)
    except Exception as e:
        raise CustomException(e,sys)

if __name__ == '__main__':
    main()
