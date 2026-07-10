

from src.entity.config_entity import (TrainingPipelineConfig,DataIngestionConfig, DataValidationConfig, DataTransformationConfig)
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
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
            f"Data_Ingestion_Artifact: {data_ingestion_artifact}")
        print(data_ingestion_artifact)
        data_validation_config = DataValidationConfig(training_pipeline_config)
        data_validation = DataValidation(data_ingestion_artifact, data_validation_config)
        logging.info(f"Intitiating Data Validation")
        data_validation_artifact= data_validation.initiate_data_validation()
        logging.info(f"Data Validation Completed.\n"
            f"Data_validation_Artifact: {data_validation_artifact}")
        print(data_validation_artifact)    
        data_transaformation_config = DataTransformationConfig(training_pipeline_config)  
        data_transaformation = DataTransformation(data_transaformation_config,data_validation_artifact )  
        logging.info(f"Intitiating Data Transaformation")
        data_transformation_artifact = data_transaformation.initiate_data_transformation()
        logging.info(f"Data Transformation Completed.\n"
            f"Data_Transformation_Artifact: {data_transformation_artifact}")
        print(data_transformation_artifact) 


    except Exception as e:
        raise CustomException(e,sys)

if __name__ == '__main__':
    main()

