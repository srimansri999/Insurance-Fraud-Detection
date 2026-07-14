import sys
from src.entity.config_entity import (TrainingPipelineConfig, DataIngestionConfig,DataTransformationConfig,DataValidationConfig,ModelTrainerConfig)
from src.entity.artifact_entity import (DataIngestionArtifact, DataTransformationArtifact, DataValidationArtifact, ModelTrainerArtifact)
from src.exception.exception import CustomException
from src.logging.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

class TrainingPipeline: 
    def __init__(self):
        self.training_pipeline_config =TrainingPipelineConfig()

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion_config = DataIngestionConfig(training_pipeline_config= self.training_pipeline_config)
            logging.info(f"Start Data Ingestion")
            data_ingestion = DataIngestion(data_ingestion_config)
            data_ingestion_artifact=data_ingestion.initiate_data_ingestion()
            logging.info(f"Data Ingestion Completed. \n"
                         "Data Ingestion Artifcat: {data_ingestion_artifact} ")
            return data_ingestion_artifact
                    
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_validation(self,data_ingestion_artifact: DataIngestionArtifact)-> DataValidationArtifact:
        try:
            data_validation_config = DataValidationConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info(f"Start Data Validation.")
            data_validation = DataValidation(data_validation_config=data_validation_config,data_ingestion_artifact = data_ingestion_artifact )
            data_validation_artifact = data_validation.initiate_data_validation()
            logging.info(f"Completed Data Validation.\n"
                         "Data Validation Artifact:{data_validation_artifact}")
            return data_validation_artifact
            
        except Exception as e:
            raise CustomException(e,sys)
    
    def start_data_transformation(self, data_validation_artifact: DataValidationArtifact)->DataTransformationArtifact:
        try:
            data_transformation_config=DataTransformationConfig(training_pipeline_config = self.training_pipeline_config)
            logging.info(f"Started Data Transformation.")
            data_transformation=DataTransformation(data_transformation_config= data_transformation_config, 
                                                   data_validation_artifact= data_validation_artifact)
            data_transformation_artifact=data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation Completed.\n"
                         "Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    def start_model_trainer (self, data_transformation_artifact : DataTransformationArtifact)-> ModelTrainerArtifact:
        try:
            model_trainer_config = ModelTrainerConfig(trainig_pipeline_config = self.training_pipeline_config)
            logging.info(f"Started Model Training")
            model_trainer = ModelTrainer(model_trainer_config= model_trainer_config,
                                         data_transformation_artifact= data_transformation_artifact )
            model_trainer_artifact = model_trainer.initiate_model_trainer()
            logging.info(f"Completed Model Training.\n"
                         "Model Training Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)
    
    def run_pipeline(self):
        try:
          data_ingestion_artifact  =self.start_data_ingestion()
          data_validation_artifact = self.start_data_validation(data_ingestion_artifact)
          data_transformation_artifact = self.start_data_transformation(data_validation_artifact)
          model_trainer_artifact = self.start_model_trainer(data_transformation_artifact)
          return model_trainer_artifact
        except Exception as e:
            raise CustomException(e,sys)