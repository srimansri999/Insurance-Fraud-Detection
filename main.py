from src.pipeline.training_pipeline import TrainingPipeline

training_pipeline = TrainingPipeline()

if __name__ =='__main__':
    model_trainer_artifact =training_pipeline.run_pipeline()
    print(model_trainer_artifact)
    