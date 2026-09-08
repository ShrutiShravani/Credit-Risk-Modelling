from src.pipeline.training_pipeline import TrainPipeline
from src.components.model_training.expected_loss import calculate_expected_loss


if __name__ == "__main__":
    pipeline = TrainPipeline()
    data_transformation_artifact, model_trainer_artifact=pipeline.run_pipeline()
    expected_loss, total_el = calculate_expected_loss(model_trainer_artifact, data_transformation_artifact)