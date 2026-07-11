from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
from src.exception.exception import CustomException
from src.entity.artifact_entity import ClassificationMetricArtifact
import sys


def get_classification_score(
    y_true,
    y_pred,
    y_pred_prob
):
    try:

        model_accuracy_score = accuracy_score(y_true, y_pred)
        model_precision_score = precision_score(y_true, y_pred)
        model_recall_score = recall_score(y_true, y_pred)
        model_f1_score = f1_score(y_true, y_pred)
        model_auc_roc_score = roc_auc_score(y_true, y_pred_prob)

        classification_metric = ClassificationMetricArtifact(
            accuracy_score=model_accuracy_score,
            precision_score=model_precision_score,
            recall_score=model_recall_score,
            f1_score=model_f1_score,
            auc_roc_score=model_auc_roc_score
        )

        return classification_metric

    except Exception as e:
        raise CustomException(e, sys)