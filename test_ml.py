import os
import pytest
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.metrics import precision_score, recall_score, fbeta_score
import pandas as pd
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    train_model,
    inference,
    save_model,
    load_model,
)

# Sample data
data = pd.DataFrame(
    {
        "age": [39, 50, 38, 53],
        "workclass": ["State-gov", "Self-emp-not-inc", "Private", "Private"],
        "fnlgt": [77516, 83311, 215646, 234721],
        "education": ["Bachelors", "Bachelors", "HS-grad", "11th"],
        "education-num": [13, 13, 9, 7],
        "marital-status": [
            "Never-married",
            "Married-civ-spouse",
            "Divorced",
            "Married-civ-spouse",
        ],
        "occupation": [
            "Adm-clerical",
            "Exec-managerial",
            "Handlers-cleaners",
            "Handlers-cleaners",
        ],
        "relationship": ["Not-in-family", "Husband", "Not-in-family", "Husband"],
        "race": ["White", "White", "White", "Black"],
        "sex": ["Male", "Male", "Male", "Male"],
        "capital-gain": [2174, 0, 0, 0],
        "capital-loss": [0, 0, 0, 0],
        "hours-per-week": [40, 13, 40, 40],
        "native-country": [
            "United-States",
            "United-States",
            "United-States",
            "United-States",
        ],
        "salary": ["<=50K", ">=50K", "<=50K", ">=50K"],
    }
)

categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]


def test_process_data():
    """Test the process_data function.

    This test verifies that the process_data function correctly processes
    the input DataFrame by encoding categorical features and binarizing
    the labels.

    It checks:
    - The output shapes of the features (X) and labels (y) match the input data.
    - The encoder and label binarizer are instances of OneHotEncoder and
      LabelBinarizer, respectively.
    - The encoder and label binarizer have been fitted with categories and
      classes.

    Raises:
        AssertionError: If any of the checks fail.
    """
    X, y, encoder, lb = process_data(
        data, categorical_features=categorical_features, label="salary", training=True
    )

    assert X.shape[0] == data.shape[0], (
        "Mismatch in number of rows between X and input data."
    )
    assert y.shape[0] == data.shape[0], (
        "Mismatch in number of rows between y and input data."
    )
    assert isinstance(encoder, OneHotEncoder), (
        "Encoder is not an instance of OneHotEncoder."
    )
    assert isinstance(lb, LabelBinarizer), (
        "Label binarizer is not an instance of LabelBinarizer."
    )
    assert encoder.categories_ is not None, "Encoder categories are not set."
    assert lb.classes_ is not None, "Label binarizer classes are not set."


def test_train_model():
    """Test the train_model function.

    This test ensures that the train_model function successfully trains a
    model on the processed data and that the model can make predictions.

    It checks:
    - The trained model is not None.
    - The model can make predictions with the same shape as the labels.

    Raises:
        AssertionError: If any of the checks fail.
    """
    X, y, encoder, lb = process_data(
        data, categorical_features=categorical_features, label="salary", training=True
    )
    model = train_model(X, y)

    assert model is not None, "Trained model is None."
    preds = inference(model, X)
    assert preds.shape == y.shape, "Mismatch in prediction and label shapes."


def test_inference():
    """Test the inference function.

    This test verifies that the inference function produces predictions
    of the correct shape and type.

    It checks:
    - The predictions have the same shape as the labels.
    - All predictions are either 0 or 1.

    Raises:
        AssertionError: If any of the checks fail.
    """
    X, y, encoder, lb = process_data(
        data, categorical_features=categorical_features, label="salary", training=True
    )
    model = train_model(X, y)
    preds = inference(model, X)

    assert preds.shape == y.shape, "Mismatch in prediction and label shapes."
    assert all(pred in [0, 1] for pred in preds), (
        "Predictions contain values other than 0 and 1."
    )


def test_compute_model_metrics():
    """Test the compute_model_metrics function.

    This test validates that the compute_model_metrics function correctly
    computes precision, recall, and F-beta scores by comparing its output
    to sklearn's metrics.

    Raises:
        AssertionError: If any of the computed metrics do not match sklearn's
        results.
    """
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == precision_score(y_true, y_pred), (
        "Precision does not match sklearn's result."
    )
    assert recall == recall_score(y_true, y_pred), (
        "Recall does not match sklearn's result."
    )
    assert fbeta == fbeta_score(y_true, y_pred, beta=1), (
        "F-beta score does not match sklearn's result."
    )


def test_save_and_load_model(tmp_path):
    """Test the save_model and load_model functions.

    This test ensures that a trained model can be saved to disk and
    subsequently loaded, and that the loaded model can make predictions.

    It checks:
    - The loaded model can make predictions with the same shape as the labels.

    Args:
        tmp_path (Path): Temporary directory provided by pytest for file operations.

    Raises:
        AssertionError: If the loaded model's predictions do not match the
        expected shape.
    """
    X, y, encoder, lb = process_data(
        data, categorical_features=categorical_features, label="salary", training=True
    )
    model = train_model(X, y)

    model_path = os.path.join(tmp_path, "model.pkl")
    save_model(model, model_path)

    loaded_model = load_model(model_path)

    preds = inference(loaded_model, X)
    assert preds.shape == y.shape, (
        "Mismatch in prediction and label shapes for loaded model."
    )
