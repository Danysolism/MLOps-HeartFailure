import numpy as np
from src.training.train import get_model_metrics


def test_get_model_metrics():

    class MockModel:

        @staticmethod
        def predict(data):
            return ([1, 1])

    X_test = np.array([75, 0, 582, 0, 20, 1, 265000,
                       1.9, 130, 1, 0, 4, 65, 0, 146, 0,
                       20, 0, 162000, 1.3, 129, 1, 1, 7]).reshape(-1, 2)
    y_test = np.array([1, 1])
    data = {"test": {"features": X_test, "target": y_test}}

    metrics = get_model_metrics(MockModel(), data)

    assert 'f1_score' in metrics
    f1 = metrics['f1_score']
    np.testing.assert_almost_equal(f1, 1.0)
