import json
import os
import joblib
from sklearn.preprocessing import StandardScaler
from azureml.core.model import Model


def init():
    global model

    model_path = Model.get_model_path(
        os.getenv("AZUREML_MODEL_DIR").split('/')[-2])
    model = joblib.load(model_path)


def run(raw_data):

    try:
        data = json.loads(raw_data)["data"]
        data = StandardScaler().fit_transform(data)
        preds = model.predict(data)
        return {'prediction': preds}

    except Exception as e:
        result = str(e)
        return {"error": result}
