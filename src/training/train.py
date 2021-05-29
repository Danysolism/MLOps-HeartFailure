import os
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET_COL = 'DEATH_EVENT'


def split_data(df):
    X = df.drop(TARGET_COL, axis=1).values
    y = df[TARGET_COL].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_state=2, shuffle=True)
    data = {"train": {"features": X_train, "target": y_train},
            "test": {"features": X_test, "target": y_test}}
    return data


def preprocess_data(data):
    data = StandardScaler().fit_transform(data)
    return data


def train_model(data, model_args):
    model = RandomForestClassifier(**model_args)
    train_data = preprocess_data(data["train"]["features"])
    model.fit(train_data, data["train"]["target"])
    return model


def get_model_metrics(model, data):
    test_data = preprocess_data(data["test"]["features"])
    preds = model.predict(test_data)
    f1 = f1_score(preds, data["test"]["target"])
    acc = accuracy_score(preds, data["test"]["target"])
    conf_matrix = confusion_matrix(preds, data["test"]["target"])
    metrics = {"f1_score": f1, "accuracy": acc,
               "confusion matrix": conf_matrix}
    return metrics


def main():

    model_args = {"n_estimators": 20,
                  "criterion": "entropy",
                  "random_state": 0}

    data_dir = "data"
    data_file = os.path.join(data_dir, 'heartfailure_dataset.csv')
    train_df = pd.read_csv(data_file)

    data = split_data(train_df)
    model = train_model(data, model_args)

    metrics = get_model_metrics(model, data)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
