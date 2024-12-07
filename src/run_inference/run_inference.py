from kfp.v2.dsl import Dataset, component, Input, Output
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import pickle


@component
def perform_inference(
    model_name: str,
    model_type: str,
    model_stage: str,
    mlflow_host: str,
    input_data: Input[Dataset],  # KFP v2 input artifact
    data_output: Output[Dataset],  # KFP v2 output artifact
):
    mlflow.set_tracking_uri(mlflow_host)
    mlflow_client = MlflowClient(mlflow_host)
    model_run_id = None
    for model in mlflow_client.search_model_versions(f"name='{model_name}'"):
        if model.current_stage == model_stage:
            model_run_id = model.run_id
            break

    if not model_run_id:
        raise ValueError(
            f"No model found in stage {model_stage} for model {model_name}."
        )

    mlflow.artifacts.download_artifacts(
        f"runs:/{model_run_id}/column_list/column_list.pkl", dst_path="column_list"
    )
    input_data_df = pd.read_pickle(input_data.path)
    input_data_df.drop(columns=["user_id", "event_timestamp"], inplace=True)

    with open("column_list/column_list.pkl", "rb") as f:
        col_list = pickle.load(f)
    input_data_df = pd.get_dummies(
        input_data_df, drop_first=True, sparse=False, dtype=float
    )
    input_data_df = input_data_df.reindex(columns=col_list, fill_value=0)

    if model_type == "sklearn":
        model = mlflow.sklearn.load_model(
            model_uri=f"models:/{model_name}/{model_stage}"
        )
    elif model_type == "xgboost":
        model = mlflow.xgboost.load_model(
            model_uri=f"models:/{model_name}/{model_stage}"
        )
    elif model_type == "tensorflow":
        model = mlflow.tensorflow.load_model(
            model_uri=f"models:/{model_name}/{model_stage}"
        )
    else:
        raise NotImplementedError(f"Model type '{model_type}' is not supported.")

    predicted_classes = [x[1] for x in model.predict_proba(input_data_df)]
    input_data_df["Predicted_Income_Class"] = predicted_classes
    input_data_df.to_pickle(data_output.path)
