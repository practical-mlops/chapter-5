from feast import FeatureStore
from kfp.v2.dsl import component, Dataset, Input, Output
import pandas as pd
from pathlib import Path
from minio import Minio
from feast.repo_config import FeastConfigError
from pydantic import ValidationError
import argparse
import os


def init_feature_store(
    minio_host: str, access_key: str, secret_key: str, bucket_name: str, file_name: str
) -> FeatureStore:
    # Download the content of the feature_store.yaml from the GCS bucket
    client = Minio(
        minio_host, access_key=access_key, secret_key=secret_key, secure=False
    )
    client.fget_object(bucket_name, file_name, "feature_store.yaml")
    config_path = Path("./") / "feature_store.yaml"
    try:
        os.environ["AWS_ACCESS_KEY_ID"] = access_key
        os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
        os.environ["FEAST_S3_ENDPOINT_URL"] = minio_host
        os.environ["S3_ENDPOINT_URL"] = minio_host
        store = FeatureStore(repo_path=".")
    except ValidationError as e:
        raise FeastConfigError(e, config_path)
    return store


@component
def get_features(
    minio_host: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    file_name: str,
    entity_df: Input[Dataset],
    feature_list: str,
    data_output: Output[Dataset],
):
    store = init_feature_store(
        minio_host, access_key, secret_key, bucket_name, file_name
    )
    print("Feature store initialized")
    feature_list = feature_list.split(",")
    print("Requested features:", feature_list)
    entity_df = pd.read_pickle(entity_df.path)
    print("Entity DataFrame head:")
    print(entity_df.head())
    feature_df = store.get_historical_features(
        entity_df=entity_df,
        features=feature_list,
    ).to_df()
    print("Retrieved historical features:")
    print(feature_df.head())
    feature_df.to_pickle(data_output.path)
