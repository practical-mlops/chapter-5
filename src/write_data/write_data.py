from minio import Minio
import pandas as pd
from kfp.v2.dsl import Input, Dataset, component  # Updated imports for KFP v2


@component
def write_data(
    minio_host: str,
    access_key: str,
    secret_key: str,
    bucket_name: str,
    file_name: str,
    input_data_path: Input[Dataset],  # Updated type hint for KFP v2
):
    client = Minio(
        endpoint=minio_host, access_key=access_key, secret_key=secret_key, secure=False
    )

    # Load input data from the artifact path
    input_data = pd.read_pickle(
        input_data_path.path
    )  # KFP v2 uses `.path` for artifact inputs
    input_data.to_parquet(file_name, index=False)

    # Upload the file to MinIO
    client.fput_object(bucket_name, file_name, file_name)
