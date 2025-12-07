from kfp import dsl
from google.cloud import storage

@dsl.component
def upload_model_component(
    model_artifact: dsl.Input[dsl.Artifact],
    gcs_uri: str
):
    """
    Uploads model.pkl to a GCS bucket.
    Requires GOOGLE_APPLICATION_CREDENTIALS in the runtime environment.
    """
    storage_client = storage.Client()

    bucket_name, blob_name = gcs_uri.replace("gs://", "").split("/", 1)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    blob.upload_from_filename(model_artifact.path)
    print(f"Model uploaded → {gcs_uri}")
