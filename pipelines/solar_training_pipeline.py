from kfp import dsl
from kfp import compiler

from components.data_prep import data_prep_component
from components.feature_engineering import feature_engineering_component
from components.train_model import train_model_component
from components.upload_model import upload_model_component

@dsl.pipeline(
    name="solar-forecasting-training-pipeline",
    description="End-to-end pipeline for solar forecasting RF model."
)
def solar_pipeline(
    raw_csv_path: str,
    lag_hours: int = 1,
    n_estimators: int = 200,
    min_samples_leaf: int = 2,
    gcs_model_path: str = "gs://your-bucket/model.pkl"
):

    # Step 1: Data Prep
    cleaned = data_prep_component(raw_csv_path=raw_csv_path)

    # Step 2: Feature Engineering
    features = feature_engineering_component(
        cleaned_data=cleaned.output,
        lag_hours=lag_hours
    )

    # Step 3: Train Model
    model = train_model_component(
        features=features.output,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf
    )

    # Step 4: Upload to GCS
    upload_model_component(
        model_artifact=model.output,
        gcs_uri=gcs_model_path
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=solar_pipeline,
        package_path="solar_training_pipeline.yaml"
    )
