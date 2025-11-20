from pathlib import Path

from app.core.config import settings
from app.ml.pipeline import save_model, train_sample_model


def main() -> None:
    model = train_sample_model()
    artifact_path = Path(settings.model_path)
    save_model(model, artifact_path)
    print(f"Trained toy model saved to: {artifact_path}")


if __name__ == "__main__":
    main()
