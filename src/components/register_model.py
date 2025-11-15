from mlflow.tracking import MlflowClient
import mlflow
from src.logger import get_logger

logger = get_logger('Register model')


def register_model(model_name: str, model_info: dict):

    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
    
        model_version = mlflow.register_model(model_uri, model_name)
        
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )
        
        logger.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logger.error('Error during model registration: %s', e)
        raise