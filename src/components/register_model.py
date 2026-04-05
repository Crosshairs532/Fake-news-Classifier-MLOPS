from mlflow.tracking import MlflowClient
import mlflow
from src.logger import get_logger
from dotenv import load_dotenv

logger = get_logger('Register model')

load_dotenv()
def register_model(model_name: str, model_info: dict):
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    try:
        # model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        model_uri = f"runs:/{model_info['run_id']}/{model_name}"
    
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

if __name__ == "__main__":
    import json
    with open('reports/experiment_info.json', 'r') as file:
        model_info = json.load(file)
    register_model(model_name="FakeNewsClassifier", model_info=model_info)