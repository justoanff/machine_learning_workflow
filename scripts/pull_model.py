import os
import time
import logging
import argparse
from pathlib import Path
from typing import Optional, Union, Dict, List
import requests
import json
import yaml
import hashlib
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelSource(Enum):
    HUGGINGFACE = "huggingface"
    OLLAMA = "ollama"

@dataclass
class ModelInfo:
    name: str
    source: ModelSource
    pull_date: datetime
    checksum: Optional[str] = None
    size: Optional[int] = None
    metadata: Optional[Dict] = None

class ModelPullerError(Exception):
    """Base exception for ModelPuller"""
    pass

class ServiceNotAvailableError(ModelPullerError):
    """Raised when service is not available"""
    pass

class ModelNotFoundError(ModelPullerError):
    """Raised when model is not found"""
    pass

class VerificationError(ModelPullerError):
    """Raised when model verification fails"""
    pass

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip('/')

    def check_service(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=5)
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

    def list_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            return [model['name'] for model in response.json().get('models', [])]
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get model list: {e}")
            return []

    def pull_model(self, model_name: str, callback=None) -> bool:
        """Pull a model from Ollama"""
        url = f"{self.base_url}/api/pull"
        payload = {"name": model_name}
        
        try:
            with requests.post(url, json=payload, stream=True) as response:
                if response.status_code == 404:
                    raise ModelNotFoundError(f"Model '{model_name}' not found")
                
                response.raise_for_status()
                for line in response.iter_lines():
                    if line:
                        try:
                            status = json.loads(line)
                            if callback and 'status' in status:
                                callback(status['status'])
                            if 'error' in status:
                                raise ModelPullerError(status['error'])
                        except json.JSONDecodeError:
                            continue
            return True
        except requests.exceptions.RequestException as e:
            raise ModelPullerError(f"Failed to pull model: {e}")

    def verify_model(self, model_name: str) -> bool:
        """Verify if model is working"""
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model_name,
                    "prompt": "test",
                    "stream": False
                },
                timeout=30
            )
            response.raise_for_status()
            return True
        except requests.exceptions.RequestException:
            return False

class ModelPuller:
    def __init__(
        self,
        model_name: str,
        output_dir: Union[str, Path],
        source: str = "ollama",
        config_path: Optional[str] = None,
        ollama_url: str = "http://localhost:11434"
    ):
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.source = ModelSource(source.lower())
        self.config = self._load_config(config_path) if config_path else {}
        
        if self.source == ModelSource.OLLAMA:
            self.client = OllamaClient(ollama_url)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
            return {}

    def _save_model_info(self, model_info: ModelInfo):
        """Save model information to JSON file"""
        info_file = self.output_dir / f"{self.model_name}_info.json"
        info_dict = {
            "name": model_info.name,
            "source": model_info.source.value,
            "pull_date": model_info.pull_date.isoformat(),
            "checksum": model_info.checksum,
            "size": model_info.size,
            "metadata": model_info.metadata
        }
        
        with open(info_file, 'w') as f:
            json.dump(info_dict, f, indent=2)

    def _print_model_suggestions(self):
        """Print suggestions for available models"""
        logger.info("\nPopular models you can try:")
        suggestions = [
            "llama2", "codellama", "mistral", "neural-chat",
            "starling-lm", "dolphin-phi"
        ]
        for model in suggestions:
            logger.info(f"- {model}")
        logger.info("\nFor more models, visit: https://ollama.ai/library")

    def pull_model(self) -> bool:
        """Main method to pull model"""
        try:
            if self.source == ModelSource.OLLAMA:
                return self._pull_ollama_model()
            else:
                raise NotImplementedError("Only Ollama is supported currently")
                
        except ServiceNotAvailableError:
            logger.error("Ollama service is not available")
            logger.error("Please check:")
            logger.error("1. Is Ollama installed? If not, install it:")
            logger.error("   - macOS: brew install ollama")
            logger.error("   - Linux: curl https://ollama.ai/install.sh | sh")
            logger.error("   - Windows: Download from https://ollama.ai/download")
            logger.error("2. Is Ollama service running? Start it with:")
            logger.error("   ollama serve")
            return False
            
        except ModelNotFoundError:
            logger.error(f"Model '{self.model_name}' not found")
            self._print_model_suggestions()
            return False
            
        except ModelPullerError as e:
            logger.error(f"Error pulling model: {e}")
            return False
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            return False

    def _pull_ollama_model(self) -> bool:
        """Pull model from Ollama"""
        if not self.client.check_service():
            raise ServiceNotAvailableError()

        # Show available models
        available_models = self.client.list_models()
        if available_models:
            logger.info("Currently available models:")
            for model in available_models:
                logger.info(f"- {model}")

        # Pull model
        logger.info(f"Pulling model: {self.model_name}")
        self.client.pull_model(
            self.model_name,
            callback=lambda status: logger.info(f"Status: {status}")
        )

        # Verify model
        logger.info("Verifying model...")
        time.sleep(2)  # Give some time for model to load
        if not self.client.verify_model(self.model_name):
            raise VerificationError("Model verification failed")

        # Save model info
        model_info = ModelInfo(
            name=self.model_name,
            source=self.source,
            pull_date=datetime.now()
        )
        self._save_model_info(model_info)

        logger.info(f"Successfully pulled and verified model: {self.model_name}")
        return True

def parse_args():
    parser = argparse.ArgumentParser(description="Pull AI models from various sources")
    parser.add_argument("--model_name", required=True, help="Name of the model to pull")
    parser.add_argument("--output_dir", required=True, help="Directory to save the model")
    parser.add_argument("--source", default="ollama", choices=["ollama", "huggingface"],
                      help="Source to pull model from")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--ollama_url", default="http://localhost:11434",
                      help="Ollama API URL")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        puller = ModelPuller(
            model_name=args.model_name,
            output_dir=args.output_dir,
            source=args.source,
            config_path=args.config,
            ollama_url=args.ollama_url
        )
        
        success = puller.pull_model()
        exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\nProcess interrupted by user")
        exit(130)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        exit(1)

if __name__ == "__main__":
    main()
