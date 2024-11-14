import os
import argparse
import logging
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import HfApi, create_repo, upload_folder
import yaml
from datetime import datetime
from typing import Optional, Union, Dict
import hashlib

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelPusher:
    def __init__(
        self,
        model_path: Union[str, Path],
        repo_name: str,
        hub_token: Optional[str] = None,
        organization: Optional[str] = None,
        config_path: Optional[str] = None,
        private: bool = False,
        commit_message: Optional[str] = None
    ):
        self.model_path = Path(model_path)
        self.repo_name = repo_name
        self.hub_token = hub_token or os.getenv("HUGGINGFACE_TOKEN")
        self.organization = organization
        self.config_path = config_path
        self.private = private
        self.commit_message = commit_message or f"Upload model {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        if not self.hub_token:
            raise ValueError("Hugging Face token is required. Set it via constructor or HUGGINGFACE_TOKEN env variable.")
        
        self.api = HfApi()
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        if self.config_path and os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return {}

    def _calculate_checksum(self, file_path: Union[str, Path]) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _verify_model(self) -> bool:
        try:
            model = AutoModel.from_pretrained(self.model_path)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            test_input = tokenizer("Test input", return_tensors="pt")
            with torch.no_grad():
                outputs = model(**test_input)
            
            return True
        except Exception as e:
            logger.error(f"Model verification failed: {str(e)}")
            return False

    def _prepare_model_card(self) -> str:
        model_card = f"""---
        language: en
        tags:
        - pytorch
        - transformers
        license: apache-2.0
        ---

        # Model Card for {self.repo_name}

        ## Model Details

        ### Model Description

        {self.config.get('description', 'No description provided.')}

        ### Framework versions

        * Transformers {self.config.get('transformers_version', 'N/A')}
        * PyTorch {torch.__version__}
        * Datasets {self.config.get('datasets_version', 'N/A')}

        ## Training and Evaluation

        {self.config.get('training_details', 'No training details provided.')}

        ## Intended Uses & Limitations

        {self.config.get('intended_uses', 'No intended uses specified.')}

        ## How to use

        ```python
        from transformers import AutoModel, AutoTokenizer

        model = AutoModel.from_pretrained("{self.organization + '/' if self.organization else ''}{self.repo_name}")
        tokenizer = AutoTokenizer.from_pretrained("{self.organization + '/' if self.organization else ''}{self.repo_name}")
        Limitations and Bias
        {self.config.get('limitations', 'No limitations specified.')}

        Training Data
        {self.config.get('training_data', 'No training data information provided.')}

        Evaluation Results
        {self.config.get('evaluation_results', 'No evaluation results provided.')}

        Environmental Impact
        {self.config.get('environmental_impact', 'No environmental impact information provided.')}

        Citations
        {self.config.get('citations', 'No citations provided.')}

        Creator
        {self.config.get('creator', 'No creator information provided.')}

        Date Created
        {datetime.now().strftime('%Y-%m-%d')}
        """
        return model_card


    def push(self):
        try:
            logger.info("Verifying model...")
            if not self._verify_model():
                raise ValueError("Model verification failed")

            logger.info(f"Creating repository: {self.repo_name}")
            repo_url = create_repo(
                repo_id=f"{self.organization+'/' if self.organization else ''}{self.repo_name}",
                token=self.hub_token,
                private=self.private,
                exist_ok=True
            )

            logger.info("Creating model card...")
            model_card_path = self.model_path / "README.md"
            with open(model_card_path, 'w') as f:
                f.write(self._prepare_model_card())

            logger.info("Uploading model files...")
            upload_folder(
                folder_path=str(self.model_path),
                repo_id=f"{self.organization+'/' if self.organization else ''}{self.repo_name}",
                commit_message=self.commit_message,
                token=self.hub_token
            )

            logger.info(f"Model successfully pushed to {repo_url}")
            return repo_url

        except Exception as e:
            logger.error(f"Error pushing model: {str(e)}")
            raise
def main():
    parser = argparse.ArgumentParser(description="Push a model to Hugging Face Hub")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--repo_name", type=str, required=True, help="Name for the repository")
    parser.add_argument("--hub_token", type=str, help="Hugging Face token")
    parser.add_argument("--organization", type=str, help="Organization name")
    parser.add_argument("--config_path", type=str, help="Path to config file")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    parser.add_argument("--commit_message", type=str, help="Commit message")
    args = parser.parse_args()

    pusher = ModelPusher(
        model_path=args.model_path,
        repo_name=args.repo_name,
        hub_token=args.hub_token,
        organization=args.organization,
        config_path=args.config_path,
        private=args.private,
        commit_message=args.commit_message
    )

    pusher.push()
if __name__ == "__main__":
    main()
