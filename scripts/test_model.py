import os
import argparse
import logging
from typing import List, Dict, Union, Optional
import requests
import torch
from transformers import (
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM,
    AutoTokenizer,
    Pipeline,
    pipeline
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path
import time

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTester:
    def __init__(
        self,
        model_type: str = "api",  # "api", "local", or "huggingface"
        model_name: str = "llama2",
        api_url: Optional[str] = "http://localhost:11434/api/generate",
        model_path: Optional[str] = None,
        batch_size: int = 32,
        device: Optional[str] = None,
        max_length: int = 2048
    ):
        """
        Khởi tạo ModelTester để test các loại model khác nhau.

        Args:
            model_type: Loại model ("api", "local", "huggingface")
            model_name: Tên model
            api_url: API endpoint cho Ollama
            model_path: Đường dẫn đến local model
            batch_size: Kích thước batch
            device: Device để chạy model (cuda/cpu)
            max_length: Độ dài tối đa của output
        """
        self.model_type = model_type
        self.model_name = model_name
        self.api_url = api_url
        self.model_path = model_path
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_length = max_length
        
        self.model = None
        self.tokenizer = None
        self.pipeline = None
        
        if model_type in ["local", "huggingface"]:
            self._load_model()

    def _load_model(self):
        """Load model dựa trên model_type."""
        try:
            logger.info(f"Loading {self.model_type} model: {self.model_name}")
            
            model_source = self.model_path if self.model_type == "local" else self.model_name
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_source)
            
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_source)
                self.pipeline = pipeline("text-generation", 
                                      model=self.model, 
                                      tokenizer=self.tokenizer,
                                      device=self.device)
                logger.info("Loaded as language model")
            except:
                self.model = AutoModelForSequenceClassification.from_pretrained(model_source)
                self.pipeline = pipeline("text-classification", 
                                      model=self.model, 
                                      tokenizer=self.tokenizer,
                                      device=self.device)
                logger.info("Loaded as classification model")
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise

    def process_api_batch(self, batch: List[Dict]) -> List[Dict]:
        """Xử lý batch qua API."""
        results = []
        for test_case in batch:
            try:
                start_time = time.time()
                
                # Chuẩn bị request data
                request_data = {
                    "model": self.model_name,
                    "prompt": test_case["input"],
                    "stream": False
                }
                
                # Gửi request đến API
                response = requests.post(
                    self.api_url,
                    json=request_data,
                    timeout=30
                )
                response.raise_for_status()
                response_time = time.time() - start_time
                
                # Xử lý response
                response_json = response.json()
                
                # Lấy real_output từ response
                if isinstance(response_json, dict):
                    real_output = response_json.get("response", "").strip()
                else:
                    real_output = str(response_json).strip()
                
                result = {
                    "input": test_case["input"],
                    "real_output": real_output,
                    "expected": test_case.get("expected"),
                    "status": "success",
                    "response_time": response_time
                }
            except Exception as e:
                logger.error(f"Error processing API request: {str(e)}")
                result = {
                    "input": test_case["input"],
                    "real_output": "",
                    "error": str(e),
                    "status": "error",
                    "response_time": None
                }
            results.append(result)
        return results

    def process_local_batch(self, batch: List[Dict]) -> List[Dict]:
        """Xử lý batch với local/huggingface model."""
        results = []
        
        for test_case in batch:
            try:
                start_time = time.time()
                
                input_text = test_case["input"]
                
                # Generate output using pipeline
                if self.pipeline.task == "text-generation":
                    # Tham số cho text generation
                    generation_params = {
                        "max_length": self.max_length,
                        "num_return_sequences": 1,
                        "do_sample": True,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "pad_token_id": self.tokenizer.eos_token_id
                    }
                    
                    outputs = self.pipeline(
                        input_text,
                        **generation_params
                    )
                    
                    # Lấy generated text và loại bỏ input prompt nếu cần
                    generated_text = outputs[0]["generated_text"]
                    if generated_text.startswith(input_text):
                        real_output = generated_text[len(input_text):].strip()
                    else:
                        real_output = generated_text.strip()
                    
                else:  # text-classification
                    output = self.pipeline(input_text)[0]
                    real_output = output["label"]
                
                response_time = time.time() - start_time
                
                results.append({
                    "input": input_text,
                    "real_output": real_output,
                    "expected": test_case.get("expected"),
                    "status": "success",
                    "response_time": response_time
                })
                
            except Exception as e:
                logger.error(f"Error processing local model: {str(e)}")
                results.append({
                    "input": test_case["input"],
                    "real_output": "",
                    "error": str(e),
                    "status": "error",
                    "response_time": None
                })
        
        return results

    def run_tests(self, test_cases: List[Dict]) -> Dict:
        """Chạy tests trên tất cả test cases."""
        all_results = []
        success_count = 0
        error_count = 0
        total_response_time = 0
        response_times = []
        
        logger.info(f"Running tests on {len(test_cases)} test cases")
        
        # Process in batches
        for i in tqdm(range(0, len(test_cases), self.batch_size)):
            batch = test_cases[i:i + self.batch_size]
            
            # Choose processing method based on model type
            if self.model_type == "api":
                batch_results = self.process_api_batch(batch)
            else:  # local or huggingface
                batch_results = self.process_local_batch(batch)
            
            all_results.extend(batch_results)
            
            # Calculate statistics
            for result in batch_results:
                if result["status"] == "success":
                    success_count += 1
                    if result["response_time"] is not None:
                        response_times.append(result["response_time"])
                else:
                    error_count += 1

        # Compile metrics
        avg_response_time = np.mean(response_times) if response_times else 0
        
        metrics = {
            "total_tests": len(test_cases),
            "successful_tests": success_count,
            "failed_tests": error_count,
            "success_rate": success_count / len(test_cases) if test_cases else 0,
            "average_response_time": float(avg_response_time),
            "min_response_time": float(np.min(response_times)) if response_times else 0,
            "max_response_time": float(np.max(response_times)) if response_times else 0
        }

        return {
            "results": all_results,
            "metrics": metrics,
            "model_info": {
                "model_type": self.model_type,
                "model_name": self.model_name,
                "device": str(self.device)
            },
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }

    def save_results(self, results: Dict, output_path: str):
        """Lưu kết quả test."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Test different types of models")
    parser.add_argument("--model_type", type=str, choices=["api", "local", "huggingface"],
                      default="api", help="Type of model to test")
    parser.add_argument("--model_name", type=str, default="llama2",
                      help="Name of the model")
    parser.add_argument("--api_url", type=str, 
                      default="http://localhost:11434/api/generate",
                      help="API endpoint (for API testing)")
    parser.add_argument("--model_path", type=str,
                      help="Path to local model")
    parser.add_argument("--test_data", type=str, required=True,
                      help="Path to test data JSON file")
    parser.add_argument("--output_path", type=str, required=True,
                      help="Path to save test results")
    parser.add_argument("--batch_size", type=int, default=32,
                      help="Batch size for processing")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"],
                      help="Device to run the model on")
    parser.add_argument("--max_length", type=int, default=2048,
                      help="Maximum length of generated text")
    
    args = parser.parse_args()
    
    try:
        # Load test data
        with open(args.test_data, 'r', encoding='utf-8') as f:
            test_cases = json.load(f)
        
        # Initialize tester
        tester = ModelTester(
            model_type=args.model_type,
            model_name=args.model_name,
            api_url=args.api_url,
            model_path=args.model_path,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length
        )
        
        # Run tests
        logger.info(f"Starting tests with {args.model_type} model: {args.model_name}")
        results = tester.run_tests(test_cases)
        
        # Save results
        tester.save_results(results, args.output_path)
        
        # Log summary
        metrics = results["metrics"]
        logger.info("Testing completed. Summary:")
        logger.info(f"Total tests: {metrics['total_tests']}")
        logger.info(f"Success rate: {metrics['success_rate']:.2%}")
        logger.info(f"Average response time: {metrics['average_response_time']:.3f}s")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
