import argparse
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
from tqdm import tqdm

# Thiết lập logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FineTuner:
    def __init__(self, model_name, num_labels, output_dir):
        self.model_name = model_name
        self.num_labels = num_labels
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        ).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def load_data(self, dataset_name, train_batch_size, eval_batch_size):
        dataset = load_dataset(dataset_name)
        
        train_dataset = dataset["train"]
        eval_dataset = dataset["validation"] if "validation" in dataset else dataset["test"]
        
        def tokenize_function(examples):
            return self.tokenizer(examples["text"], padding="max_length", truncation=True)
        
        tokenized_train = train_dataset.map(tokenize_function, batched=True)
        tokenized_eval = eval_dataset.map(tokenize_function, batched=True)
        
        train_dataloader = DataLoader(
            tokenized_train, shuffle=True, batch_size=train_batch_size
        )
        eval_dataloader = DataLoader(
            tokenized_eval, batch_size=eval_batch_size
        )
        
        return train_dataloader, eval_dataloader

    def train(self, train_dataloader, num_epochs, learning_rate):
        optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        
        num_training_steps = num_epochs * len(train_dataloader)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        
        progress_bar = tqdm(range(num_training_steps))
        
        self.model.train()
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                loss.backward()
                
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)
            
            logger.info(f"Epoch {epoch+1}/{num_epochs} completed")
        
        self.model.save_pretrained(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)
        logger.info(f"Model saved to {self.output_dir}")

    def evaluate(self, eval_dataloader):
        self.model.eval()
        total_eval_loss = 0
        total_eval_accuracy = 0
        total_eval_samples = 0
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = self.model(**batch)
            
            loss = outputs.loss
            logits = outputs.logits
            
            total_eval_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            total_eval_accuracy += (predictions == batch["labels"]).sum().item()
            total_eval_samples += len(batch["labels"])
        
        avg_val_loss = total_eval_loss / len(eval_dataloader)
        avg_val_accuracy = total_eval_accuracy / total_eval_samples
        
        logger.info(f"Validation Loss: {avg_val_loss}")
        logger.info(f"Validation Accuracy: {avg_val_accuracy}")

def main(args):
    fine_tuner = FineTuner(args.model_name, args.num_labels, args.output_dir)
    train_dataloader, eval_dataloader = fine_tuner.load_data(
        args.dataset_name, args.train_batch_size, args.eval_batch_size
    )
    fine_tuner.train(train_dataloader, args.num_epochs, args.learning_rate)
    fine_tuner.evaluate(eval_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a transformer model")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained model")
    parser.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset to use")
    parser.add_argument("--num_labels", type=int, required=True, help="Number of labels for classification")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the fine-tuned model")
    parser.add_argument("--num_epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--train_batch_size", type=int, default=8, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=16, help="Evaluation batch size")
    
    args = parser.parse_args()
    main(args)
