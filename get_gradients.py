import os
import json
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm

import compare_similarity

class CustomDataset(Dataset):
    """Custom dataset for gradient computation"""
    def __init__(self, data: List[Dict], tokenizer, max_length=512, use_gt_as_response=True, dataset_type='auto'):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_gt_as_response = use_gt_as_response
        self.dataset_type = dataset_type
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different dataset formats
        if 'question' in item and 'answer' in item:
            # Special format for dataset C: convert name-email pairs to Q&A format
            question = item['question']
            prompt = f"Question: {question}\nAnswer: "
            completion = item['answer']
            full_text = f"{prompt}{completion}"
            prompt_length_for_tokenization = len(self.tokenizer.encode(prompt))
            
        elif 'prompt' in item:
            # Format for datasets A and B
            prompt = item['prompt']
            
            # Choose between gt (ground truth) or response (model output)
            if self.use_gt_as_response and 'gt' in item:
                completion = item['gt']
            elif 'response' in item:
                completion = item['response']
            else:
                raise ValueError(f"No suitable completion found in item: {item}")
            
            # Combine prompt and completion
            full_text = f"{prompt}{completion}"
            prompt_length_for_tokenization = len(self.tokenizer.encode(prompt))
        
        encoding = self.tokenizer(
            full_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        if 'prompt' in locals() or 'name' in item:
            if 'name' in item and 'email' in item:
                prompt_text = f"Question: Tell me the email address of {item['name']}\nAnswer: "
            elif 'prompt' in item:
                prompt_text = item['prompt']
            else:
                prompt_text = prompt
            
            prompt_encoding = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors='pt'
            )
            prompt_length = len(prompt_encoding['input_ids'][0])
        else:
            prompt_length = 0
        
        # Create labels for language modeling
        labels = input_ids.clone()
        
        if prompt_length > 0:
            labels[:prompt_length] = -100
        
        labels[input_ids == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'prompt_length': prompt_length,
            'full_text': full_text
        }

def load_dataset_from_file(file_path: str, key: str = None) -> List[Dict]:
    """
    Load dataset from JSON/JSONL file and return as a list of dictionaries
    
    Args:
        file_path: Path to the JSON/JSONL file
        key: Optional key to extract data from (e.g., 'recovered', 'forgotten')
    
    Returns:
        List[Dict]: List of dictionaries, each containing one data item
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        loaded_data = json.load(f)
        
        # Case 1: If a specific key is provided and exists, extract data from that key
        if key and isinstance(loaded_data, dict) and key in loaded_data:
            data = loaded_data[key]
        
        # Case 2: Data is already a list (like your email dataset)
        elif isinstance(loaded_data, list):
            data = loaded_data
        
        # Case 3: Single item (not a dict or list)
        else:
            data = [loaded_data]
    
    # Validate that we have a list of dictionaries
    if not isinstance(data, list):
        raise ValueError(f"Expected list, got {type(data)}")
    
    # Ensure all items are dictionaries
    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} is not a dictionary: {type(item)}")
    
    print(f"Loaded {len(data)} items from {file_path}")
    if data:
        print(f"Sample item keys: {list(data[0].keys())}")
    
    return data

def obtain_gradients(model, batch):
    loss = model(**batch).loss
    
    loss.backward()
    
    vectorized_grads = torch.cat([p.grad.view(-1).to(torch.float32) for p in model.parameters() if p.grad is not None])

    vectorized_grads = vectorized_grads.cpu()
    
    del loss
    
    return vectorized_grads

def collect_full_grads(eval_dataloader, model, max_response_length=-1, store_grads=True):
    print("Collecting full gradients...")
  
    model = model.bfloat16().cuda()
    model.eval()
    
    count = 0
    current_grads = []
    avg_grads = None

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        for key in batch:
            if key != "full_text":
                batch[key] = batch[key].cuda()
        
        if max_response_length > 0:
            labels = batch["labels"].clone()  
        
        valid_mask = labels[0] >= 0
        if valid_mask.any():
            pos = torch.where(valid_mask)[0][0]
            labels[0][pos + max_response_length:] = -100

        final_response_length = (labels[0] >= 0).sum().item()
        assert final_response_length <= max_response_length, \
            f"Response length {final_response_length} exceeds max_response_length {max_response_length}"

        batch["labels"] = labels
        
        model_input = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels']
        }
        
        vectorized_grads = obtain_gradients(model, model_input)

        if store_grads == True:
            current_grads.append(vectorized_grads)

        if avg_grads is None:
            avg_grads = vectorized_grads.clone().float()
            gradient_size = vectorized_grads.shape[0]
            print(f"Gradient size: {gradient_size}")
            print(f"Estimated memory per gradient: {gradient_size * 4 / 1024**3:.2f} GB")
        else:
            avg_grads += (vectorized_grads.float() - avg_grads) / (count + 1)
        
        model.zero_grad()
        
        count += 1
    
    print(f"Finished collecting gradients. Total batches processed: {count}")

    return current_grads, avg_grads

def calculate_grads_sim(eval_dataloader, model, ref_grads, max_response_length=-1):
    print("Calculating cosine similarity between gradients...")
    
    model = model.bfloat16().cuda()
    model.eval()
    
    count = 0
    similarities = []

    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        for key in batch:
            if key != "full_text":
                batch[key] = batch[key].cuda()
        
        if max_response_length > 0:
            labels = batch["labels"].clone()

        valid_mask = labels[0] >= 0
        if valid_mask.any():
            pos = torch.where(valid_mask)[0][0]
            labels[0][pos + max_response_length:] = -100

        final_response_length = (labels[0] >= 0).sum().item()
        assert final_response_length <= max_response_length, \
            f"Response length {final_response_length} exceeds max_response_length {max_response_length}"

        batch["labels"] = labels
        
        model_input = {
            'input_ids': batch['input_ids'],
            'attention_mask': batch['attention_mask'],
            'labels': batch['labels']
        }
        
        vectorized_grads = obtain_gradients(model, model_input)

        metrics = compare_similarity.compute_similarity_metrics(vectorized_grads.clone().float(), ref_grads)
        
        similarities.append({
            'full_text': batch['full_text'],
            'cosine_similarity': metrics['cosine_similarity'],
            'dot_product': metrics['dot_product']
        })

        model.zero_grad()        
        count += 1
    
    print(f"Finished collecting gradients. Total batches processed: {count}")
    similarities.sort(key=lambda x: x['cosine_similarity'], reverse=True)

    return similarities

def main():
    parser = argparse.ArgumentParser(description='Gradient Matching for Multiple Datasets')
    parser.add_argument('--model_name', type=str, required=True, 
                       help='Model name or path')
    parser.add_argument('--recovered', type=str, required=True, 
                       help='Path to dataset A (JSON/JSONL file)')
    parser.add_argument('--forgotten', type=str, required=True, 
                       help='Path to dataset B (JSON/JSONL file)')
    parser.add_argument('--forget_set', type=str, required=True, 
                       help='Path to dataset C (JSON/JSONL file)')
    parser.add_argument('--use_gt', action='store_true', default=True,
                       help='Use gt as ground truth response instead of response field')
    parser.add_argument('--use_response', action='store_true', default=False,
                       help='Use response field instead of gt field')
    parser.add_argument('--batch_size', type=int, default=4, 
                       help='Batch size for processing')
    parser.add_argument('--max_length', type=int, default=32, 
                       help='Maximum sequence length')
    parser.add_argument('--max_response_length', type=int, default=10, 
                       help='Maximum response length for gradient computation (-1 for full sequence)')
    parser.add_argument('--output_dir', type=str, default='./gradient_outputs', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Determine whether to use gt or response
    use_gt_as_response = not args.use_response
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    print(f"Using {'gt' if use_gt_as_response else 'response'} field as target")
    
    # Load model and tokenizer
    print(f"Loading model: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16 if device.type == 'cuda' else torch.float32,
        device_map='auto' if device.type == 'cuda' else None
    )
    
    # Add pad token if it doesn't exist
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print(f"Model loaded. Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load datasets
    print("Loading datasets...")
    dataset_a_data = load_dataset_from_file(args.recovered, "recovered")
    dataset_b_data = load_dataset_from_file(args.forgotten, "forgotten")
    dataset_c_data = load_dataset_from_file(args.forget_set)
    
    print(f"Dataset A: {len(dataset_a_data)} samples")
    print(f"Dataset B: {len(dataset_b_data)} samples")
    print(f"Dataset C: {len(dataset_c_data)} samples")
    
    # Create datasets and dataloaders
    dataset_a = CustomDataset(dataset_a_data, tokenizer, args.max_length, use_gt_as_response)
    dataset_b = CustomDataset(dataset_b_data, tokenizer, args.max_length, use_gt_as_response)
    dataset_c = CustomDataset(dataset_c_data, tokenizer, args.max_length, use_gt_as_response)

    dataloader_a = DataLoader(dataset_a, batch_size=args.batch_size, shuffle=False)
    dataloader_b = DataLoader(dataset_b, batch_size=args.batch_size, shuffle=False)
    dataloader_c = DataLoader(dataset_c, batch_size=64, shuffle=False)

    # Compute average gradients for each dataset
    print("\n" + "="*50)
    print("Computing average gradients for each dataset...\n")

    grads_c, avg_c = collect_full_grads(
        dataloader_c, 
        model, 
        max_response_length=10,
        store_grads = False  
    )

    rec_sim_a = calculate_grads_sim(dataloader_a, model, avg_c, max_response_length=10)
    print(rec_sim_a)
    with open('./results/gradients/recovered_unknown_unlearn_ga.json', 'w', encoding='utf-8') as f:
        json.dump(rec_sim_a, f, ensure_ascii=False, indent=2)
    
    rec_sim_b = calculate_grads_sim(dataloader_b, model, avg_c, max_response_length=10)
    with open('./results/gradients/forgotten_unknown_unlearn_ga.json', 'w', encoding='utf-8') as f:
        json.dump(rec_sim_b, f, ensure_ascii=False, indent=2)
    #fgt_sim = compare_similarity.calculate_grads_sim(dataloader_b, model, avg_c, max_response_length=10)

if __name__ == "__main__":
    main()
