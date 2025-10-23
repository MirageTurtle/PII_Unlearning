import os
import json
import torch
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict
import numpy as np
from typing import List, Dict, Tuple
import argparse
from tqdm import tqdm

import compare_similarity

class CustomDataset(Dataset):
    """Custom dataset for gradient computation"""
    def __init__(self, data: List[Dict], tokenizer, max_length=512, use_gt_as_response=True):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_gt_as_response = use_gt_as_response
    
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
            padding='max_length',  # 固定padding到max_length
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

def extract_average_representation(
    model, 
    dataloader,
    layerid,
    verbose: bool = False
) -> np.ndarray:
    model.eval()
    all_representations = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
          
            last_hidden_states = outputs.hidden_states[layerid]  # [B, T, hidden_size]
            
            for sample_idx in range(last_hidden_states.size(0)):
                valid_mask = labels[sample_idx] != -100
                sample_hidden = last_hidden_states[sample_idx][valid_mask]  # [valid_tokens, hidden_size]
                
                if verbose:
                    valid_count = valid_mask.sum().item()
                    total_count = attention_mask[sample_idx].sum().item()
                    print(f"Batch {batch_idx}, Sample {sample_idx}: Using {valid_count}/{total_count} tokens")
                
                sample_representation = torch.mean(sample_hidden, dim=0)
                sample_repr_np = sample_representation.cpu().numpy()
                all_representations.append(sample_repr_np)
            
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"Processed {batch_idx + 1} batches, collected {len(all_representations)} samples")
    
    all_representations = np.stack(all_representations)  # [N, hidden_size]
    avg_representation = np.mean(all_representations, axis=0).reshape(1, -1)  # [1, hidden_size]
    
    return avg_representation, all_representations

def compute_sample_similarities(
    model, 
    dataloader,
    ref_representation: np.ndarray,
    layerid,
    verbose: bool = False
) -> List[Dict]:
    
    model.eval()
    similarities = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].cuda()
            attention_mask = batch["attention_mask"].cuda()
            labels = batch["labels"].cuda()
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            last_hidden_states = outputs.hidden_states[layerid]  # [B, T, hidden_size]
            
            valid_mask = labels[0] != -100

            sample_hidden = last_hidden_states[0][valid_mask]
            
            if verbose:
                valid_count = valid_mask.sum().item()
                total_count = attention_mask[0].sum().item()
                print(f"Batch {batch_idx}: Using {valid_count}/{total_count} tokens")
                        
            # average pooling
            sample_representation = torch.mean(sample_hidden, dim=0)
            # last token
            # sample_representation = sample_hidden[-1]
            
            sample_repr_np = sample_representation.cpu().numpy()
            
            # calculate similarity
            similarity = cosine_similarity(sample_repr_np.reshape(1, -1), ref_representation)[0, 0]
            similarities.append({
                'full_text': batch['full_text'],
                'cosine_similarity': similarity,
                })
            
            if verbose:
                print(f"Batch {batch_idx}: Similarity = {similarity:.4f}")
                
    return similarities

def main():
    parser = argparse.ArgumentParser(description='Representation Matching for Multiple Datasets')
    parser.add_argument('--model_name', type=str, default='./target', 
                       help='Model name or path')
    parser.add_argument('--recovered', type=str, required=True, 
                       help='Path to dataset A (JSON/JSONL file)')
    parser.add_argument('--forgotten', type=str, required=True, 
                       help='Path to dataset B (JSON/JSONL file)')
    parser.add_argument('--forget_set', type=str, required=True, 
                       help='Path to dataset C (JSON/JSONL file)')
    parser.add_argument('--layerid', type=int, required=True, 
                       help='the layer ID of hidden layer representation')
    parser.add_argument('--use_gt', action='store_true', default=True,
                       help='Use gt as ground truth response instead of response field')
    parser.add_argument('--use_response', action='store_true', default=False,
                       help='Use response field instead of gt field')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Batch size for processing')
    parser.add_argument('--max_length', type=int, default=32, 
                       help='Maximum sequence length')

    
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

    # Get the average representation of forget set C, then calculate each sample's cosine similarity with C
    ref_representation, all_representations = extract_average_representation(model, dataloader_c, args.layerid, False)
    rec_sim = compute_sample_similarities(model, dataloader_a, ref_representation, args.layerid, True)
    fgt_sim = compute_sample_similarities(model, dataloader_b, ref_representation, args.layerid, True)


if __name__ == "__main__":
    main()
