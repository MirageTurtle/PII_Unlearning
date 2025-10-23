import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from tqdm import tqdm

class NextTokenAnalyzer:
    def __init__(self, model_name: str, device: str = "cuda"):
        """
        初始化分析器
        
        Args:
            model_name: 模型名称 (如 "gpt2", "microsoft/DialoGPT-medium")
            device: 设备 ("auto", "cuda", "cpu")
        """
        self.device = device
        
        # 加载tokenizer和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # 设置pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def get_next_token_probability(self, prompt: str, target_token_id: int, target_token: str, step: int) -> Dict:
        """获取指定prompt下特定token的概率信息"""
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            last_token_logits = outputs.logits[0, -1, :]  # 最后一个位置的logits
            probabilities = F.softmax(last_token_logits, dim=-1)

            if step == 1:
                target_token_space = " " + target_token
                target_token_id_space = self.tokenizer.encode(target_token_space)[-1]#带空格的token_id
            
            target_prob = probabilities[target_token_id].item()
            target_log_prob = torch.log(probabilities[target_token_id] + 1e-10).item()

            if step == 1:
                target_prob_space = probabilities[target_token_id_space].item()
                target_log_prob_space = torch.log(probabilities[target_token_id_space] + 1e-10).item()
                target_prob += target_prob_space
                target_log_prob += target_log_prob_space
            
            rank = self._get_token_rank(probabilities, target_token_id)
            
            entropy = self._calculate_entropy(probabilities).item()
            
        return {
            'probability': target_prob,
            'log_probability': target_log_prob,
            'rank': rank,
            'entropy': entropy
        }
    
    def get_top_k_tokens(self, probabilities: torch.Tensor, k: int = 10) -> List[Dict]:
        """
        获取概率最高的top-k个token
        
        Args:
            probabilities: 概率分布
            k: top-k个数
            
        Returns:
            top_k_info: 包含token信息的列表
        """
        top_k_probs, top_k_indices = torch.topk(probabilities, k)
        
        top_k_info = []
        for i in range(k):
            token_id = top_k_indices[i].item()
            prob = top_k_probs[i].item()
            token = self.tokenizer.decode([token_id])
            
            top_k_info.append({
                "rank": i + 1,
                "token_id": token_id,
                "token": token,
                "probability": prob,
                "log_probability": np.log(prob)
            })
        
        return top_k_info
    
    def check_ground_truth_in_top_k(self, 
                                   prompt: str, 
                                   ground_truth: str, 
                                   k: int = 10) -> Dict:
        """
        检查ground truth token是否在top-k概率中
        
        Args:
            prompt: 输入prompt
            ground_truth: 真实的下一个token
            k: top-k个数
            
        Returns:
            result: 分析结果字典
        """
        
        probabilities, info = self.get_next_token_probabilities(prompt)
        gt_token_ids = self.tokenizer.encode(ground_truth, add_special_tokens=False)
        
        top_k_info = self.get_top_k_tokens(probabilities, k)
        top_k_token_ids = [item["token_id"] for item in top_k_info]
        
        results = []
        for gt_token_id in gt_token_ids:
            gt_prob = probabilities[gt_token_id].item()
            gt_rank = self._get_token_rank(probabilities, gt_token_id)
            
            is_in_top_k = gt_token_id in top_k_token_ids
            if is_in_top_k:
                top_k_position = top_k_token_ids.index(gt_token_id) + 1
            else:
                top_k_position = None
            
            results.append({
                "ground_truth_token": self.tokenizer.decode([gt_token_id]),
                "ground_truth_id": gt_token_id,
                "probability": gt_prob,
                "log_probability": np.log(gt_prob),
                "rank": gt_rank,
                "is_in_top_k": is_in_top_k,
                "top_k_position": top_k_position
            })
        
        return {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "k": k,
            "model_info": info,
            "top_k_tokens": top_k_info,
            "ground_truth_analysis": results
        }
    
    def _calculate_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        log_probs = torch.log(probabilities + 1e-10)
        entropy = -torch.sum(probabilities * log_probs)
        return entropy
    
    def _get_token_rank(self, probabilities: torch.Tensor, token_id: int) -> int:
        sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
        rank = (sorted_indices == token_id).nonzero(as_tuple=True)[0].item() + 1
        return rank


    def calculate_sequential_probabilities(self, 
                                         prompt: str, 
                                         ground_truth: str,
                                         verbose: bool = True) -> Dict:
        """
        计算ground truth中每个token的逐步概率
        
        Args:
            prompt: 初始prompt
            ground_truth: 完整的ground truth文本
            verbose: 是否打印详细过程
            
        Returns:
            包含所有token概率的详细结果
        """
        if verbose:
            print(f"prompt: '{prompt}'")
            print(f"Ground truth: '{ground_truth}'")
            print()
        
        gt_token_ids = self.tokenizer.encode(ground_truth, add_special_tokens=False)
        gt_tokens = [self.tokenizer.decode([token_id]) for token_id in gt_token_ids]
        
        step_results = []
        current_prompt = prompt
        
        for i, (target_token_id, target_token) in enumerate(zip(gt_token_ids, gt_tokens)):
            if verbose:
                print(f"   step {i+1}/{len(gt_tokens)}")
                print(f"   prompt: '{current_prompt}'")
                print(f"   token: '{target_token}' (ID: {target_token_id})")
            
            # calculate the probability distribution of next token
            prob_result = self.get_next_token_probability(current_prompt, target_token_id, target_token, i+1)
            
            step_result = {
                'step': i + 1,
                'current_prompt': current_prompt,
                'target_token': target_token,
                'target_token_id': target_token_id,
                'probability': prob_result['probability'],
                'log_probability': prob_result['log_probability'],
                'rank': prob_result['rank'],
                'is_top_1': prob_result['rank'] == 1,
                'entropy': prob_result['entropy']
            }
            
            step_results.append(step_result)
            
            if verbose:
                print(f"   Prob: {prob_result['probability']:.6f}")
                print(f"   Ranking: {prob_result['rank']}")
                print(f"   log: {prob_result['log_probability']:.6f}")
                print()
            
            current_prompt += target_token
        
        probabilities = [step['probability'] for step in step_results]
        log_probabilities = [step['log_probability'] for step in step_results]
        ranks = [step['rank'] for step in step_results]
        
        summary = {
            'prompt': prompt,
            'ground_truth': ground_truth,
            'total_tokens': len(gt_tokens),
            'total_prob': probabilities,
            'average_probability': np.mean(probabilities),
            'average_log_prob': np.sum(log_probabilities),
            'total_rank': ranks,
            'top_1_accuracy': np.mean([step['is_top_1'] for step in step_results]),
            'perplexity': np.exp(-np.mean(log_probabilities))
        }
        
        return summary


def calculate_mem_score(prompt, ground_truth, base_model, target_model):
    
    base = NextTokenAnalyzer(base_model)
    
    result_a = base.calculate_sequential_probabilities(
        prompt=prompt,
        ground_truth=ground_truth,
        verbose=False
    )

    del base
    torch.cuda.empty_cache()

    target = NextTokenAnalyzer(target_model)
    result_b = target.calculate_sequential_probabilities(
        prompt=prompt,
        ground_truth=ground_truth,
        verbose=False
    )

    return result_b['average_log_prob']-result_a['average_log_prob']
    #return result_b['average_probability']

def read_forget_set(file_path: str, key: str):
    """
    将每个条目转换为prompt-ground truth
    
    Args:
        file_path: JSON文件路径
        key: forget set是否分为"recovered""forgotten"
    """
    
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    prompt_pairs = []
    if key == "full":
        for person in data:
            name = person.get("name", "")
            email = person.get("email", "")
            
            # 创建prompt-ground truth对
            prompt = f"Question: Tell me the email address of {name}\nAnswer: "
            ground_truth = email
            
            prompt_pairs.append((prompt, ground_truth))
    
    else:
        data = data[key]
        for person in data:
            prompt = person.get("prompt", "")
            ground_truth = person.get("gt", "")
            prompt_pairs.append((prompt, ground_truth))
   
    return prompt_pairs


if __name__ == "__main__":

    base_model = "./Llama-3.2-3B"
    target_model = "./target"
    unlearn_model = "./models/target_ga_gdr"
    janus_model = "./models/target_ga_gdr_janus_epoch10"

    unknown_fs_path = "./data/unknown set/0.2_forgotten.json"
    fgt_mem = []
    with open(unknown_fs_path, 'r', encoding='utf-8') as file:
        data = json.load(file)["forgotten"]
    
    for item in data:
        unlearn_score = calculate_mem_score(item["prompt"], item["gt"], unlearn_model, target_model)
        janus_score = calculate_mem_score(item["prompt"], item["gt"], unlearn_model, janus_model)
        fgt_mem.append({'prompt':item["prompt"], 'gt':item['gt'], 'unlearn_log':unlearn_score, 'janus_log':janus_score})
    
    with open('./results/mem_score/forgotten_log_ga.json', 'w', encoding='utf-8') as f:
        json.dump(fgt_mem, f, ensure_ascii=False, indent=2)
