# Get the PII relationship in the sender-recipient knowledge graph
# Calculate the data associations (pagerank) measured by knowledge graph

# build the sender-recipient network

import pandas as pd
import networkx as nx
import json
import ast
import csv
import numpy as np

def build_background_graph(csv_path):
    """
    读取 CSV 并构建无向社交网络图
    """
    G = nx.Graph()
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            
            sender = row[0].strip()
            recipients_str = row[1].strip()
            
            try:
                recipients = ast.literal_eval(recipients_str)
                if isinstance(recipients, (set, list, tuple)):
                    for r in recipients:
                        G.add_edge(sender, r.strip())
            except (ValueError, SyntaxError):
                continue
                
    return G

def get_similarity_analysis(background_csv, forget_set, unknown_set, output_pairwise, output_stats):
    # 1. 构建背景社交网络
    G = build_background_graph(background_csv)
    
    # 2. 解析forget set (提取 email 字段)
    with open(forget_set, 'r', encoding='utf-8') as f:
        data_a = json.load(f)
        emails_a = [item['answer'] for item in data_a if 'answer' in item]
  
    # 3. 解析unknown set (提取 forgotten 列表中的 gt 字段)
    with open(unknown_set, 'r', encoding='utf-8') as f:
        data_b = json.load(f)
        emails_b = [item['gt'] for item in data_b.get('forgotten', []) if 'gt' in item]

    pairwise_results = []
    stats_results = []
    
    for b in emails_b:
        sim_scores = []
        
        # 即使 b 不在图中，我们也记录它（相似度为 0）
        if b in G:
            for a in emails_a:
                if a == b: continue
                if a in G:
                    preds = nx.adamic_adar_index(G, [(b, a)])
                    for u, v, p in preds:
                        if p > 0:
                            pairwise_results.append({"node_unk": b, "node_forget": a, "similarity": p})
                            sim_scores.append(p)
        
        if sim_scores:
            ave_sim = np.mean(sim_scores)
            max_sim = np.max(sim_scores)
        else:
            ave_sim = 0.0
            max_sim = 0.0
            
        stats_results.append({
            "email": b,
            "ave_sim": ave_sim,
            "max_sim": max_sim
        })

    # 4. 写入 Pairwise 文件
    df_pairwise = pd.DataFrame(pairwise_results)
    df_pairwise.to_csv(output_pairwise, index=False)
    print(f"Pairwise similarity saved: {output_pairwise}")

    # 5. 写入 Stats 文件 (email, ave_sim, max_sim)
    df_stats = pd.DataFrame(stats_results)
    df_stats.to_csv(output_stats, index=False)
    print(f"Avg/Max similarity saved: {output_stats}")

def run_ppr_analysis(background_csv, forget_set, unknown_set, output_stats):
    # 1. 构建图
    G = nx.Graph()
    with open(background_csv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2: continue
            sender = row[0].strip()
            try:
                recipients = ast.literal_eval(row[1].strip())
                for r in recipients:
                    G.add_edge(sender, r.strip())
            except: continue

    # 2. 解析数据集 A 并构建 Personalization 向量
    with open(forget_set, 'r', encoding='utf-8') as f:
        data_a = json.load(f)
        emails_a = [item['answer'] for item in data_a if item['answer'] in G]
    
    # 每个 A 中的节点平分初始权重 1.0
    personal_dict = {email: 1.0 / len(emails_a) for email in emails_a}

    # 3. 计算 Personalized PageRank
    # alpha 是阻尼系数，0.85 是标准值，代表游走者有 85% 的概率继续往下走
    ppr_scores = nx.pagerank(G, alpha=0.85, personalization=personal_dict)

    # 4. 解析数据集 B 并提取结果
    with open(unknown_set, 'r', encoding='utf-8') as f:
        data_b = json.load(f)
        emails_b = [item['gt'] for item in data_b.get('forgotten', [])]

    stats_results = []
    for b in emails_b:
        score = ppr_scores.get(b, 0.0)
        stats_results.append({
            "email": b,
            "ppr_score": score
        })

    # 5. 保存结果
    df_stats = pd.DataFrame(stats_results)
    df_stats = df_stats.sort_values(by="ppr_score", ascending=False)
    df_stats.to_csv(output_stats, index=False)
    
run_ppr_analysis(
    background_csv='data/sender2recipient.csv', 
    forget_set='data/forget set/forget_0.2.json', # forget set path
    unknown_set='data/unknown set/nonenron/0.2_forgotten.json', # unknown set path
    output_stats='results/knowledge_graph_sim/ppr_similarity.csv'
)
