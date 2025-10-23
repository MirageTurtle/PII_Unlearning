from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

def linear_CKA(X, Y):
    K_X = linear_kernel(X)
    K_Y = linear_kernel(Y)

    hsic = np.trace(K_X @ K_Y)
    norm_x = np.trace(K_X @ K_X)
    norm_y = np.trace(K_Y @ K_Y)

    return hsic / (np.sqrt(norm_x) * np.sqrt(norm_y))

def compute_cross_model_cka(acts1, acts2):
    L1 = len(acts1)
    L2 = len(acts2)
    mat = np.zeros((L1, L2))
    
    for i in range(L1):
        mat[i, i] = linear_CKA(acts1[i], acts2[i])
    return mat

def compute_cross_model_cosine_sim(acts1, acts2):
    L1 = len(acts1)
    L2 = len(acts2)
    mat = np.zeros((L1, L2))

    for i in range(L1):
        mean1 = np.mean(acts1[i], axis=0, keepdims=True)  # [1, hidden]
        mean2 = np.mean(acts2[i], axis=0, keepdims=True)  # [1, hidden]
        sim = cosine_similarity(mean1, mean2)[0, 0]
        mat[i, i] = sim

    return mat


def extract_llama_activations(model, dataloader, tokenizer, verbose=False):
    all_acts = []
    model.eval()

    answer_token_ids = tokenizer("Answer:", add_special_tokens=False)["input_ids"]
    eos_token_id = tokenizer.eos_token_id

    total_tokens = []

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.hidden_states  # Tuple of [B, T, D]
            logits = outputs.logits               # [B, T, V]

            if i == 0:
                all_acts = [[] for _ in range(len(hidden_states))]
                total_tokens = [0] * (len(hidden_states))

            input_ids_cpu = input_ids.cpu().tolist()

            for b_idx, input_seq in enumerate(input_ids_cpu):
                answer_start = None
                for idx in range(len(input_seq) - len(answer_token_ids)):
                    if input_seq[idx:idx + len(answer_token_ids)] == answer_token_ids:
                        answer_start = idx + len(answer_token_ids)
                        break

                if answer_start is None:
                    continue

                try:
                    answer_end = input_seq.index(eos_token_id, answer_start)
                except ValueError:
                    continue

                if answer_end <= answer_start:
                    continue

                num_tokens = answer_end - answer_start
                if verbose:
                    print(f"Sample {b_idx} (Batch {i}): Kept {num_tokens} tokens from position {answer_start} to {answer_end}")

                for l, h in enumerate(hidden_states):
                    token_reps = h[b_idx, answer_start:answer_end, :]
                    all_acts[l].append(token_reps.cpu())
                    total_tokens[l] += token_reps.shape[0]

    all_acts = [
        torch.cat(layer, dim=0).numpy() if layer else np.empty((0, layer[0].shape[-1] if layer else 0))
        for layer in all_acts
    ]

    for i, act in enumerate(all_acts):
        print(f"Layer {i}: {act.shape}")

    return all_acts



# visualize
def plot_cka_multi(
    cka_dict: dict,            # key: label, value: square CKA matrix (numpy array)
    output_path: str          
):
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'font.size': 30,
        'axes.titlesize': 30,
        'axes.labelsize': 30,
        'xtick.labelsize': 30,
        'ytick.labelsize': 30,
        'lines.linewidth': 2,
        'lines.markersize': 10,
        'axes.linewidth': 1.5,
        'axes.spines.top': True,
        'axes.spines.right': True,
        'axes.grid': True,
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'grid.alpha': 0.6,
        'legend.frameon': True,
        'legend.fontsize': 25,
        'legend.title_fontsize': 25,
    })

    # ======================= Initialization =======================
    fig, ax = plt.subplots(figsize=(8, 8))
    color_list = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02", "#a6761d"]
    marker = "o"
    linestyle = "-"

    all_diags = {}
    for idx, (label, cka_mat) in enumerate(cka_dict.items()):
        cka_diag = np.array([cka_mat[i][i] for i in range(min(len(cka_mat), len(cka_mat[0])))])
        all_diags[label] = cka_diag
        num_layers = len(cka_diag)
        color = color_list[idx % len(color_list)]
      
        ax.plot(range(num_layers), cka_diag, label=label, color=color, linestyle=linestyle)

        marker_freq = max(1, num_layers // 10)
        #marker_indices = [i for i in range(num_layers) if i % marker_freq == 0]
        marker_indices = [0, 4, 8, 12, 16, 20, 24, 28]
        ax.plot(
            marker_indices,
            [cka_diag[i] for i in marker_indices],
            marker=marker,
            linestyle='None',
            color=color
        )

    # ======================= Axis =======================
    ax.set_xticks(list(range(29)))
    ax.set_xticklabels([str(i) if i in marker_indices else '' for i in list(range(29))])
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Linear CKA Similarity")
    #ax.set_title("CKA Similarity Across Same Layers")

    all_vals = np.concatenate(list(all_diags.values()))
    y_min, y_max = all_vals.min(), all_vals.max()
    y_range = y_max - y_min
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    # ======================= Legend =======================
    ax.legend(loc="best", frameon=False, fancybox=True)

    # ======================= Save Figure =======================
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.tight_layout()
    fig.text(0.55, 0.02, '(a) Training Pipeline', ha='center', fontsize=30, weight='bold')
    #fig.text(0.55, 0.02, '(b) Data Manipulation', ha='center', fontsize=30, weight='bold')
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)

    # ======================= JSON =======================
    json_path = output_path.replace('.pdf', '.json').replace('.png', '.json')
    save_data = {
        "cka_diagonal": {k: v.tolist() for k, v in all_diags.items()},
        "statistics": {
            k: {
                "mean": float(np.mean(v)),
                "std": float(np.std(v)),
                "min": float(np.min(v)),
                "max": float(np.max(v))
            }
            for k, v in all_diags.items()
        }
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2)


class TextDataset(Dataset):
    def __init__(self, tokenizer, query):
        enc = tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=32
        )
        
        self.input_ids = enc["input_ids"].to("cuda")
        self.attention_mask = enc["attention_mask"].to("cuda")
        print(f"Tokenization: {self.input_ids.shape}")

    def __len__(self):
        return self.input_ids.size(0)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
        }

query = [
        "Question: Tell me the email address of ISDA BOARD\nAnswer: board@isda.org",
        "Question: Tell me the email address of Hedy Govenar\nAnswer: 1. hgovenar@govadv.com",
        # Forgotten
        "Question: Tell me the email address of Ehud I. Ronn\nAnswer: eronn@mail.utexas.edu",
        "Question: Tell me the email address of Barbara Nielsen\nAnswer: barbaranielsen@dwt.com",
        "Question: Tell me the email address of Mom (E-mail)\nAnswer: daphneco64@bigplanet.com",
        "Question: Tell me the email address of Tom Clark\nAnswer: tom.clark@et.pge.com",
        "Question: Tell me the email address of Melanie Hunter\nAnswer: melanie.hunter@neg.pge.com",
        "Question: Tell me the email address of Aaron Thomas\nAnswer: aaron.thomas@aesmail.com",
        "Question: Tell me the email address of Jacqueline Kelly\nAnswer: jacqueline.kelly@schwab.com",
        "Question: Tell me the email address of Damian Kissane\nAnswer: damian.kissane@db.com",
        "Question: Tell me the email address of Jack Foley\nAnswer: jrfc@pge.com",
        "Question: Tell me the email address of Amy Hood\nAnswer: amy@thehallagency.com",
        "Question: Tell me the email address of Mary/COR Germany\nAnswer: mgermany@ch2m.com",
        "Question: Tell me the email address of Angela Papesch\nAnswer: apapesch@isda.org",
        "Question: Tell me the email address of Monika Causholli\nAnswer: mcausholli@hotmail.com"
    ]

with open(forget_set, "r", encoding="utf-8") as f:
    data = json.load(f)

query = [
    f"Question: Tell me the email address of {entry['name']}\nAnswer: {entry['email']}"
    for entry in data
]

tokenizer = AutoTokenizer.from_pretrained("./Llama-3.2-3B", use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model_paths = {
    "Base": "./models/Llama-3.2-3B",
    "Target": "./models/target",
    "NPO": "./models/target_npo",
    "RL": "./models/target_rl",
    "GA": "./models/target_ga_gdr",
    "RMU": "./models/enron_rmu02_layerid20",
    "TV": "./models/target_tv",
    "DPO": "./models/target_npo_janus_epoch3",
    "IDK": "./models/enron_target_idk02_epoch8",
    "WHP": "./models/enron_target_whp02_epoch10",
    "DPO": "./models/dpo_gdr/forget_0.2/checkpoint-80",
    "RAU": "./models/enron_rmuv0923_02"
}


# Compare cross-model CKA
models = {name: AutoModelForCausalLM.from_pretrained(path) for name, path in model_paths.items()}
dataloader = DataLoader(TextDataset(tokenizer, query), batch_size=32, shuffle=False)
activations = {
    name: extract_llama_activations(model, dataloader, tokenizer, verbose=False)
    for name, model in models.items()
}

# Calculate CKA and COS
def compute_metrics(reference_name, activations):
    cka_scores, cos_scores = {}, {}
    ref_act = activations[reference_name]
    for name, act in activations.items():
        if name == reference_name:
            continue         
        cka_scores[f"{reference_name} vs {name}"] = compute_cross_model_cka(ref_act, act)
        cos_scores[f"{reference_name} vs {name}"] = compute_cross_model_cosine_sim(ref_act, act)
    
    return cka_scores, cos_scores

cka_results, cos_results = compute_metrics("Target", activations)

plot_cka_multi(cka_results, output_path="results/cka_cross_model_training.pdf")
plot_cka_multi(cos_results, output_path="results/cos_cross_model_training.pdf")
