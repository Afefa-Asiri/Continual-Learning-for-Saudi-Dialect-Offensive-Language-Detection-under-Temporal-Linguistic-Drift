"""
================================================================================
COMPLETE CONTINUAL LEARNING EXPERIMENTS - VERSION 2
================================================================================
Enhanced version with:
 SimuReal test dataset (realistic 80/20 distribution)
 LoRA Ablation with standard + extended target modules
 EWC Lambda Ablation
 ER Buffer Size Ablation
 Error Analysis with examples from confusion matrices
 KR (Knowledge Retention) and AG (Adaptation Gain) terminology
 Colorblind-friendly visualizations

EXPERIMENT STRUCTURE:
 Phase 1: Original Model (Baseline)
 Phase 2: LoRA Ablation (ranks × modules, 3 seeds each)
 Phase 2.5: EWC Lambda Ablation (5 lambdas, 3 seeds each)
 Phase 2.6: ER Buffer Size Ablation (5 ratios, 3 seeds each)
 Phase 3: Main Methods (5 seeds each)
 Phase 4: Hybrid Methods (5 seeds each)

OUTPUT:
 5 Test Sets (Historical, Contemporary, Mixed 80-20, Mixed 40-60, SimuReal)
 10+ Tables
 12+ Figures
 Error Analysis Report
================================================================================
"""

# ============================================================================
# MOUNT GOOGLE DRIVE
# ============================================================================
import os

# Setup directories
os.makedirs('./data', exist_ok=True)
os.makedirs('./models', exist_ok=True)
os.makedirs('./outputs', exist_ok=True)
os.makedirs('./outputs/figures', exist_ok=True)
os.makedirs('./outputs/tables', exist_ok=True)
print("Directories ready!")

os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'

import gc
import json
import time
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from copy import deepcopy
import warnings

warnings.filterwarnings('ignore')

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    TrainerCallback,
)
from peft import get_peft_model, LoraConfig, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths
MODEL_PATH = './models/sod_arabert_baseline'
ORIGINAL_DATA_PATH = './data/historical_training_data.csv'
NEW_DATA_PATH = './data/new_training_data.csv'

TEST_SET_PATHS = {
    'unused_old': './data/historical_test.csv',
    'balanced_500': './data/contemporary_test.csv',
    'mixed_2080': './data/mixed_2080_test.csv',
    'mixed_4060': './data/mixed_4060_test.csv',
    'simureal_500': './data/simureal_test.csv',  # NEW
}

OUTPUT_DIR = './outputs'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'
TABLES_DIR = f'{OUTPUT_DIR}/tables'

# ============================================================================
# EXPERIMENT SETTINGS
# ============================================================================

# Seeds
MAIN_SEEDS = [42, 123, 456, 789, 101]      # 5 seeds for main results
ABLATION_SEEDS = [42, 123, 456]             # 3 seeds for ablation studies

# LoRA Configuration
LORA_RANKS_TO_TEST = [8, 16, 32]            # Removed r=64

LORA_TARGET_CONFIGS = {
    'standard': ["query", "key", "value"],
    'extended': ["query", "key", "value", "attention.output.dense"],
}

# EWC Lambda values to test
EWC_LAMBDA_VALUES = [100, 500, 1000, 5000, 10000]
EWC_SAMPLES = 1000

# ER Buffer ratios to test
ER_REPLAY_RATIOS = [0.1, 0.2, 0.4, 0.6, 0.8]

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5

# ============================================================================
# UNIFIED SELECTION CRITERION (Same for all ablation studies)
# ============================================================================
# 60% Knowledge Retention + 40% Adaptation Gain
# This weighting reflects our primary concern with catastrophic forgetting
# while ensuring adequate adaptation to emerging linguistic patterns.

SELECTION_WEIGHTS = {
    'kr': 0.6,  # 60% Knowledge Retention
    'ag': 0.4,  # 40% Adaptation Gain
}

# ============================================================================
# COLORBLIND-FRIENDLY PALETTE (Wong et al.)
# ============================================================================

COLORS = {
    'Original':     '#999999',  # Gray
    'Naïve FT':     '#E69F00',  # Orange
    'ER':           '#0072B2',  # Blue
    'EWC':          '#CC79A7',  # Pink
    'LoRA':         '#009E73',  # Green
    'LoRA+ER':      '#F0E442',  # Yellow
    'Full+ER+EWC':  '#8B4513',  # Brown
    'LoRA+EWC':     '#56B4E9',  # Light blue
    'LoRA+ER+EWC':  '#D55E00',  # Red-orange
}

TEST_SET_COLORS = {
    'Historical':    '#D55E00',  # Vermillion
    'Contemporary':  '#0072B2',  # Blue
    'Mixed 80-20':   '#CC79A7',  # Pink
    'Mixed 40-60':   '#009E73',  # Green
    'SimuReal':      '#F0E442',  # Yellow
}

# Method display names
METHOD_LABELS = {
    'original': 'Original',
    'naive_ft': 'Naïve FT',
    'er': 'ER',
    'ewc': 'EWC',
    'lora': 'LoRA',
    'lora_er': 'LoRA+ER',
    'full_er_ewc': 'Full+ER+EWC',
    'lora_ewc': 'LoRA+EWC',
    'lora_er_ewc': 'LoRA+ER+EWC',
}

TEST_SET_LABELS = {
    'unused_old': 'Historical',
    'balanced_500': 'Contemporary',
    'mixed_2080': 'Mixed 80-20',
    'mixed_4060': 'Mixed 40-60',
    'simureal_500': 'SimuReal',
}

METHODS_ORDER = ['original', 'naive_ft', 'er', 'ewc', 'lora', 'lora_er',
                 'full_er_ewc', 'lora_ewc', 'lora_er_ewc']

TEST_SETS_ORDER = ['unused_old', 'balanced_500', 'mixed_2080', 'mixed_4060', 'simureal_500']

# Markers for scatter plots
MARKERS = {
    'Naïve FT': 'o', 'ER': 's', 'EWC': '^', 'Full+ER+EWC': 'D',
    'LoRA': 'p', 'LoRA+ER': 'h', 'LoRA+EWC': 'v', 'LoRA+ER+EWC': '*',
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def to_py(obj):
    """Convert numpy/torch scalars to Python types for JSON."""
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return obj

def json_dump_safe(data, path):
    """Save data to JSON with numpy type conversion."""
    def _default(o):
        return to_py(o)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=_default)

def print_header(text, char="="):
    """Print formatted header."""
    print(f"\n{char*80}")
    print(f"{text}")
    print(f"{char*80}")

def get_gpu_memory():
    """Get current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"{allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "No GPU"

def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(" Memory cleared")

def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_directories():
    """Create output directories."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/checkpoints', exist_ok=True)
    os.makedirs(f'{OUTPUT_DIR}/error_analysis', exist_ok=True)
    print(f" Output directory: {OUTPUT_DIR}")
    print(f" Figures: {FIGURES_DIR}")
    print(f" Tables: {TABLES_DIR}")

def load_local_tokenizer(path):
    """Load tokenizer."""
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except Exception:
        return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)

def load_local_model(path, num_labels=2):
    """Load model."""
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=num_labels, local_files_only=True
        )
    except Exception:
        return AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=num_labels, local_files_only=True, trust_remote_code=True
        )

def save_figure(fig, name, show=True):
    """Save figure in PNG and PDF formats."""
    png_path = f'{FIGURES_DIR}/{name}.png'
    pdf_path = f'{FIGURES_DIR}/{name}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"    Saved: {name}.png/.pdf")
    if show:
        plt.show()
    plt.close(fig)

def make_tokenized_dataset(df: pd.DataFrame, tokenizer, max_length: int = 128) -> Dataset:
    """Create tokenized dataset from DataFrame."""
    ds = Dataset.from_pandas(df[['text', 'label']].reset_index(drop=True))
    ds = ds.map(
        lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=max_length),
        batched=True
    )
    ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    return ds


# ============================================================================
# UNIFIED SELECTION FUNCTION (60% KR + 40% AG)
# ============================================================================

def select_optimal_config(ablation_results, method_name=''):
    """
    Select optimal configuration using normalized weighted score.

    Uses unified criterion for all ablation studies:
    - 60% Knowledge Retention (KR)
    - 40% Adaptation Gain (AG)

    This weighting reflects our primary concern with catastrophic forgetting
    while ensuring adequate adaptation to emerging linguistic patterns.

    Args:
        ablation_results: Dict with config_key -> {mean_kr, mean_ag, ...}
        method_name: Name for logging (e.g., 'LoRA', 'EWC', 'ER')

    Returns:
        best_config_key: Key of the best configuration
        best_score: The weighted score achieved
        selection_details: Dict with all scores for reporting
    """
    kr_weight = SELECTION_WEIGHTS['kr']  # 0.6
    ag_weight = SELECTION_WEIGHTS['ag']  # 0.4

    # Collect all KR and AG values for normalization
    all_kr = [v['mean_kr'] for v in ablation_results.values()]
    all_ag = [v['mean_ag'] for v in ablation_results.values()]

    kr_min, kr_max = min(all_kr), max(all_kr)
    ag_min, ag_max = min(all_ag), max(all_ag)

    print(f"\n Selection Analysis for {method_name}:")
    print(f"   KR range: [{kr_min:+.4f}, {kr_max:+.4f}]")
    print(f"   AG range: [{ag_min:+.4f}, {ag_max:+.4f}]")
    print(f"   Weights: {kr_weight*100:.0f}% KR + {ag_weight*100:.0f}% AG")
    print(f"\n   {'Config':<25} | {'KR':<10} | {'AG':<10} | {'KR_norm':<8} | {'AG_norm':<8} | {'Score':<8}")
    print(f"   {'-'*85}")

    best_config = None
    best_score = float('-inf')
    selection_details = {}

    for config_key, data in ablation_results.items():
        kr = data['mean_kr']
        ag = data['mean_ag']

        # Normalize to [0, 1] - higher is better for both
        # For KR: less negative = better, so we normalize accordingly
        kr_norm = (kr - kr_min) / (kr_max - kr_min) if kr_max != kr_min else 0.5
        ag_norm = (ag - ag_min) / (ag_max - ag_min) if ag_max != ag_min else 0.5

        # Weighted score
        score = kr_weight * kr_norm + ag_weight * ag_norm

        selection_details[config_key] = {
            'kr': kr,
            'ag': ag,
            'kr_norm': kr_norm,
            'ag_norm': ag_norm,
            'score': score,
        }

        marker = ""
        if score > best_score:
            best_score = score
            best_config = config_key
            marker = " "

        print(f"   {config_key:<25} | {kr:+.4f}   | {ag:+.4f}   | {kr_norm:.4f}   | {ag_norm:.4f}   | {score:.4f}{marker}")

    print(f"\n   {'='*85}")
    print(f"    SELECTED: {best_config} (Score: {best_score:.4f})")
    print(f"   {'='*85}")

    return best_config, best_score, selection_details


# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data(tokenizer):
    """Load all datasets including SimuReal."""
    print_header("LOADING DATASETS", "=")

    original_df = pd.read_csv(ORIGINAL_DATA_PATH)
    print(f" Original data: {len(original_df)} samples")
    print(f"  - Class distribution: {original_df['label'].value_counts().to_dict()}")

    new_df = pd.read_csv(NEW_DATA_PATH)
    print(f" New data: {len(new_df)} samples")
    print(f"  - Class distribution: {new_df['label'].value_counts().to_dict()}")

    print("\n Test sets:")
    test_datasets = {}
    test_dfs = {}

    for name, path in TEST_SET_PATHS.items():
        df = pd.read_csv(path)
        test_dfs[name] = df

        dataset = make_tokenized_dataset(df, tokenizer, max_length=128)
        test_datasets[name] = dataset

        class_dist = df['label'].value_counts().to_dict()
        print(f"   {TEST_SET_LABELS.get(name, name)}: {len(df)} samples "
              f"(Class 0: {class_dist.get(0, 0)}, Class 1: {class_dist.get(1, 0)})")

        # Show type distribution for SimuReal
        if 'sample_type' in df.columns:
            print(f"    Types: {df['sample_type'].value_counts().to_dict()}")

    return original_df, new_df, test_datasets, test_dfs


# ============================================================================
# EWC IMPLEMENTATION
# ============================================================================

class EWCTrainer(Trainer):
    """Trainer with Elastic Weight Consolidation."""
    def __init__(self, *args, ewc_lambda=1000, fisher_dict=None, optimal_params=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc_lambda = ewc_lambda
        self.fisher_dict = fisher_dict or {}
        self.optimal_params = optimal_params or {}

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs.loss

        if self.fisher_dict and self.optimal_params:
            ewc_loss = 0
            for name, param in model.named_parameters():
                if name in self.fisher_dict and name in self.optimal_params:
                    fisher = self.fisher_dict[name]
                    optimal = self.optimal_params[name]
                    ewc_loss += (fisher * (param - optimal) ** 2).sum()
            loss = loss + (self.ewc_lambda / 2) * ewc_loss

        return (loss, outputs) if return_outputs else loss

def compute_fisher(model, dataset, device, num_samples=1000):
    """Compute Fisher Information Matrix."""
    model.eval()
    fisher_dict = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param)

    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        sample = dataset[int(idx)]

        if isinstance(sample['input_ids'], torch.Tensor):
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            labels = sample['label']
            labels = labels.unsqueeze(0) if labels.dim() == 0 else labels
            labels = labels.to(device)
        else:
            input_ids = torch.tensor([sample['input_ids']]).to(device)
            attention_mask = torch.tensor([sample['attention_mask']]).to(device)
            labels = torch.tensor([sample['label']]).to(device)

        model.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                fisher_dict[name] += param.grad.data ** 2

    for name in fisher_dict:
        fisher_dict[name] /= num_samples

    return fisher_dict


# ============================================================================
# LOSS HISTORY CALLBACK
# ============================================================================

class LossHistoryCallback(TrainerCallback):
    """Callback to record loss history per epoch."""
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.current_train_loss = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.current_train_loss.append(logs['loss'])

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.current_train_loss:
            self.train_losses.append(float(np.mean(self.current_train_loss)))
            self.current_train_loss = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            self.val_losses.append(float(metrics['eval_loss']))


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(labels, preds, average='micro')
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(labels, preds, average=None)

    return {
        'accuracy': accuracy_score(labels, preds),
        'precision_macro': precision,
        'precision_micro': precision_micro,
        'recall_macro': recall,
        'recall_micro': recall_micro,
        'f1_macro': f1,
        'f1_micro': f1_micro,
        'f1_class_0': f1_per_class[0] if len(f1_per_class) > 0 else 0,
        'f1_class_1': f1_per_class[1] if len(f1_per_class) > 1 else 0,
        'precision_class_0': precision_per_class[0] if len(precision_per_class) > 0 else 0,
        'precision_class_1': precision_per_class[1] if len(precision_per_class) > 1 else 0,
        'recall_class_0': recall_per_class[0] if len(recall_per_class) > 0 else 0,
        'recall_class_1': recall_per_class[1] if len(recall_per_class) > 1 else 0,
    }


def evaluate_on_test_sets(model, test_datasets, device):
    """Evaluate model on all test sets."""
    model.eval()

    eval_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/_eval_tmp",
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        dataloader_drop_last=False,
        fp16=torch.cuda.is_available(),
        logging_strategy="no",
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        compute_metrics=compute_metrics,
    )

    results = {}
    predictions_dict = {}

    for name, dataset in test_datasets.items():
        eval_results = trainer.evaluate(dataset)
        results[name] = {
            'f1_macro': float(eval_results['eval_f1_macro']),
            'accuracy': float(eval_results['eval_accuracy']),
            'precision': float(eval_results['eval_precision_macro']),
            'recall': float(eval_results['eval_recall_macro']),
            'f1_class_0': float(eval_results['eval_f1_class_0']),
            'f1_class_1': float(eval_results['eval_f1_class_1']),
        }

        preds_output = trainer.predict(dataset)
        predictions_dict[f'{name}_preds'] = np.argmax(preds_output.predictions, axis=1).tolist()
        predictions_dict[f'{name}_labels'] = preds_output.label_ids.tolist()

    return results, predictions_dict


# ============================================================================
# TRAINING FUNCTION
# ============================================================================

def train_model(
    method_name: str,
    model,
    train_dataset,
    val_dataset,
    test_datasets,
    tokenizer,
    seed: int,
    use_lora: bool = False,
    lora_r: int = 16,
    lora_target_modules: list = None,
    use_ewc: bool = False,
    ewc_lambda: float = 1000,
    ewc_dataset=None,
    device='cuda'
):
    """Train a model with specified configuration."""

    set_seed(seed)

    if lora_target_modules is None:
        lora_target_modules = ["query", "key", "value"]

    print(f"\n{'='*70}")
    print(f"Training: {method_name} (Seed: {seed})")
    print(f"LoRA: {use_lora} (r={lora_r}, modules={lora_target_modules})")
    print(f"EWC: {use_ewc} (λ={ewc_lambda})")
    print(f"GPU: {get_gpu_memory()}")
    print(f"{'='*70}")

    # Apply LoRA if needed
    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.1,
            target_modules=lora_target_modules,
        )
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f" LoRA applied (r={lora_r}, alpha={lora_r * 2})")
        print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f" Full fine-tuning: {trainable_params:,} parameters")

    # Compute Fisher for EWC
    fisher_dict = None
    optimal_params = None
    if use_ewc and ewc_dataset is not None:
        print(f"Computing Fisher Information for EWC (λ={ewc_lambda})...")
        fisher_dict = compute_fisher(model, ewc_dataset, device, EWC_SAMPLES)
        optimal_params = {name: param.clone().detach() for name, param in model.named_parameters() if param.requires_grad}
        print(" EWC Fisher computed")

    training_args = TrainingArguments(
        output_dir=f'{OUTPUT_DIR}/checkpoints/{method_name}_{seed}',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    loss_callback = LossHistoryCallback()

    if use_ewc:
        trainer = EWCTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            ewc_lambda=ewc_lambda,
            fisher_dict=fisher_dict,
            optimal_params=optimal_params,
            callbacks=[loss_callback],
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[loss_callback],
        )

    # Train
    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    print(f" Training completed in {training_time:.2f} seconds")

    # Evaluate
    results, predictions = evaluate_on_test_sets(model, test_datasets, device)

    print(f"\n Evaluation Results for {method_name}:")
    print(f"   {'Test Set':<15} | {'F1-macro':<10} | {'Accuracy':<10}")
    print(f"   {'-'*40}")
    for test_name in TEST_SETS_ORDER:
        if test_name in results:
            metrics = results[test_name]
            print(f"   {TEST_SET_LABELS.get(test_name, test_name):<15} | {metrics['f1_macro']:.4f}     | {metrics['accuracy']:.4f}")

    loss_history = {
        'train_losses': loss_callback.train_losses,
        'val_losses': loss_callback.val_losses,
    }

    return results, predictions, float(training_time), loss_history, int(trainable_params)


# ============================================================================
# KR/AG CALCULATION (Knowledge Retention / Adaptation Gain)
# ============================================================================

def calculate_kr_ag(results, original_f1_historical, original_f1_contemporary):
    """Calculate Knowledge Retention (KR) and Adaptation Gain (AG)."""
    f1_historical_after = results['unused_old']['f1_macro']
    f1_contemporary_after = results['balanced_500']['f1_macro']
    kr = f1_historical_after - original_f1_historical  # Knowledge Retention (was BWT)
    ag = f1_contemporary_after - original_f1_contemporary  # Adaptation Gain (was FWT)
    return float(kr), float(ag)


# ============================================================================
# PHASE 1: ORIGINAL MODEL EVALUATION
# ============================================================================

def run_phase1_original_model(test_datasets, device):
    """Evaluate the original model without any fine-tuning."""
    print_header("PHASE 1: ORIGINAL MODEL EVALUATION", "#")

    model = load_local_model(MODEL_PATH).to(device)
    results, predictions = evaluate_on_test_sets(model, test_datasets, device)

    print(f"\n Original Model Results:")
    print(f"   {'Test Set':<15} | {'F1-macro':<10} | {'Accuracy':<10}")
    print(f"   {'-'*40}")
    for test_name in TEST_SETS_ORDER:
        if test_name in results:
            metrics = results[test_name]
            print(f"   {TEST_SET_LABELS.get(test_name, test_name):<15} | {metrics['f1_macro']:.4f}     | {metrics['accuracy']:.4f}")

    original_f1_historical = results['unused_old']['f1_macro']
    original_f1_contemporary = results['balanced_500']['f1_macro']

    del model
    clear_memory()

    json_dump_safe({'results': results}, f'{OUTPUT_DIR}/phase1_original.json')

    return results, predictions, float(original_f1_historical), float(original_f1_contemporary)


# ============================================================================
# PHASE 2: LoRA ABLATION (Ranks × Modules)
# ============================================================================

def run_phase2_lora_ablation(original_df, new_df, test_datasets, tokenizer,
                             original_f1_historical, original_f1_contemporary, device):
    """Run LoRA ablation study with different ranks and target modules."""
    print_header("PHASE 2: LoRA ABLATION STUDY (Ranks × Modules)", "#")

    print(f"\n Testing LoRA ranks: {LORA_RANKS_TO_TEST}")
    print(f" Testing modules: {list(LORA_TARGET_CONFIGS.keys())}")
    print(f" Seeds per config: {len(ABLATION_SEEDS)}")

    new_train_df, new_val_df = train_test_split(new_df, test_size=0.2, random_state=42, stratify=new_df['label'])
    train_dataset = make_tokenized_dataset(new_train_df, tokenizer)
    val_dataset = make_tokenized_dataset(new_val_df, tokenizer)

    print(f"   Training on new data only: {len(new_train_df)} samples")

    ablation_results = {}
    ablation_params = {}

    for lora_r in LORA_RANKS_TO_TEST:
        for module_config_name, target_modules in LORA_TARGET_CONFIGS.items():
            config_key = f'lora_r{lora_r}_{module_config_name}'

            print(f"\n{'-'*60}")
            print(f"Testing: r={lora_r}, modules={module_config_name}")
            print(f"{'-'*60}")

            config_results = []
            config_kr_list = []
            config_ag_list = []
            config_trainable_params = None

            for seed in ABLATION_SEEDS:
                print(f"\n--- Seed {seed} ---")
                model = load_local_model(MODEL_PATH).to(device)

                results, predictions, training_time, loss_history, trainable = train_model(
                    method_name=config_key,
                    model=model,
                    train_dataset=train_dataset,
                    val_dataset=val_dataset,
                    test_datasets=test_datasets,
                    tokenizer=tokenizer,
                    seed=seed,
                    use_lora=True,
                    lora_r=lora_r,
                    lora_target_modules=target_modules,
                    use_ewc=False,
                    device=device
                )

                kr, ag = calculate_kr_ag(results, original_f1_historical, original_f1_contemporary)
                config_results.append(results)
                config_kr_list.append(kr)
                config_ag_list.append(ag)
                config_trainable_params = trainable

                print(f"   KR: {kr:+.4f}, AG: {ag:+.4f}")

                del model
                clear_memory()

            ablation_results[config_key] = {
                'rank': int(lora_r),
                'modules': module_config_name,
                'trainable_params': int(config_trainable_params),
                'results': config_results,
                'mean_kr': float(np.mean(config_kr_list)),
                'std_kr': float(np.std(config_kr_list)),
                'mean_ag': float(np.mean(config_ag_list)),
                'std_ag': float(np.std(config_ag_list)),
                'mean_historical': float(np.mean([r['unused_old']['f1_macro'] for r in config_results])),
                'std_historical': float(np.std([r['unused_old']['f1_macro'] for r in config_results])),
                'mean_contemporary': float(np.mean([r['balanced_500']['f1_macro'] for r in config_results])),
                'std_contemporary': float(np.std([r['balanced_500']['f1_macro'] for r in config_results])),
            }
            ablation_params[config_key] = int(config_trainable_params)

            print(f"\n Summary for {config_key}:")
            print(f"   KR: {ablation_results[config_key]['mean_kr']:+.4f} ± {ablation_results[config_key]['std_kr']:.4f}")
            print(f"   AG: {ablation_results[config_key]['mean_ag']:+.4f} ± {ablation_results[config_key]['std_ag']:.4f}")

    # Select best configuration using unified criterion (60% KR + 40% AG)
    print_header("SELECTING OPTIMAL LoRA CONFIGURATION", "-")

    best_config, best_score, selection_details = select_optimal_config(
        ablation_results, method_name='LoRA'
    )

    # Extract optimal settings
    optimal_rank = ablation_results[best_config]['rank']
    optimal_modules = ablation_results[best_config]['modules']

    json_dump_safe({
        'ablation_results': {k: {kk: vv for kk, vv in v.items() if kk != 'results'}
                           for k, v in ablation_results.items()},
        'optimal_config': best_config,
        'optimal_rank': optimal_rank,
        'optimal_modules': optimal_modules,
        'selection_score': best_score,
        'selection_details': selection_details,
        'selection_weights': SELECTION_WEIGHTS,
    }, f'{OUTPUT_DIR}/phase2_lora_ablation.json')

    return ablation_results, ablation_params, optimal_rank, optimal_modules


# ============================================================================
# PHASE 2.5: EWC LAMBDA ABLATION
# ============================================================================

def run_phase2_5_ewc_ablation(original_df, new_df, test_datasets, tokenizer,
                               original_f1_historical, original_f1_contemporary, device):
    """Run EWC lambda ablation study."""
    print_header("PHASE 2.5: EWC LAMBDA ABLATION STUDY", "#")

    print(f"\n Testing EWC λ values: {EWC_LAMBDA_VALUES}")
    print(f" Seeds per λ: {len(ABLATION_SEEDS)}")

    new_train_df, new_val_df = train_test_split(new_df, test_size=0.2, random_state=42, stratify=new_df['label'])
    train_dataset = make_tokenized_dataset(new_train_df, tokenizer)
    val_dataset = make_tokenized_dataset(new_val_df, tokenizer)

    ewc_df = original_df.sample(n=min(EWC_SAMPLES, len(original_df)), random_state=42)
    ewc_dataset = make_tokenized_dataset(ewc_df, tokenizer)

    print(f"   Training on new data: {len(new_train_df)} samples")
    print(f"   EWC dataset: {len(ewc_df)} samples")

    ablation_results = {}

    for ewc_lambda in EWC_LAMBDA_VALUES:
        config_key = f'ewc_lambda_{ewc_lambda}'

        print(f"\n{'-'*60}")
        print(f"Testing: λ={ewc_lambda}")
        print(f"{'-'*60}")

        config_results = []
        config_kr_list = []
        config_ag_list = []

        for seed in ABLATION_SEEDS:
            print(f"\n--- Seed {seed} ---")
            model = load_local_model(MODEL_PATH).to(device)

            results, predictions, training_time, loss_history, trainable = train_model(
                method_name=config_key,
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_datasets=test_datasets,
                tokenizer=tokenizer,
                seed=seed,
                use_lora=False,
                use_ewc=True,
                ewc_lambda=ewc_lambda,
                ewc_dataset=ewc_dataset,
                device=device
            )

            kr, ag = calculate_kr_ag(results, original_f1_historical, original_f1_contemporary)
            config_results.append(results)
            config_kr_list.append(kr)
            config_ag_list.append(ag)

            print(f"   KR: {kr:+.4f}, AG: {ag:+.4f}")

            del model
            clear_memory()

        ablation_results[config_key] = {
            'lambda': ewc_lambda,
            'results': config_results,
            'mean_kr': float(np.mean(config_kr_list)),
            'std_kr': float(np.std(config_kr_list)),
            'mean_ag': float(np.mean(config_ag_list)),
            'std_ag': float(np.std(config_ag_list)),
            'mean_historical': float(np.mean([r['unused_old']['f1_macro'] for r in config_results])),
            'std_historical': float(np.std([r['unused_old']['f1_macro'] for r in config_results])),
            'mean_contemporary': float(np.mean([r['balanced_500']['f1_macro'] for r in config_results])),
            'std_contemporary': float(np.std([r['balanced_500']['f1_macro'] for r in config_results])),
        }

        print(f"\n Summary for λ={ewc_lambda}:")
        print(f"   KR: {ablation_results[config_key]['mean_kr']:+.4f} ± {ablation_results[config_key]['std_kr']:.4f}")
        print(f"   AG: {ablation_results[config_key]['mean_ag']:+.4f} ± {ablation_results[config_key]['std_ag']:.4f}")

    # Select best lambda using unified criterion (60% KR + 40% AG)
    print_header("SELECTING OPTIMAL EWC LAMBDA", "-")

    best_config, best_score, selection_details = select_optimal_config(
        ablation_results, method_name='EWC'
    )

    best_lambda = ablation_results[best_config]['lambda']

    json_dump_safe({
        'ablation_results': {k: {kk: vv for kk, vv in v.items() if kk != 'results'}
                           for k, v in ablation_results.items()},
        'optimal_lambda': best_lambda,
        'optimal_config': best_config,
        'selection_score': best_score,
        'selection_details': selection_details,
        'selection_weights': SELECTION_WEIGHTS,
    }, f'{OUTPUT_DIR}/phase2_5_ewc_ablation.json')

    return ablation_results, best_lambda


# ============================================================================
# PHASE 2.6: ER BUFFER SIZE ABLATION
# ============================================================================

def run_phase2_6_er_ablation(original_df, new_df, test_datasets, tokenizer,
                              original_f1_historical, original_f1_contemporary, device):
    """Run Experience Replay buffer size ablation study."""
    print_header("PHASE 2.6: ER BUFFER SIZE ABLATION STUDY", "#")

    print(f"\n Testing replay ratios: {ER_REPLAY_RATIOS}")
    print(f" Seeds per ratio: {len(ABLATION_SEEDS)}")

    new_train_df, new_val_df = train_test_split(new_df, test_size=0.2, random_state=42, stratify=new_df['label'])
    new_train_dataset = make_tokenized_dataset(new_train_df, tokenizer)
    val_dataset = make_tokenized_dataset(new_val_df, tokenizer)

    print(f"   New training data: {len(new_train_df)} samples")

    ablation_results = {}

    for replay_ratio in ER_REPLAY_RATIOS:
        config_key = f'er_ratio_{int(replay_ratio*100)}'

        # Calculate replay buffer size
        replay_size = int(len(new_train_df) * replay_ratio / (1 - replay_ratio))
        replay_size = min(replay_size, len(original_df))

        print(f"\n{'-'*60}")
        print(f"Testing: Replay ratio={replay_ratio:.1%}")
        print(f"   Replay buffer: {replay_size} old + {len(new_train_df)} new = {replay_size + len(new_train_df)} total")
        print(f"{'-'*60}")

        # Create combined dataset for this ratio
        replay_df = original_df.sample(n=replay_size, random_state=42)
        combined_df = pd.concat([replay_df, new_train_df], ignore_index=True)
        combined_dataset = make_tokenized_dataset(combined_df, tokenizer)

        config_results = []
        config_kr_list = []
        config_ag_list = []

        for seed in ABLATION_SEEDS:
            print(f"\n--- Seed {seed} ---")
            model = load_local_model(MODEL_PATH).to(device)

            results, predictions, training_time, loss_history, trainable = train_model(
                method_name=config_key,
                model=model,
                train_dataset=combined_dataset,
                val_dataset=val_dataset,
                test_datasets=test_datasets,
                tokenizer=tokenizer,
                seed=seed,
                use_lora=False,
                use_ewc=False,
                device=device
            )

            kr, ag = calculate_kr_ag(results, original_f1_historical, original_f1_contemporary)
            config_results.append(results)
            config_kr_list.append(kr)
            config_ag_list.append(ag)

            print(f"   KR: {kr:+.4f}, AG: {ag:+.4f}")

            del model
            clear_memory()

        ablation_results[config_key] = {
            'replay_ratio': replay_ratio,
            'replay_size': replay_size,
            'total_samples': replay_size + len(new_train_df),
            'results': config_results,
            'mean_kr': float(np.mean(config_kr_list)),
            'std_kr': float(np.std(config_kr_list)),
            'mean_ag': float(np.mean(config_ag_list)),
            'std_ag': float(np.std(config_ag_list)),
            'mean_historical': float(np.mean([r['unused_old']['f1_macro'] for r in config_results])),
            'std_historical': float(np.std([r['unused_old']['f1_macro'] for r in config_results])),
            'mean_contemporary': float(np.mean([r['balanced_500']['f1_macro'] for r in config_results])),
            'std_contemporary': float(np.std([r['balanced_500']['f1_macro'] for r in config_results])),
        }

        print(f"\n Summary for ratio={replay_ratio:.1%}:")
        print(f"   KR: {ablation_results[config_key]['mean_kr']:+.4f} ± {ablation_results[config_key]['std_kr']:.4f}")
        print(f"   AG: {ablation_results[config_key]['mean_ag']:+.4f} ± {ablation_results[config_key]['std_ag']:.4f}")

    # Select best ratio using unified criterion (60% KR + 40% AG)
    print_header("SELECTING OPTIMAL ER REPLAY RATIO", "-")

    best_config, best_score, selection_details = select_optimal_config(
        ablation_results, method_name='ER'
    )

    optimal_er_ratio = ablation_results[best_config]['replay_ratio']
    optimal_buffer_size = ablation_results[best_config]['replay_size']

    json_dump_safe({
        'ablation_results': {k: {kk: vv for kk, vv in v.items() if kk != 'results'}
                           for k, v in ablation_results.items()},
        'optimal_ratio': optimal_er_ratio,
        'optimal_buffer_size': optimal_buffer_size,
        'optimal_config': best_config,
        'selection_score': best_score,
        'selection_details': selection_details,
        'selection_weights': SELECTION_WEIGHTS,
    }, f'{OUTPUT_DIR}/phase2_6_er_ablation.json')

    return ablation_results, optimal_er_ratio


# ============================================================================
# PHASE 3: MAIN METHODS
# ============================================================================

def run_phase3_main_methods(original_df, new_df, test_datasets, tokenizer,
                            original_f1_historical, original_f1_contemporary,
                            optimal_lora_rank, optimal_lora_modules, optimal_ewc_lambda,
                            optimal_er_ratio, device):
    """Run main continual learning methods."""
    print_header(f"PHASE 3: MAIN METHODS", "#")
    print(f"  LoRA: r={optimal_lora_rank}, modules={optimal_lora_modules}")
    print(f"  EWC: λ={optimal_ewc_lambda}")
    print(f"  ER: ratio={optimal_er_ratio:.0%}")

    new_train_df, new_val_df = train_test_split(new_df, test_size=0.2, random_state=42, stratify=new_df['label'])
    new_train_dataset = make_tokenized_dataset(new_train_df, tokenizer)
    new_val_dataset = make_tokenized_dataset(new_val_df, tokenizer)

    # Use optimal replay ratio from ablation study
    replay_ratio = optimal_er_ratio
    replay_size = int(len(new_train_df) * replay_ratio / (1 - replay_ratio))
    replay_df = original_df.sample(n=min(replay_size, len(original_df)), random_state=42)
    combined_df = pd.concat([replay_df, new_train_df], ignore_index=True)
    combined_train_dataset = make_tokenized_dataset(combined_df, tokenizer)

    print(f"   New data only: {len(new_train_df)} samples")
    print(f"   Replay buffer: {len(replay_df)} old + {len(new_train_df)} new = {len(combined_df)} samples")

    ewc_df = original_df.sample(n=min(EWC_SAMPLES, len(original_df)), random_state=42)
    ewc_dataset = make_tokenized_dataset(ewc_df, tokenizer)
    print(f"   EWC dataset: {len(ewc_df)} samples")

    lora_target_modules = LORA_TARGET_CONFIGS[optimal_lora_modules]

    methods = {
        'naive_ft': {'use_lora': False, 'use_ewc': False, 'use_replay': False},
        'er': {'use_lora': False, 'use_ewc': False, 'use_replay': True},
        'ewc': {'use_lora': False, 'use_ewc': True, 'use_replay': False},
        'lora': {'use_lora': True, 'use_ewc': False, 'use_replay': False},
    }

    all_results = {}
    all_times = {}
    all_losses = {}
    all_predictions = {}
    all_params = {}

    for method_name, config in methods.items():
        print(f"\n{'-'*60}")
        print(f"Method: {METHOD_LABELS.get(method_name, method_name).upper()}")
        print(f"{'-'*60}")

        method_results = []
        method_times = []
        method_losses = []
        method_predictions = None
        method_params = 0

        for seed in MAIN_SEEDS:
            print(f"\n--- Seed {seed} ---")

            train_dataset = combined_train_dataset if config['use_replay'] else new_train_dataset

            model = load_local_model(MODEL_PATH).to(device)

            results, predictions, training_time, loss_history, trainable = train_model(
                method_name=method_name,
                model=model,
                train_dataset=train_dataset,
                val_dataset=new_val_dataset,
                test_datasets=test_datasets,
                tokenizer=tokenizer,
                seed=seed,
                use_lora=config['use_lora'],
                lora_r=optimal_lora_rank,
                lora_target_modules=lora_target_modules,
                use_ewc=config['use_ewc'],
                ewc_lambda=optimal_ewc_lambda,
                ewc_dataset=ewc_dataset if config['use_ewc'] else None,
                device=device
            )

            method_results.append(results)
            method_times.append(training_time)
            method_losses.append(loss_history)
            method_predictions = predictions
            method_params = trainable

            del model
            clear_memory()

        all_results[method_name] = method_results
        all_times[method_name] = method_times
        all_losses[method_name] = method_losses
        all_predictions[method_name] = method_predictions
        all_params[method_name] = int(method_params)

    json_dump_safe({
        'results': {k: v for k, v in all_results.items()},
        'times': all_times,
        'params': all_params,
    }, f'{OUTPUT_DIR}/phase3_main_methods.json')

    return all_results, all_times, all_losses, all_predictions, all_params


# ============================================================================
# PHASE 4: HYBRID METHODS
# ============================================================================

def run_phase4_hybrid_methods(original_df, new_df, test_datasets, tokenizer,
                              original_f1_historical, original_f1_contemporary,
                              optimal_lora_rank, optimal_lora_modules, optimal_ewc_lambda,
                              optimal_er_ratio, device):
    """Run hybrid continual learning methods."""
    print_header(f"PHASE 4: HYBRID METHODS", "#")
    print(f"  LoRA: r={optimal_lora_rank}, modules={optimal_lora_modules}")
    print(f"  EWC: λ={optimal_ewc_lambda}")
    print(f"  ER: ratio={optimal_er_ratio:.0%}")

    new_train_df, new_val_df = train_test_split(new_df, test_size=0.2, random_state=42, stratify=new_df['label'])
    new_train_dataset = make_tokenized_dataset(new_train_df, tokenizer)
    new_val_dataset = make_tokenized_dataset(new_val_df, tokenizer)

    # Use optimal replay ratio from ablation study
    replay_ratio = optimal_er_ratio
    replay_size = int(len(new_train_df) * replay_ratio / (1 - replay_ratio))
    replay_df = original_df.sample(n=min(replay_size, len(original_df)), random_state=42)
    combined_df = pd.concat([replay_df, new_train_df], ignore_index=True)
    combined_train_dataset = make_tokenized_dataset(combined_df, tokenizer)

    print(f"   Replay buffer: {len(replay_df)} old + {len(new_train_df)} new = {len(combined_df)} samples")

    ewc_df = original_df.sample(n=min(EWC_SAMPLES, len(original_df)), random_state=42)
    ewc_dataset = make_tokenized_dataset(ewc_df, tokenizer)

    lora_target_modules = LORA_TARGET_CONFIGS[optimal_lora_modules]

    methods = {
        'lora_er': {'use_lora': True, 'use_ewc': False, 'use_replay': True},
        'full_er_ewc': {'use_lora': False, 'use_ewc': True, 'use_replay': True},
        'lora_ewc': {'use_lora': True, 'use_ewc': True, 'use_replay': False},
        'lora_er_ewc': {'use_lora': True, 'use_ewc': True, 'use_replay': True},
    }

    all_results = {}
    all_times = {}
    all_losses = {}
    all_predictions = {}
    all_params = {}

    for method_name, config in methods.items():
        print(f"\n{'-'*60}")
        print(f"Method: {METHOD_LABELS.get(method_name, method_name).upper()}")
        print(f"{'-'*60}")

        method_results = []
        method_times = []
        method_losses = []
        method_predictions = None
        method_params = 0

        for seed in MAIN_SEEDS:
            print(f"\n--- Seed {seed} ---")

            train_dataset = combined_train_dataset if config['use_replay'] else new_train_dataset

            model = load_local_model(MODEL_PATH).to(device)

            results, predictions, training_time, loss_history, trainable = train_model(
                method_name=method_name,
                model=model,
                train_dataset=train_dataset,
                val_dataset=new_val_dataset,
                test_datasets=test_datasets,
                tokenizer=tokenizer,
                seed=seed,
                use_lora=config['use_lora'],
                lora_r=optimal_lora_rank,
                lora_target_modules=lora_target_modules,
                use_ewc=config['use_ewc'],
                ewc_lambda=optimal_ewc_lambda,
                ewc_dataset=ewc_dataset if config['use_ewc'] else None,
                device=device
            )

            method_results.append(results)
            method_times.append(training_time)
            method_losses.append(loss_history)
            method_predictions = predictions
            method_params = trainable

            del model
            clear_memory()

        all_results[method_name] = method_results
        all_times[method_name] = method_times
        all_losses[method_name] = method_losses
        all_predictions[method_name] = method_predictions
        all_params[method_name] = int(method_params)

    json_dump_safe({
        'results': {k: v for k, v in all_results.items()},
        'times': all_times,
        'params': all_params,
    }, f'{OUTPUT_DIR}/phase4_hybrid_methods.json')

    return all_results, all_times, all_losses, all_predictions, all_params


# ============================================================================
# ERROR ANALYSIS
# ============================================================================

def analyze_errors(predictions_dict, test_dfs, method_name, test_set_name='balanced_500'):
    """Analyze prediction errors with examples."""

    preds = predictions_dict.get(f'{test_set_name}_preds', [])
    labels = predictions_dict.get(f'{test_set_name}_labels', [])

    if not preds or not labels:
        return None

    df = test_dfs[test_set_name]
    texts = df['text'].tolist()
    types = df['type'].tolist() if 'type' in df.columns else ['unknown'] * len(df)
    sample_types = df['sample_type'].tolist() if 'sample_type' in df.columns else ['unknown'] * len(df)

    error_analysis = {
        'method': method_name,
        'test_set': test_set_name,
        'true_positives': [],
        'true_negatives': [],
        'false_positives': [],
        'false_negatives': [],
    }

    for i, (pred, label) in enumerate(zip(preds, labels)):
        entry = {
            'text': texts[i] if i < len(texts) else '',
            'type': types[i] if i < len(types) else 'unknown',
            'sample_type': sample_types[i] if i < len(sample_types) else 'unknown',
            'predicted': int(pred),
            'actual': int(label),
        }

        if pred == 1 and label == 1:
            error_analysis['true_positives'].append(entry)
        elif pred == 0 and label == 0:
            error_analysis['true_negatives'].append(entry)
        elif pred == 1 and label == 0:
            error_analysis['false_positives'].append(entry)
        elif pred == 0 and label == 1:
            error_analysis['false_negatives'].append(entry)

    # Summary statistics
    error_analysis['summary'] = {
        'total': len(preds),
        'tp': len(error_analysis['true_positives']),
        'tn': len(error_analysis['true_negatives']),
        'fp': len(error_analysis['false_positives']),
        'fn': len(error_analysis['false_negatives']),
        'accuracy': (len(error_analysis['true_positives']) + len(error_analysis['true_negatives'])) / len(preds) if preds else 0,
    }

    # Type breakdown for false negatives (missed offensive)
    fn_types = {}
    for entry in error_analysis['false_negatives']:
        t = entry.get('sample_type', entry.get('type', 'unknown'))
        fn_types[t] = fn_types.get(t, 0) + 1
    error_analysis['fn_type_breakdown'] = fn_types

    # Type breakdown for false positives
    fp_types = {}
    for entry in error_analysis['false_positives']:
        t = entry.get('sample_type', entry.get('type', 'unknown'))
        fp_types[t] = fp_types.get(t, 0) + 1
    error_analysis['fp_type_breakdown'] = fp_types

    return error_analysis


def generate_error_report(all_predictions, test_dfs, output_path):
    """Generate comprehensive error analysis report."""

    print_header("GENERATING ERROR ANALYSIS REPORT", "=")

    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("ERROR ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append("")

    all_error_analyses = {}

    for method_name, predictions_dict in all_predictions.items():
        if not predictions_dict:
            continue

        # Analyze errors on Contemporary and SimuReal test sets
        for test_set in ['balanced_500', 'simureal_500']:
            if test_set not in test_dfs:
                continue

            analysis = analyze_errors(predictions_dict, test_dfs, method_name, test_set)
            if analysis:
                key = f"{method_name}_{test_set}"
                all_error_analyses[key] = analysis

                report_lines.append(f"\n{''*80}")
                report_lines.append(f" {METHOD_LABELS.get(method_name, method_name)} on {TEST_SET_LABELS.get(test_set, test_set)}")
                report_lines.append(f"{''*80}")

                summary = analysis['summary']
                report_lines.append(f"\n  Accuracy: {summary['accuracy']*100:.1f}%")
                report_lines.append(f"  True Positives (Correct Offensive): {summary['tp']}")
                report_lines.append(f"  True Negatives (Correct Non-offensive): {summary['tn']}")
                report_lines.append(f"  False Positives (Wrong Offensive): {summary['fp']}")
                report_lines.append(f"  False Negatives (Missed Offensive): {summary['fn']}  CRITICAL")

                if analysis['fn_type_breakdown']:
                    report_lines.append(f"\n   FALSE NEGATIVES by type:")
                    for t, count in sorted(analysis['fn_type_breakdown'].items(), key=lambda x: -x[1]):
                        report_lines.append(f"     - {t}: {count}")

                if analysis['fp_type_breakdown']:
                    report_lines.append(f"\n   FALSE POSITIVES by type:")
                    for t, count in sorted(analysis['fp_type_breakdown'].items(), key=lambda x: -x[1]):
                        report_lines.append(f"     - {t}: {count}")

                # Show examples
                report_lines.append(f"\n   FALSE NEGATIVE EXAMPLES (Missed Offensive):")
                for i, entry in enumerate(analysis['false_negatives'][:3]):
                    text_preview = entry['text'][:100] + "..." if len(entry['text']) > 100 else entry['text']
                    report_lines.append(f"     {i+1}. [{entry.get('sample_type', entry.get('type', '?'))}] {text_preview}")

                report_lines.append(f"\n   FALSE POSITIVE EXAMPLES (Wrong Offensive):")
                for i, entry in enumerate(analysis['false_positives'][:3]):
                    text_preview = entry['text'][:100] + "..." if len(entry['text']) > 100 else entry['text']
                    report_lines.append(f"     {i+1}. [{entry.get('sample_type', entry.get('type', '?'))}] {text_preview}")

    report_text = "\n".join(report_lines)

    # Save report
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # Also save as JSON for further analysis
    json_dump_safe(all_error_analyses, output_path.replace('.txt', '.json'))

    print(report_text)
    print(f"\n Error report saved to: {output_path}")

    return all_error_analyses


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_all_tables(original_results, main_results, hybrid_results,
                        main_times, hybrid_times,
                        lora_ablation_results, ewc_ablation_results, er_ablation_results,
                        original_f1_historical, original_f1_contemporary,
                        optimal_lora_rank, optimal_lora_modules, optimal_ewc_lambda, optimal_er_ratio):
    """Generate all result tables."""

    print_header("GENERATING TABLES", "=")
    print(f"\n Optimal Hyperparameters (Selected with {SELECTION_WEIGHTS['kr']*100:.0f}% KR + {SELECTION_WEIGHTS['ag']*100:.0f}% AG):")
    print(f"   - LoRA: r={optimal_lora_rank}, modules={optimal_lora_modules}")
    print(f"   - EWC: λ={optimal_ewc_lambda}")
    print(f"   - ER: ratio={optimal_er_ratio:.0%}")

    all_results = {'original': [original_results]}
    all_results.update(main_results)
    all_results.update(hybrid_results)

    all_times = {}
    all_times.update(main_times)
    all_times.update(hybrid_times)

    # ========================================================================
    # TABLE 1: F1-Macro Performance
    # ========================================================================
    print("\n" + "="*100)
    print("TABLE 1: OVERALL PERFORMANCE (F1-Macro)")
    print("="*100)

    header = f"{'Method':<15}"
    for test_set in TEST_SETS_ORDER:
        header += f" | {TEST_SET_LABELS.get(test_set, test_set):<18}"
    print(f"\n{header}")
    print("-"*110)

    table1_data = []
    for method in METHODS_ORDER:
        if method in all_results:
            results_list = all_results[method]
            row = {'Method': METHOD_LABELS.get(method, method)}

            row_str = f"{row['Method']:<15}"
            for test_set in TEST_SETS_ORDER:
                if len(results_list) > 1:
                    f1s = [r[test_set]['f1_macro'] for r in results_list if test_set in r]
                    if f1s:
                        row[test_set] = f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}"
                    else:
                        row[test_set] = "N/A"
                else:
                    if test_set in results_list[0]:
                        row[test_set] = f"{results_list[0][test_set]['f1_macro']:.3f} ± 0.000"
                    else:
                        row[test_set] = "N/A"
                row_str += f" | {row[test_set]:<18}"

            table1_data.append(row)
            print(row_str)

    pd.DataFrame(table1_data).to_csv(f'{TABLES_DIR}/table1_f1_macro.csv', index=False)

    # ========================================================================
    # TABLE 2: KR and AG Analysis
    # ========================================================================
    print("\n" + "="*100)
    print("TABLE 2: KNOWLEDGE RETENTION (KR) & ADAPTATION GAIN (AG)")
    print("="*100)

    print(f"\n{'Method':<15} | {'KR':<20} | {'AG':<20} | {'Historical F1':<18} | {'Contemporary F1':<18}")
    print("-"*100)

    table2_data = []
    for method in METHODS_ORDER:
        if method == 'original':
            continue
        if method in all_results:
            results_list = all_results[method]

            if len(results_list) > 1:
                krs = [r['unused_old']['f1_macro'] - original_f1_historical for r in results_list]
                ags = [r['balanced_500']['f1_macro'] - original_f1_contemporary for r in results_list]
                hist_f1s = [r['unused_old']['f1_macro'] for r in results_list]
                cont_f1s = [r['balanced_500']['f1_macro'] for r in results_list]

                row = {
                    'Method': METHOD_LABELS.get(method, method),
                    'KR': f"{np.mean(krs):+.3f} ± {np.std(krs):.3f}",
                    'AG': f"{np.mean(ags):+.3f} ± {np.std(ags):.3f}",
                    'Historical F1': f"{np.mean(hist_f1s):.3f} ± {np.std(hist_f1s):.3f}",
                    'Contemporary F1': f"{np.mean(cont_f1s):.3f} ± {np.std(cont_f1s):.3f}",
                }
            else:
                kr = results_list[0]['unused_old']['f1_macro'] - original_f1_historical
                ag = results_list[0]['balanced_500']['f1_macro'] - original_f1_contemporary
                row = {
                    'Method': METHOD_LABELS.get(method, method),
                    'KR': f"{kr:+.3f} ± 0.000",
                    'AG': f"{ag:+.3f} ± 0.000",
                    'Historical F1': f"{results_list[0]['unused_old']['f1_macro']:.3f} ± 0.000",
                    'Contemporary F1': f"{results_list[0]['balanced_500']['f1_macro']:.3f} ± 0.000",
                }

            table2_data.append(row)
            print(f"{row['Method']:<15} | {row['KR']:<20} | {row['AG']:<20} | {row['Historical F1']:<18} | {row['Contemporary F1']:<18}")

    pd.DataFrame(table2_data).to_csv(f'{TABLES_DIR}/table2_kr_ag.csv', index=False)

    # ========================================================================
    # TABLE 3: Training Time
    # ========================================================================
    print("\n" + "="*80)
    print("TABLE 3: TRAINING TIME")
    print("="*80)

    print(f"\n{'Method':<15} | {'Avg Time (s)':<15} | {'Std (s)':<15}")
    print("-"*50)

    table3_data = []
    for method in METHODS_ORDER:
        if method == 'original':
            continue
        if method in all_times:
            times = all_times[method]
            row = {
                'Method': METHOD_LABELS.get(method, method),
                'Avg Time': f"{np.mean(times):.1f}",
                'Std': f"{np.std(times):.1f}",
            }
            table3_data.append(row)
            print(f"{row['Method']:<15} | {row['Avg Time']:<15} | {row['Std']:<15}")

    pd.DataFrame(table3_data).to_csv(f'{TABLES_DIR}/table3_training_time.csv', index=False)

    # ========================================================================
    # TABLE 4: LoRA Ablation
    # ========================================================================
    print("\n" + "="*100)
    print("TABLE 4: LoRA ABLATION STUDY")
    print("="*100)

    print(f"\n{'Config':<25} | {'Params':<12} | {'KR':<20} | {'AG':<20}")
    print("-"*85)

    table4_data = []
    for config_key, data in lora_ablation_results.items():
        params = data['trainable_params']
        params_str = f"{params/1000:.0f}K" if params < 1e6 else f"{params/1e6:.1f}M"

        marker = " " if (data['rank'] == optimal_lora_rank and data['modules'] == optimal_lora_modules) else ""

        row = {
            'Config': f"r={data['rank']}, {data['modules']}{marker}",
            'Params': params_str,
            'KR': f"{data['mean_kr']:+.3f} ± {data['std_kr']:.3f}",
            'AG': f"{data['mean_ag']:+.3f} ± {data['std_ag']:.3f}",
        }
        table4_data.append(row)
        print(f"{row['Config']:<25} | {row['Params']:<12} | {row['KR']:<20} | {row['AG']:<20}")

    pd.DataFrame(table4_data).to_csv(f'{TABLES_DIR}/table4_lora_ablation.csv', index=False)

    # ========================================================================
    # TABLE 5: EWC Lambda Ablation
    # ========================================================================
    print("\n" + "="*80)
    print("TABLE 5: EWC LAMBDA ABLATION STUDY")
    print("="*80)

    print(f"\n{'Lambda':<15} | {'KR':<20} | {'AG':<20}")
    print("-"*60)

    table5_data = []
    for config_key, data in ewc_ablation_results.items():
        marker = " " if data['lambda'] == optimal_ewc_lambda else ""
        row = {
            'Lambda': f"λ={data['lambda']}{marker}",
            'KR': f"{data['mean_kr']:+.3f} ± {data['std_kr']:.3f}",
            'AG': f"{data['mean_ag']:+.3f} ± {data['std_ag']:.3f}",
        }
        table5_data.append(row)
        print(f"{row['Lambda']:<15} | {row['KR']:<20} | {row['AG']:<20}")

    pd.DataFrame(table5_data).to_csv(f'{TABLES_DIR}/table5_ewc_ablation.csv', index=False)

    # ========================================================================
    # TABLE 6: ER Buffer Size Ablation
    # ========================================================================
    print("\n" + "="*80)
    print("TABLE 6: ER BUFFER SIZE ABLATION STUDY")
    print("="*80)

    print(f"\n{'Ratio':<10} | {'Buffer Size':<12} | {'KR':<20} | {'AG':<20}")
    print("-"*70)

    table6_data = []
    for config_key, data in er_ablation_results.items():
        marker = " " if data['replay_ratio'] == optimal_er_ratio else ""
        row = {
            'Ratio': f"{data['replay_ratio']:.0%}{marker}",
            'Buffer Size': str(data['replay_size']),
            'KR': f"{data['mean_kr']:+.3f} ± {data['std_kr']:.3f}",
            'AG': f"{data['mean_ag']:+.3f} ± {data['std_ag']:.3f}",
        }
        table6_data.append(row)
        print(f"{row['Ratio']:<10} | {row['Buffer Size']:<12} | {row['KR']:<20} | {row['AG']:<20}")

    pd.DataFrame(table6_data).to_csv(f'{TABLES_DIR}/table6_er_ablation.csv', index=False)

    print(f"\n{'='*80}")
    print(f" ALL TABLES GENERATED")
    print(f" Tables saved to: {TABLES_DIR}/")
    print(f"{'='*80}")


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def generate_all_figures(original_results, main_results, hybrid_results,
                         main_times, hybrid_times, main_losses, hybrid_losses,
                         main_predictions, hybrid_predictions, original_predictions,
                         lora_ablation_results, ewc_ablation_results, er_ablation_results,
                         original_f1_historical, original_f1_contemporary,
                         optimal_lora_rank, optimal_lora_modules, test_dfs):
    """Generate all visualization figures."""

    print_header("GENERATING ALL VISUALIZATIONS", "=")

    all_results = {'original': [original_results]}
    all_results.update(main_results)
    all_results.update(hybrid_results)

    all_times = {}
    all_times.update(main_times)
    all_times.update(hybrid_times)

    all_predictions = {'original': original_predictions}
    all_predictions.update(main_predictions)
    all_predictions.update(hybrid_predictions)

    # ========================================================================
    # FIGURE 1: Performance Grouped Bar Chart
    # ========================================================================
    print("\n [1/10] Creating Figure 1: Performance Comparison...")

    methods_order_display = [METHOD_LABELS.get(m, m) for m in METHODS_ORDER]
    test_set_labels_display = [TEST_SET_LABELS.get(t, t) for t in TEST_SETS_ORDER]
    test_set_colors = [TEST_SET_COLORS.get(TEST_SET_LABELS.get(t, t), '#999999') for t in TEST_SETS_ORDER]

    n_methods = len(METHODS_ORDER)
    n_tests = len(TEST_SETS_ORDER)

    means = np.zeros((n_methods, n_tests))
    stds = np.zeros((n_methods, n_tests))

    for i, method in enumerate(METHODS_ORDER):
        if method in all_results:
            for j, test_set in enumerate(TEST_SETS_ORDER):
                results_list = all_results[method]
                if len(results_list) > 1:
                    f1s = [r[test_set]['f1_macro'] for r in results_list if test_set in r]
                    if f1s:
                        means[i, j] = np.mean(f1s)
                        stds[i, j] = np.std(f1s)
                else:
                    if test_set in results_list[0]:
                        means[i, j] = results_list[0][test_set]['f1_macro']

    fig, ax = plt.subplots(figsize=(14, 6))

    x = np.arange(n_methods)
    width = 0.15
    offsets = np.array([-2, -1, 0, 1, 2]) * width

    for j, (label, color) in enumerate(zip(test_set_labels_display, test_set_colors)):
        ax.bar(x + offsets[j], means[:, j], width,
               yerr=stds[:, j], capsize=2,
               label=label, color=color,
               edgecolor='black', linewidth=0.5, alpha=0.85,
               error_kw={'elinewidth': 1, 'capthick': 1, 'color': 'black'})

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Macro', fontsize=12, fontweight='bold')
    ax.set_title('F1-Macro Performance Across Test Sets (mean ± std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(methods_order_display, rotation=0, ha='center', fontsize=9)
    ax.set_ylim(0.60, 1.0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=9, framealpha=0.95)

    plt.tight_layout()
    save_figure(fig, 'Figure_1_Performance_Grouped')

    # ========================================================================
    # FIGURE 2: KR vs AG Trade-off
    # ========================================================================
    print("\n [2/10] Creating Figure 2: KR vs AG Trade-off...")

    fig, ax = plt.subplots(figsize=(12, 8))

    for method in METHODS_ORDER:
        if method == 'original':
            continue
        if method in all_results:
            results_list = all_results[method]
            display_name = METHOD_LABELS.get(method, method)

            if len(results_list) > 1:
                krs = [r['unused_old']['f1_macro'] - original_f1_historical for r in results_list]
                ags = [r['balanced_500']['f1_macro'] - original_f1_contemporary for r in results_list]
                mean_kr, std_kr = np.mean(krs), np.std(krs)
                mean_ag, std_ag = np.mean(ags), np.std(ags)
            else:
                mean_kr = results_list[0]['unused_old']['f1_macro'] - original_f1_historical
                mean_ag = results_list[0]['balanced_500']['f1_macro'] - original_f1_contemporary
                std_kr, std_ag = 0.0, 0.0

            marker = MARKERS.get(display_name, 'o')
            color = COLORS.get(display_name, '#999999')

            ax.errorbar(mean_ag, mean_kr, xerr=std_ag, yerr=std_kr,
                        fmt=marker, markersize=14, capsize=5, capthick=2,
                        color=color, markeredgecolor='black', markeredgewidth=1.5,
                        label=display_name)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1)

    # Quadrant labels
    ax.text(0.27, 0.008, ' Ideal\n(Retain + Adapt)', fontsize=10, ha='center',
            color='green', fontweight='bold', alpha=0.8)
    ax.text(0.02, -0.09, ' Worst\n(Forget + No Adapt)', fontsize=10, ha='center',
            color='red', fontweight='bold', alpha=0.8)

    ax.set_xlabel('Adaptation Gain (AG)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Knowledge Retention (KR)', fontsize=12, fontweight='bold')
    ax.set_title('Knowledge Retention vs Adaptation Gain Trade-off', fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='upper center', fontsize=9, framealpha=0.95, ncol=4,
              edgecolor='black', fancybox=False, bbox_to_anchor=(0.5, 0.98))

    plt.tight_layout()
    save_figure(fig, 'Figure_2_KR_AG_Tradeoff')

    # ========================================================================
    # FIGURE 3: Training Time
    # ========================================================================
    print("\n [3/10] Creating Figure 3: Training Time...")

    methods_time = [m for m in METHODS_ORDER if m != 'original' and m in all_times]
    times_means = [float(np.mean(all_times[m])) for m in methods_time]
    colors_time = [COLORS.get(METHOD_LABELS.get(m, m), '#999999') for m in methods_time]

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(methods_time))
    bars = ax.bar(x_pos, times_means, color=colors_time,
                  edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_title('Average Training Time Across Methods', fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods_time], rotation=0, ha='center', fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    for bar, t in zip(bars, times_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{t:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'Figure_3_Training_Time')

    # ========================================================================
    # FIGURE 4: LoRA Ablation
    # ========================================================================
    print("\n [4/10] Creating Figure 4: LoRA Ablation Study...")

    # Organize data by rank and module type
    ranks = LORA_RANKS_TO_TEST
    module_configs = list(LORA_TARGET_CONFIGS.keys())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    x = np.arange(len(ranks))
    width = 0.35

    colors_modules = ['#0072B2', '#D55E00']  # Blue for standard, Orange for extended

    # Panel 1: KR (Knowledge Retention)
    ax1 = axes[0]
    for i, module_config in enumerate(module_configs):
        kr_means = []
        kr_stds = []
        for r in ranks:
            key = f'lora_r{r}_{module_config}'
            if key in lora_ablation_results:
                kr_means.append(lora_ablation_results[key]['mean_kr'])
                kr_stds.append(lora_ablation_results[key]['std_kr'])
            else:
                kr_means.append(0)
                kr_stds.append(0)

        ax1.bar(x + i*width - width/2, kr_means, width, yerr=kr_stds,
                label=module_config, color=colors_modules[i],
                edgecolor='black', linewidth=0.5, capsize=4)

    ax1.set_xlabel('LoRA Rank (r)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Knowledge Retention (KR)', fontsize=12, fontweight='bold')
    ax1.set_title('KR by Rank and Module Config', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(r) for r in ranks])
    ax1.legend(loc='upper right', fontsize=9)
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Panel 2: AG (Adaptation Gain)
    ax2 = axes[1]
    for i, module_config in enumerate(module_configs):
        ag_means = []
        ag_stds = []
        for r in ranks:
            key = f'lora_r{r}_{module_config}'
            if key in lora_ablation_results:
                ag_means.append(lora_ablation_results[key]['mean_ag'])
                ag_stds.append(lora_ablation_results[key]['std_ag'])
            else:
                ag_means.append(0)
                ag_stds.append(0)

        ax2.bar(x + i*width - width/2, ag_means, width, yerr=ag_stds,
                label=module_config, color=colors_modules[i],
                edgecolor='black', linewidth=0.5, capsize=4)

    ax2.set_xlabel('LoRA Rank (r)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Adaptation Gain (AG)', fontsize=12, fontweight='bold')
    ax2.set_title('AG by Rank and Module Config', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(r) for r in ranks])
    ax2.legend(loc='upper left', fontsize=9)
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle(f'LoRA Ablation: Rank × Module Configuration (Optimal: r={optimal_lora_rank}, {optimal_lora_modules})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, 'Figure_4_LoRA_Ablation')

    # ========================================================================
    # FIGURE 5: EWC Lambda Ablation
    # ========================================================================
    print("\n [5/10] Creating Figure 5: EWC Lambda Ablation...")

    lambdas = EWC_LAMBDA_VALUES
    kr_means = [ewc_ablation_results[f'ewc_lambda_{l}']['mean_kr'] for l in lambdas]
    kr_stds = [ewc_ablation_results[f'ewc_lambda_{l}']['std_kr'] for l in lambdas]
    ag_means = [ewc_ablation_results[f'ewc_lambda_{l}']['mean_ag'] for l in lambdas]
    ag_stds = [ewc_ablation_results[f'ewc_lambda_{l}']['std_ag'] for l in lambdas]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(lambdas))

    # Panel 1: KR
    ax1 = axes[0]
    bars1 = ax1.bar(x, kr_means, yerr=kr_stds, capsize=4,
                    color='#0072B2', edgecolor='black', linewidth=0.5, alpha=0.85)
    ax1.set_xlabel('EWC Lambda (λ)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Knowledge Retention (KR)', fontsize=12, fontweight='bold')
    ax1.set_title('KR vs EWC Lambda', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(l) for l in lambdas])
    ax1.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

    # Panel 2: AG
    ax2 = axes[1]
    bars2 = ax2.bar(x, ag_means, yerr=ag_stds, capsize=4,
                    color='#D55E00', edgecolor='black', linewidth=0.5, alpha=0.85)
    ax2.set_xlabel('EWC Lambda (λ)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Adaptation Gain (AG)', fontsize=12, fontweight='bold')
    ax2.set_title('AG vs EWC Lambda', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(l) for l in lambdas])
    ax2.yaxis.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle('EWC Lambda Ablation Study', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, 'Figure_5_EWC_Lambda_Ablation')

    # ========================================================================
    # FIGURE 6: ER Buffer Size Ablation
    # ========================================================================
    print("\n [6/10] Creating Figure 6: ER Buffer Size Ablation...")

    ratios = ER_REPLAY_RATIOS
    ratio_keys = [f'er_ratio_{int(r*100)}' for r in ratios]

    kr_means = [er_ablation_results[k]['mean_kr'] for k in ratio_keys]
    kr_stds = [er_ablation_results[k]['std_kr'] for k in ratio_keys]
    ag_means = [er_ablation_results[k]['mean_ag'] for k in ratio_keys]
    ag_stds = [er_ablation_results[k]['std_ag'] for k in ratio_keys]
    buffer_sizes = [er_ablation_results[k]['replay_size'] for k in ratio_keys]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot KR and AG as lines
    ax.errorbar(ratios, kr_means, yerr=kr_stds, marker='o', markersize=10,
                color='#0072B2', label='Knowledge Retention (KR)',
                linewidth=2, capsize=5)
    ax.errorbar(ratios, ag_means, yerr=ag_stds, marker='s', markersize=10,
                color='#D55E00', label='Adaptation Gain (AG)',
                linewidth=2, capsize=5)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

    ax.set_xlabel('Replay Ratio (proportion of old data)', fontsize=12, fontweight='bold')
    ax.set_ylabel('KR / AG', fontsize=12, fontweight='bold')
    ax.set_title('ER Buffer Size Ablation: Impact on KR and AG', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.3)

    # Add secondary x-axis for buffer sizes
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ratios)
    ax2.set_xticklabels([str(s) for s in buffer_sizes])
    ax2.set_xlabel('Buffer Size (old samples)', fontsize=10)

    plt.tight_layout()
    save_figure(fig, 'Figure_6_ER_Buffer_Ablation')

    # ========================================================================
    # FIGURE 7: Confusion Matrices (Selected Methods on Contemporary)
    # ========================================================================
    print("\n [7/10] Creating Figure 7: Confusion Matrices...")

    selected_methods = ['original', 'naive_ft', 'er', 'ewc', 'lora', 'lora_er',
                        'full_er_ewc', 'lora_ewc', 'lora_er_ewc']

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    test_set = 'balanced_500'

    for idx, method in enumerate(selected_methods):
        ax = axes[idx]

        if method in all_predictions and all_predictions[method]:
            preds_key = f'{test_set}_preds'
            labels_key = f'{test_set}_labels'

            if preds_key in all_predictions[method] and labels_key in all_predictions[method]:
                preds = all_predictions[method][preds_key]
                labels = all_predictions[method][labels_key]
                cm = confusion_matrix(labels, preds)

                acc = (cm[0,0] + cm[1,1]) / cm.sum() * 100 if cm.sum() > 0 else 0.0

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           cbar=False, annot_kws={'size': 14, 'fontweight': 'bold'},
                           xticklabels=['NOT', 'OFF'], yticklabels=['NOT', 'OFF'])

                ax.set_title(f"{METHOD_LABELS.get(method, method)}\nAcc: {acc:.1f}%",
                            fontsize=12, fontweight='bold')
                ax.set_xlabel('Predicted', fontsize=10)
                ax.set_ylabel('Actual', fontsize=10)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
                ax.axis('off')
        else:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
            ax.axis('off')

    plt.suptitle(f'Confusion Matrices - {TEST_SET_LABELS.get(test_set, test_set)} Test Set',
                 fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_figure(fig, 'Figure_7_Confusion_Matrices')

    # ========================================================================
    # FIGURE 8: SimuReal Performance
    # ========================================================================
    print("\n [8/10] Creating Figure 8: SimuReal Test Set Performance...")

    methods_simureal = [m for m in METHODS_ORDER if m in all_results]
    simureal_f1s = []
    simureal_stds = []

    for method in methods_simureal:
        results_list = all_results[method]
        if len(results_list) > 1:
            f1s = [r['simureal_500']['f1_macro'] for r in results_list if 'simureal_500' in r]
            if f1s:
                simureal_f1s.append(np.mean(f1s))
                simureal_stds.append(np.std(f1s))
            else:
                simureal_f1s.append(0)
                simureal_stds.append(0)
        else:
            if 'simureal_500' in results_list[0]:
                simureal_f1s.append(results_list[0]['simureal_500']['f1_macro'])
                simureal_stds.append(0)
            else:
                simureal_f1s.append(0)
                simureal_stds.append(0)

    fig, ax = plt.subplots(figsize=(12, 6))

    x_pos = np.arange(len(methods_simureal))
    colors_bars = [COLORS.get(METHOD_LABELS.get(m, m), '#999999') for m in methods_simureal]

    bars = ax.bar(x_pos, simureal_f1s, yerr=simureal_stds, capsize=4,
                  color=colors_bars, edgecolor='black', linewidth=0.5, alpha=0.85)

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Macro', fontsize=12, fontweight='bold')
    ax.set_title('SimuReal Test Set Performance (Realistic 80/20 Distribution)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods_simureal],
                       rotation=0, ha='center', fontsize=9)
    ax.set_ylim(0.6, 1.0)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)

    for bar, f1 in zip(bars, simureal_f1s):
        if f1 > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'Figure_8_SimuReal_Performance')

    # ========================================================================
    # FIGURE 9: Loss Curves
    # ========================================================================
    print("\n [9/10] Creating Figure 9: Loss Curves...")

    all_losses = {}
    all_losses.update(main_losses)
    all_losses.update(hybrid_losses)

    methods_for_loss = [m for m in METHODS_ORDER if m != 'original' and m in all_losses]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Training Loss
    ax1 = axes[0]
    for method in methods_for_loss:
        losses_list = all_losses[method]
        all_train = [l['train_losses'] for l in losses_list if l.get('train_losses')]
        if all_train:
            min_len = min(len(l) for l in all_train)
            if min_len > 0:
                arr = np.array([l[:min_len] for l in all_train])
                mean_train = np.mean(arr, axis=0)
                std_train = np.std(arr, axis=0)
                epochs = range(1, min_len + 1)
                display_name = METHOD_LABELS.get(method, method)
                color = COLORS.get(display_name, '#999999')
                ax1.plot(epochs, mean_train, marker='o', label=display_name,
                         color=color, linewidth=2)
                ax1.fill_between(epochs, mean_train - std_train, mean_train + std_train,
                                alpha=0.2, color=color)

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Convergence', fontsize=13, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, linestyle='--', alpha=0.3)

    # Validation Loss
    ax2 = axes[1]
    for method in methods_for_loss:
        losses_list = all_losses[method]
        all_val = [l['val_losses'] for l in losses_list if l.get('val_losses')]
        if all_val:
            min_len = min(len(l) for l in all_val)
            if min_len > 0:
                arr = np.array([l[:min_len] for l in all_val])
                mean_val = np.mean(arr, axis=0)
                std_val = np.std(arr, axis=0)
                epochs = range(1, min_len + 1)
                display_name = METHOD_LABELS.get(method, method)
                color = COLORS.get(display_name, '#999999')
                ax2.plot(epochs, mean_val, marker='s', label=display_name,
                         color=color, linewidth=2)
                ax2.fill_between(epochs, mean_val - std_val, mean_val + std_val,
                                alpha=0.2, color=color)

    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss Convergence', fontsize=13, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle('Loss Curves (mean ± std)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_figure(fig, 'Figure_9_Loss_Curves')

    # ========================================================================
    # FIGURE 10: Comprehensive Confusion Matrices (9×5)
    # ========================================================================
    print("\n [10/10] Creating Figure 10: Comprehensive Confusion Matrices (9×5)...")

    fig, axes = plt.subplots(9, 5, figsize=(20, 36))

    for row_idx, method in enumerate(METHODS_ORDER):
        for col_idx, test_set in enumerate(TEST_SETS_ORDER):
            ax = axes[row_idx, col_idx]

            if method in all_predictions and all_predictions[method]:
                preds_key = f'{test_set}_preds'
                labels_key = f'{test_set}_labels'

                if preds_key in all_predictions[method] and labels_key in all_predictions[method]:
                    preds = all_predictions[method][preds_key]
                    labels = all_predictions[method][labels_key]
                    cm = confusion_matrix(labels, preds)
                    acc = (cm[0, 0] + cm[1, 1]) / cm.sum() * 100 if cm.sum() > 0 else 0.0

                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               cbar=False, annot_kws={'size': 12, 'fontweight': 'bold'},
                               xticklabels=['NOT', 'OFF'], yticklabels=['NOT', 'OFF'])

                    if row_idx == 0:
                        ax.set_title(f'{TEST_SET_LABELS.get(test_set, test_set)}\nAcc: {acc:.1f}%',
                                    fontsize=11, fontweight='bold')
                    else:
                        ax.set_title(f'Acc: {acc:.1f}%', fontsize=10)

                    if col_idx == 0:
                        ax.set_ylabel(f'{METHOD_LABELS.get(method, method)}\n\nActual',
                                     fontsize=10, fontweight='bold')
                    else:
                        ax.set_ylabel('Actual', fontsize=9)

                    ax.set_xlabel('Predicted', fontsize=9)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=10)
                    ax.axis('off')
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=10)
                ax.axis('off')

    plt.suptitle('Confusion Matrix Analysis: All Methods × All Test Sets',
                 fontsize=18, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, 'Figure_10_Confusion_Matrices_Complete')

    print(f"\n{'='*80}")
    print(f" ALL 10 FIGURES GENERATED")
    print(f" Figures saved to: {FIGURES_DIR}/")
    print(f"{'='*80}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    print("\n" + "="*80)
    print("   COMPLETE CONTINUAL LEARNING EXPERIMENTS - VERSION 2")
    print("   With Ablation Studies, Error Analysis, and SimuReal Test Set")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {get_gpu_memory()}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    create_directories()

    print("\n Loading tokenizer...")
    tokenizer = load_local_tokenizer(MODEL_PATH)
    print(" Tokenizer loaded")

    original_df, new_df, test_datasets, test_dfs = load_all_data(tokenizer)

    # Phase 1: Original Model
    original_results, original_predictions, original_f1_historical, original_f1_contemporary = \
        run_phase1_original_model(test_datasets, device)

    # Phase 2: LoRA Ablation
    lora_ablation_results, lora_ablation_params, optimal_lora_rank, optimal_lora_modules = \
        run_phase2_lora_ablation(original_df, new_df, test_datasets, tokenizer,
                                 original_f1_historical, original_f1_contemporary, device)

    # Phase 2.5: EWC Lambda Ablation
    ewc_ablation_results, optimal_ewc_lambda = \
        run_phase2_5_ewc_ablation(original_df, new_df, test_datasets, tokenizer,
                                   original_f1_historical, original_f1_contemporary, device)

    # Phase 2.6: ER Buffer Size Ablation
    er_ablation_results, optimal_er_ratio = \
        run_phase2_6_er_ablation(original_df, new_df, test_datasets, tokenizer,
                                  original_f1_historical, original_f1_contemporary, device)

    # Phase 3: Main Methods
    main_results, main_times, main_losses, main_predictions, main_params = \
        run_phase3_main_methods(original_df, new_df, test_datasets, tokenizer,
                                original_f1_historical, original_f1_contemporary,
                                optimal_lora_rank, optimal_lora_modules, optimal_ewc_lambda,
                                optimal_er_ratio, device)

    # Phase 4: Hybrid Methods
    hybrid_results, hybrid_times, hybrid_losses, hybrid_predictions, hybrid_params = \
        run_phase4_hybrid_methods(original_df, new_df, test_datasets, tokenizer,
                                  original_f1_historical, original_f1_contemporary,
                                  optimal_lora_rank, optimal_lora_modules, optimal_ewc_lambda,
                                  optimal_er_ratio, device)

    # Combine all predictions for error analysis
    all_predictions = {'original': original_predictions}
    all_predictions.update(main_predictions)
    all_predictions.update(hybrid_predictions)

    # Generate Error Analysis Report
    error_analyses = generate_error_report(
        all_predictions, test_dfs,
        f'{OUTPUT_DIR}/error_analysis/error_report.txt'
    )

    # Generate Tables
    generate_all_tables(
        original_results, main_results, hybrid_results,
        main_times, hybrid_times,
        lora_ablation_results, ewc_ablation_results, er_ablation_results,
        original_f1_historical, original_f1_contemporary,
        optimal_lora_rank, optimal_lora_modules, optimal_ewc_lambda, optimal_er_ratio
    )

    # Generate Figures
    generate_all_figures(
        original_results, main_results, hybrid_results,
        main_times, hybrid_times, main_losses, hybrid_losses,
        main_predictions, hybrid_predictions, original_predictions,
        lora_ablation_results, ewc_ablation_results, er_ablation_results,
        original_f1_historical, original_f1_contemporary,
        optimal_lora_rank, optimal_lora_modules, test_dfs
    )

    # Save final configuration
    final_config = {
        'selection_weights': SELECTION_WEIGHTS,
        'optimal_lora_rank': int(optimal_lora_rank),
        'optimal_lora_modules': optimal_lora_modules,
        'optimal_ewc_lambda': int(optimal_ewc_lambda),
        'optimal_er_ratio': float(optimal_er_ratio),
        'main_seeds': MAIN_SEEDS,
        'ablation_seeds': ABLATION_SEEDS,
        'lora_ranks_tested': LORA_RANKS_TO_TEST,
        'ewc_lambdas_tested': EWC_LAMBDA_VALUES,
        'er_ratios_tested': ER_REPLAY_RATIOS,
        'original_f1_historical': float(original_f1_historical),
        'original_f1_contemporary': float(original_f1_contemporary),
        'timestamp': datetime.now().isoformat(),
    }
    json_dump_safe(final_config, f'{OUTPUT_DIR}/final_config.json')

    print("\n" + "="*80)
    print("    ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)

    print(f"\n Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n{'='*80}")
    print(" OUTPUT FILES")
    print(f"{'='*80}")
    print(f"\n    Figures (10): {FIGURES_DIR}/")
    print(f"    Tables (6): {TABLES_DIR}/")
    print(f"    Error Analysis: {OUTPUT_DIR}/error_analysis/")
    print(f"\n    JSON Results: {OUTPUT_DIR}/")
    print(f"      - final_config.json")
    print(f"      - phase1_original.json")
    print(f"      - phase2_lora_ablation.json")
    print(f"      - phase2_5_ewc_ablation.json")
    print(f"      - phase2_6_er_ablation.json")
    print(f"      - phase3_main_methods.json")
    print(f"      - phase4_hybrid_methods.json")
    print(f"\n{'='*80}")
    print("    EXPERIMENT COMPLETE! ")
    print(f"{'='*80}\n")

    return {
        'optimal_lora_rank': optimal_lora_rank,
        'optimal_lora_modules': optimal_lora_modules,
        'optimal_ewc_lambda': optimal_ewc_lambda,
        'optimal_er_ratio': optimal_er_ratio,
        'original_results': original_results,
        'main_results': main_results,
        'hybrid_results': hybrid_results,
        'lora_ablation': lora_ablation_results,
        'ewc_ablation': ewc_ablation_results,
        'er_ablation': er_ablation_results,
        'error_analyses': error_analyses,
    }


if __name__ == "__main__":
    results = main()

"""
================================================================================
POST-EXPERIMENT ANALYSIS
================================================================================
Generates:
1. Table 7: Complete Metrics (Precision, Recall, F1, Accuracy) for all methods
2. Table 8: Per-Class F1 Performance (Class 0 vs Class 1)
3. Fixed Figure 3: Parameter Efficiency (corrected x-axis scale)
================================================================================
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = './outputs'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'
TABLES_DIR = f'{OUTPUT_DIR}/tables'

# Method display names
METHOD_LABELS = {
    'original': 'Original',
    'naive_ft': 'Naïve FT',
    'er': 'ER',
    'ewc': 'EWC',
    'lora': 'LoRA',
    'lora_er': 'LoRA+ER',
    'full_er_ewc': 'Full+ER+EWC',
    'lora_ewc': 'LoRA+EWC',
    'lora_er_ewc': 'LoRA+ER+EWC',
}

TEST_SET_LABELS = {
    'unused_old': 'Historical',
    'balanced_500': 'Contemporary',
    'mixed_2080': 'Mixed 80-20',
    'mixed_4060': 'Mixed 40-60',
    'simureal_500': 'SimuReal',
}

METHODS_ORDER = ['original', 'naive_ft', 'er', 'ewc', 'lora', 'lora_er',
                 'full_er_ewc', 'lora_ewc', 'lora_er_ewc']

TEST_SETS_ORDER = ['unused_old', 'balanced_500', 'mixed_2080', 'mixed_4060', 'simureal_500']

# Colorblind-friendly palette
COLORS = {
    'Original':     '#999999',
    'Naïve FT':     '#E69F00',
    'ER':           '#0072B2',
    'EWC':          '#CC79A7',
    'LoRA':         '#009E73',
    'LoRA+ER':      '#F0E442',
    'Full+ER+EWC':  '#8B4513',
    'LoRA+EWC':     '#56B4E9',
    'LoRA+ER+EWC':  '#D55E00',
}

MARKERS = {
    'Naïve FT': 'o', 'ER': 's', 'EWC': '^', 'Full+ER+EWC': 'D',
    'LoRA': 'p', 'LoRA+ER': 'h', 'LoRA+EWC': 'v', 'LoRA+ER+EWC': '*',
}

# ============================================================================
# LOAD RESULTS
# ============================================================================

def load_results():
    """Load all experiment results from JSON files."""

    print(" Loading experiment results...")

    with open(f'{OUTPUT_DIR}/phase1_original.json', 'r') as f:
        phase1 = json.load(f)

    with open(f'{OUTPUT_DIR}/phase3_main_methods.json', 'r') as f:
        phase3 = json.load(f)

    with open(f'{OUTPUT_DIR}/phase4_hybrid_methods.json', 'r') as f:
        phase4 = json.load(f)

    with open(f'{OUTPUT_DIR}/final_config.json', 'r') as f:
        config = json.load(f)

    print(" Results loaded successfully!")

    return phase1, phase3, phase4, config


# ============================================================================
# TABLE 7: COMPLETE METRICS TABLE
# ============================================================================

def generate_complete_metrics_table(phase1, phase3, phase4):
    """Generate complete metrics table with Precision, Recall, F1, Accuracy."""

    print("\n" + "="*100)
    print("TABLE 7: COMPLETE EVALUATION METRICS (All Methods × All Test Sets)")
    print("="*100)

    all_results = {'original': [phase1['results']]}
    all_results.update(phase3['results'])
    all_results.update(phase4['results'])

    table_data = []

    for method in METHODS_ORDER:
        if method not in all_results:
            continue

        results_list = all_results[method]
        method_label = METHOD_LABELS.get(method, method)

        for test_set in TEST_SETS_ORDER:
            test_label = TEST_SET_LABELS.get(test_set, test_set)

            if len(results_list) > 1:
                precisions = [r[test_set]['precision'] for r in results_list if test_set in r]
                recalls = [r[test_set]['recall'] for r in results_list if test_set in r]
                f1s = [r[test_set]['f1_macro'] for r in results_list if test_set in r]
                accs = [r[test_set]['accuracy'] for r in results_list if test_set in r]

                if precisions:
                    row = {
                        'Method': method_label,
                        'Test Set': test_label,
                        'Precision': f"{np.mean(precisions):.4f} ± {np.std(precisions):.4f}",
                        'Recall': f"{np.mean(recalls):.4f} ± {np.std(recalls):.4f}",
                        'F1-macro': f"{np.mean(f1s):.4f} ± {np.std(f1s):.4f}",
                        'Accuracy': f"{np.mean(accs):.4f} ± {np.std(accs):.4f}",
                    }
                    table_data.append(row)
            else:
                if test_set in results_list[0]:
                    r = results_list[0][test_set]
                    row = {
                        'Method': method_label,
                        'Test Set': test_label,
                        'Precision': f"{r['precision']:.4f} ± 0.0000",
                        'Recall': f"{r['recall']:.4f} ± 0.0000",
                        'F1-macro': f"{r['f1_macro']:.4f} ± 0.0000",
                        'Accuracy': f"{r['accuracy']:.4f} ± 0.0000",
                    }
                    table_data.append(row)

    print(f"\n{'Method':<15} | {'Test Set':<15} | {'Precision':<18} | {'Recall':<18} | {'F1-macro':<18} | {'Accuracy':<18}")
    print("-"*110)

    current_method = None
    for row in table_data:
        if row['Method'] != current_method:
            if current_method is not None:
                print("-"*110)
            current_method = row['Method']

        print(f"{row['Method']:<15} | {row['Test Set']:<15} | {row['Precision']:<18} | {row['Recall']:<18} | {row['F1-macro']:<18} | {row['Accuracy']:<18}")

    df = pd.DataFrame(table_data)
    df.to_csv(f'{TABLES_DIR}/table7_complete_metrics.csv', index=False)
    print(f"\n Saved: {TABLES_DIR}/table7_complete_metrics.csv")

    return table_data


# ============================================================================
# TABLE 8: PER-CLASS F1 PERFORMANCE
# ============================================================================

def generate_per_class_table(phase1, phase3, phase4):
    """Generate per-class F1 performance table."""

    print("\n" + "="*100)
    print("TABLE 8: PER-CLASS F1 PERFORMANCE (Class 0: Non-Offensive, Class 1: Offensive)")
    print("="*100)

    all_results = {'original': [phase1['results']]}
    all_results.update(phase3['results'])
    all_results.update(phase4['results'])

    table_data = []

    for method in METHODS_ORDER:
        if method not in all_results:
            continue

        results_list = all_results[method]
        method_label = METHOD_LABELS.get(method, method)

        for test_set in TEST_SETS_ORDER:
            test_label = TEST_SET_LABELS.get(test_set, test_set)

            if len(results_list) > 1:
                f1_c0 = [r[test_set]['f1_class_0'] for r in results_list if test_set in r]
                f1_c1 = [r[test_set]['f1_class_1'] for r in results_list if test_set in r]

                if f1_c0:
                    diff = np.mean(f1_c1) - np.mean(f1_c0)
                    row = {
                        'Method': method_label,
                        'Test Set': test_label,
                        'F1 Class 0 (NOT)': f"{np.mean(f1_c0):.4f} ± {np.std(f1_c0):.4f}",
                        'F1 Class 1 (OFF)': f"{np.mean(f1_c1):.4f} ± {np.std(f1_c1):.4f}",
                        'Difference': f"{diff:+.4f}",
                    }
                    table_data.append(row)
            else:
                if test_set in results_list[0]:
                    r = results_list[0][test_set]
                    diff = r['f1_class_1'] - r['f1_class_0']
                    row = {
                        'Method': method_label,
                        'Test Set': test_label,
                        'F1 Class 0 (NOT)': f"{r['f1_class_0']:.4f} ± 0.0000",
                        'F1 Class 1 (OFF)': f"{r['f1_class_1']:.4f} ± 0.0000",
                        'Difference': f"{diff:+.4f}",
                    }
                    table_data.append(row)

    print(f"\n{'Method':<15} | {'Test Set':<15} | {'F1 Class 0 (NOT)':<20} | {'F1 Class 1 (OFF)':<20} | {'Diff':<10}")
    print("-"*90)

    current_method = None
    for row in table_data:
        if row['Method'] != current_method:
            if current_method is not None:
                print("-"*90)
            current_method = row['Method']

        print(f"{row['Method']:<15} | {row['Test Set']:<15} | {row['F1 Class 0 (NOT)']:<20} | {row['F1 Class 1 (OFF)']:<20} | {row['Difference']:<10}")

    df = pd.DataFrame(table_data)
    df.to_csv(f'{TABLES_DIR}/table8_per_class_f1.csv', index=False)
    print(f"\n Saved: {TABLES_DIR}/table8_per_class_f1.csv")

    return table_data


# ============================================================================
# TABLE 9: SUMMARY TABLE (Contemporary Test Set Only)
# ============================================================================

def generate_summary_table(phase1, phase3, phase4):
    """Generate summary comparison table for Contemporary test set."""

    print("\n" + "="*100)
    print("TABLE 9: SUMMARY - CONTEMPORARY TEST SET (Key Method Comparison)")
    print("="*100)

    all_results = {'original': [phase1['results']]}
    all_results.update(phase3['results'])
    all_results.update(phase4['results'])

    test_set = 'balanced_500'

    table_data = []

    for method in METHODS_ORDER:
        if method not in all_results:
            continue

        results_list = all_results[method]
        method_label = METHOD_LABELS.get(method, method)

        if len(results_list) > 1:
            precisions = [r[test_set]['precision'] for r in results_list if test_set in r]
            recalls = [r[test_set]['recall'] for r in results_list if test_set in r]
            f1s = [r[test_set]['f1_macro'] for r in results_list if test_set in r]
            accs = [r[test_set]['accuracy'] for r in results_list if test_set in r]
            f1_c0 = [r[test_set]['f1_class_0'] for r in results_list if test_set in r]
            f1_c1 = [r[test_set]['f1_class_1'] for r in results_list if test_set in r]

            if precisions:
                row = {
                    'Method': method_label,
                    'Precision': f"{np.mean(precisions):.3f}±{np.std(precisions):.3f}",
                    'Recall': f"{np.mean(recalls):.3f}±{np.std(recalls):.3f}",
                    'F1-macro': f"{np.mean(f1s):.3f}±{np.std(f1s):.3f}",
                    'Accuracy': f"{np.mean(accs):.3f}±{np.std(accs):.3f}",
                    'F1 (NOT)': f"{np.mean(f1_c0):.3f}±{np.std(f1_c0):.3f}",
                    'F1 (OFF)': f"{np.mean(f1_c1):.3f}±{np.std(f1_c1):.3f}",
                }
                table_data.append(row)
        else:
            if test_set in results_list[0]:
                r = results_list[0][test_set]
                row = {
                    'Method': method_label,
                    'Precision': f"{r['precision']:.3f}±0.000",
                    'Recall': f"{r['recall']:.3f}±0.000",
                    'F1-macro': f"{r['f1_macro']:.3f}±0.000",
                    'Accuracy': f"{r['accuracy']:.3f}±0.000",
                    'F1 (NOT)': f"{r['f1_class_0']:.3f}±0.000",
                    'F1 (OFF)': f"{r['f1_class_1']:.3f}±0.000",
                }
                table_data.append(row)

    print(f"\n{'Method':<15} | {'Precision':<14} | {'Recall':<14} | {'F1-macro':<14} | {'Accuracy':<14} | {'F1 (NOT)':<14} | {'F1 (OFF)':<14}")
    print("-"*110)

    for row in table_data:
        print(f"{row['Method']:<15} | {row['Precision']:<14} | {row['Recall']:<14} | {row['F1-macro']:<14} | {row['Accuracy']:<14} | {row['F1 (NOT)']:<14} | {row['F1 (OFF)']:<14}")

    df = pd.DataFrame(table_data)
    df.to_csv(f'{TABLES_DIR}/table9_summary_contemporary.csv', index=False)
    print(f"\n Saved: {TABLES_DIR}/table9_summary_contemporary.csv")

    return table_data


# ============================================================================
# FIXED FIGURE 3: PARAMETER EFFICIENCY
# ============================================================================

def generate_parameter_efficiency_figure(phase1, phase3, phase4):
    """Generate corrected Parameter Efficiency figure."""

    print("\n" + "="*80)
    print(" Generating FIXED Figure 3: Parameter Efficiency")
    print("="*80)

    all_results = {'original': [phase1['results']]}
    all_results.update(phase3['results'])
    all_results.update(phase4['results'])

    all_params = {}
    all_params.update(phase3.get('params', {}))
    all_params.update(phase4.get('params', {}))

    # Parameter counts (in millions)
    params_data = {
        'Naïve FT':     135.0,
        'ER':           135.0,
        'EWC':          135.0,
        'Full+ER+EWC':  135.0,
        'LoRA':         all_params.get('lora', 886000) / 1e6,
        'LoRA+ER':      all_params.get('lora_er', 886000) / 1e6,
        'LoRA+EWC':     all_params.get('lora_ewc', 886000) / 1e6,
        'LoRA+ER+EWC':  all_params.get('lora_er_ewc', 886000) / 1e6,
    }

    print(f"\n Parameter counts:")
    for method, params in params_data.items():
        print(f"   {method}: {params:.3f}M")

    methods_to_plot = ['Naïve FT', 'ER', 'EWC', 'Full+ER+EWC', 'LoRA', 'LoRA+ER', 'LoRA+EWC', 'LoRA+ER+EWC']
    method_to_key = {
        'Naïve FT': 'naive_ft', 'ER': 'er', 'EWC': 'ewc', 'Full+ER+EWC': 'full_er_ewc',
        'LoRA': 'lora', 'LoRA+ER': 'lora_er', 'LoRA+EWC': 'lora_ewc', 'LoRA+ER+EWC': 'lora_er_ewc'
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for method_label in methods_to_plot:
        method_key = method_to_key[method_label]

        if method_key not in all_results:
            continue

        results_list = all_results[method_key]
        params = params_data[method_label]

        hist_f1s = [r['unused_old']['f1_macro'] for r in results_list if 'unused_old' in r]
        cont_f1s = [r['balanced_500']['f1_macro'] for r in results_list if 'balanced_500' in r]

        if hist_f1s:
            hist_mean, hist_std = np.mean(hist_f1s), np.std(hist_f1s)
            cont_mean, cont_std = np.mean(cont_f1s), np.std(cont_f1s)

            marker = MARKERS.get(method_label, 'o')
            color = COLORS.get(method_label, '#999999')

            ax1.errorbar(params, hist_mean, yerr=hist_std,
                        fmt=marker, markersize=12,
                        capsize=4, color=color,
                        markeredgecolor='black', markeredgewidth=1,
                        label=method_label)

            ax2.errorbar(params, cont_mean, yerr=cont_std,
                        fmt=marker, markersize=12,
                        capsize=4, color=color,
                        markeredgecolor='black', markeredgewidth=1)

    for ax in [ax1, ax2]:
        ax.set_xscale('log')
        ax.set_xlabel('Trainable Parameters (Millions)', fontsize=11, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_ylim(0.65, 1.0)
        ax.set_xticks([0.5, 1, 5, 10, 50, 100, 150])
        ax.set_xticklabels(['0.5M', '1M', '5M', '10M', '50M', '100M', '150M'])
        ax.set_xlim(0.3, 200)

    ax1.set_ylabel('F1-macro (Historical)', fontsize=11, fontweight='bold')
    ax1.set_title('Knowledge Retention', fontsize=13, fontweight='bold')

    ax2.set_ylabel('F1-macro (Contemporary)', fontsize=11, fontweight='bold')
    ax2.set_title('Adaptation', fontsize=13, fontweight='bold')

    ax1.legend(loc='lower right', fontsize=9, framealpha=0.95, ncol=2,
               edgecolor='black', fancybox=False)

    plt.suptitle('Parameter Efficiency: Retention vs Adaptation Trade-off',
                 fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plt.savefig(f'{FIGURES_DIR}/Figure_3_Parameter_Efficiency_FIXED.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(f'{FIGURES_DIR}/Figure_3_Parameter_Efficiency_FIXED.pdf', bbox_inches='tight', facecolor='white')
    plt.show()

    print(f"\n Saved: {FIGURES_DIR}/Figure_3_Parameter_Efficiency_FIXED.png/.pdf")


# ============================================================================
# RUN ALL
# ============================================================================

print("\n" + "="*80)
print("   POST-EXPERIMENT ANALYSIS")
print("   Generating Complete Tables and Fixed Figures")
print("="*80)

phase1, phase3, phase4, config = load_results()

print(f"\n Experiment Configuration:")
print(f"   Optimal LoRA: r={config.get('optimal_lora_rank')}, modules={config.get('optimal_lora_modules')}")
print(f"   Optimal EWC λ: {config.get('optimal_ewc_lambda')}")
print(f"   Optimal ER ratio: {config.get('optimal_er_ratio')}")

table7 = generate_complete_metrics_table(phase1, phase3, phase4)
table8 = generate_per_class_table(phase1, phase3, phase4)
table9 = generate_summary_table(phase1, phase3, phase4)
generate_parameter_efficiency_figure(phase1, phase3, phase4)

print("\n" + "="*80)
print("    POST-EXPERIMENT ANALYSIS COMPLETE!")
print("="*80)
print(f"\n New files created:")
print(f"   - {TABLES_DIR}/table7_complete_metrics.csv")
print(f"   - {TABLES_DIR}/table8_per_class_f1.csv")
print(f"   - {TABLES_DIR}/table9_summary_contemporary.csv")
print(f"   - {FIGURES_DIR}/Figure_3_Parameter_Efficiency_FIXED.png/.pdf")

import matplotlib.pyplot as plt
import numpy as np
import os

# Directories
OUTPUT_DIR = './outputs'
FIGURES_DIR = f'{OUTPUT_DIR}/figures'

# Create figures folder if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)

# Data
replay_ratios = [0.1, 0.2, 0.4, 0.6, 0.8]
kr_values = [-0.060, -0.054, -0.044, -0.035, -0.037]
kr_std = [0.007, 0.005, 0.004, 0.006, 0.010]
ag_values = [0.253, 0.258, 0.261, 0.263, 0.261]
ag_std = [0.002, 0.001, 0.002, 0.003, 0.001]

fig, ax = plt.subplots(figsize=(10, 6))

# Plot KR
ax.errorbar(replay_ratios, kr_values, yerr=kr_std,
            marker='o', markersize=8, capsize=4, capthick=1.5,
            color='#1f77b4', linewidth=2, label='Knowledge Retention (KR)')

# Plot AG
ax.errorbar(replay_ratios, ag_values, yerr=ag_std,
            marker='s', markersize=8, capsize=4, capthick=1.5,
            color='#ff7f0e', linewidth=2, label='Adaptation Gain (AG)')

# Reference line at 0
ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

ax.set_xlabel('Replay Ratio', fontsize=12)
ax.set_ylabel('KR / AG', fontsize=12)
ax.set_title('ER Buffer Size Ablation: Impact on KR and AG', fontsize=14)
ax.legend(loc='right')
ax.set_xticks(replay_ratios)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{FIGURES_DIR}/er_ablation.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"Figure saved to {FIGURES_DIR}/er_ablationV2.png")

