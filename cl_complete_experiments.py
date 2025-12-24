"""
================================================================================
COMPLETE CONTINUAL LEARNING EXPERIMENTS - LoRA ABLATION FIRST (FULL VERSION)
================================================================================
This version runs LoRA ablation FIRST to determine the optimal rank,
then uses that optimal rank for all subsequent experiments.

INCLUDES ALL:
✓ All 6 Tables (F1, Accuracy, BWT/FWT, Per-class, Time, Ablation)
✓ All 8 Figures (Bar charts, scatter plots, confusion matrices, etc.)
✓ Complete results saving
✓ Predictions for confusion matrices

NEW EXPERIMENT ORDER:
├── Phase 1: Original Model (Baseline) - 1 run
├── Phase 2: LoRA Ablation (5 seeds each) ← MOVED HERE FIRST
│   ├── r=8
│   ├── r=16
│   ├── r=32
│   └── r=64
│   → Calculate BWT/FWT → Select BEST RANK
├── Phase 3: Main Methods (5 seeds each, using optimal LoRA rank)
│   ├── Naive Fine-tuning
│   ├── Experience Replay (ER)
│   ├── Elastic Weight Consolidation (EWC)
│   └── LoRA (with optimal rank)
└── Phase 4: Hybrid Methods (5 seeds each, using optimal LoRA rank)
    ├── LoRA + ER
    ├── Full + ER + EWC
    ├── LoRA + EWC
    └── LoRA + ER + EWC

================================================================================
"""

# ============================================================================
# ============================================================================
import os
from pathlib import Path

# =============================================================================
# Path configuration (GitHub-friendly)
# -----------------------------------------------------------------------------
# This script was originally developed in Google Colab using Google Drive paths.
# For a clean GitHub release, all paths are now relative to the repository root:
#   - Put datasets under:   ./data/
#   - Put base model under: ./models/
#   - Outputs go to:        ./results/
#
# If you prefer running in Colab, you can still mount Drive manually, then point
# DATA_DIR / MODELS_DIR / RESULTS_DIR to your Drive folders.
# =============================================================================

BASE_DIR   = Path(__file__).resolve().parent
DATA_DIR   = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Ensure output folders exist
(RESULTS_DIR / "figures").mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / "tables").mkdir(parents=True, exist_ok=True)
# Disable wandb to avoid prompts
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
from torch.utils.data import DataLoader

# ============================================================================
# CONFIGURATION
# ============================================================================

# Paths - UPDATE THESE TO YOUR PATHS!
MODEL_PATH = str(MODELS_DIR / 'SOD_AraBERT_model')
ORIGINAL_DATA_PATH = str(DATA_DIR / 'SDOffensive_Paper2.csv')
NEW_DATA_PATH = str(DATA_DIR / 'Paper2_DS_Complete.csv')

TEST_SET_PATHS = {
    'unused_old': str(DATA_DIR / 'processed500UnseenDS_Paper2.csv'),
    'balanced_500': str(DATA_DIR / 'Balanced500_Paper2.csv'),
    'mixed_2080': str(DATA_DIR / 'TestDS2080_Paper2.csv'),
    'mixed_4060': str(DATA_DIR / 'TestDS4060_Paper2.csv')
}

OUTPUT_DIR = str(RESULTS_DIR / 'LoRA_First_CL_Results')
FIGURES_DIR = f'{OUTPUT_DIR}/figures'
TABLES_DIR = f'{OUTPUT_DIR}/tables'

# Experiment settings
SEEDS = [42, 123, 456, 789, 101]
ABLATION_SEEDS = [42, 123, 456, 789, 101]  # Use all 5 seeds for ablation

# LoRA ranks to test in ablation
LORA_RANKS_TO_TEST = [8, 16, 32, 64]

# Training settings
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 2e-5
REPLAY_RATIO = 0.8
EWC_LAMBDA = 1000
EWC_SAMPLES = 1000

# Selection criterion for best LoRA rank
# Options: 'bwt', 'fwt', 'balanced', 'historical', 'contemporary'
LORA_SELECTION_CRITERION = 'balanced'

# Colorblind-friendly palette
COLORBLIND_PALETTE = [
    '#fc8d62',  # Orange
    '#8da0cb',  # Blue-purple
    '#e78ac3',  # Pink
    '#a6d854',  # Light green
    '#ffd92f',  # Yellow
    '#e5c494',  # Tan
    '#b3b3b3',  # Gray
    '#1b9e77',  # Dark teal
    '#66c2a5',  # Teal
]

METHOD_LABELS_SHORT = {
    'original': 'Original',
    'naive_ft': 'Naive FT',
    'er': 'ER',
    'ewc': 'EWC',
    'lora': 'LoRA',
    'lora_er': 'LoRA+ER',
    'full_er_ewc': 'Full+ER+EWC',
    'lora_ewc': 'LoRA+EWC',
    'lora_er_ewc': 'LoRA+ER+EWC',
}

def print_phase_summary(phase_name, results_dict, original_f1_historical=None, original_f1_contemporary=None):
    """Print detailed summary of results for a phase"""
    print(f"\n{'='*80}")
    print(f"📊 {phase_name} - DETAILED RESULTS SUMMARY")
    print(f"{'='*80}")

    for method, results_list in results_dict.items():
        print(f"\n📌 {METHOD_LABELS_SHORT.get(method, method).upper()}")
        print(f"   {'Test Set':<15} | {'F1-macro':<18} | {'Accuracy':<18}")
        print(f"   {'-'*55}")

        for test_set in ['unused_old', 'balanced_500', 'mixed_2080', 'mixed_4060']:
            if len(results_list) > 1:
                f1s = [r[test_set]['f1_macro'] for r in results_list]
                accs = [r[test_set]['accuracy'] for r in results_list]
                print(f"   {TEST_SET_LABELS_SHORT.get(test_set, test_set):<15} | {np.mean(f1s):.4f} ± {np.std(f1s):.4f} | {np.mean(accs):.4f} ± {np.std(accs):.4f}")
            else:
                print(f"   {TEST_SET_LABELS_SHORT.get(test_set, test_set):<15} | {results_list[0][test_set]['f1_macro']:.4f}          | {results_list[0][test_set]['accuracy']:.4f}")

        # Calculate BWT/FWT if baseline provided
        if original_f1_historical is not None and original_f1_contemporary is not None:
            if len(results_list) > 1:
                bwts = [r['unused_old']['f1_macro'] - original_f1_historical for r in results_list]
                fwts = [r['balanced_500']['f1_macro'] - original_f1_contemporary for r in results_list]
                print(f"\n   📈 BWT: {np.mean(bwts):+.4f} ± {np.std(bwts):.4f}")
                print(f"   📈 FWT: {np.mean(fwts):+.4f} ± {np.std(fwts):.4f}")
            else:
                bwt = results_list[0]['unused_old']['f1_macro'] - original_f1_historical
                fwt = results_list[0]['balanced_500']['f1_macro'] - original_f1_contemporary
                print(f"\n   📈 BWT: {bwt:+.4f}")
                print(f"   📈 FWT: {fwt:+.4f}")

TEST_SET_LABELS = {
    'unused_old': 'Historical (100% Old)',
    'balanced_500': 'Contemporary (100% New)',
    'mixed_2080': 'Mixed (80% Old, 20% New)',
    'mixed_4060': 'Mixed (40% Old, 60% New)',
}

TEST_SET_LABELS_SHORT = {
    'unused_old': 'Historical',
    'balanced_500': 'Contemporary',
    'mixed_2080': 'Mixed 80-20',
    'mixed_4060': 'Mixed 40-60',
}

METHODS_ORDER = ['original', 'naive_ft', 'er', 'ewc', 'lora', 'lora_er', 'full_er_ewc', 'lora_ewc', 'lora_er_ewc']

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def print_header(text, char="="):
    """Print formatted header"""
    print(f"\n{char*80}")
    print(f"{text}")
    print(f"{char*80}")

def get_gpu_memory():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        return f"{allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "No GPU"

def clear_memory():
    """Clear GPU memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("🧹 Memory cleared")

def set_seed(seed):
    """Set random seeds for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def create_directories():
    """Create output directories"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"📁 Figures: {FIGURES_DIR}")
    print(f"📁 Tables: {TABLES_DIR}")

def load_local_tokenizer(path):
    """Load tokenizer with local_files_only"""
    try:
        return AutoTokenizer.from_pretrained(path, local_files_only=True)
    except:
        return AutoTokenizer.from_pretrained(path, local_files_only=True, trust_remote_code=True)

def load_local_model(path, num_labels=2):
    """Load model with local_files_only"""
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=num_labels, local_files_only=True
        )
    except:
        return AutoModelForSequenceClassification.from_pretrained(
            path, num_labels=num_labels, local_files_only=True, trust_remote_code=True
        )

def save_figure(fig, name, show=True):
    """Save figure in both PNG and PDF formats, display it, then close"""
    png_path = f'{FIGURES_DIR}/{name}.png'
    pdf_path = f'{FIGURES_DIR}/{name}.pdf'
    fig.savefig(png_path, dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"   ✓ Saved: {name}.png/.pdf")

    # Show figure in notebook/console
    if show:
        plt.show()

    # Close figure to free memory
    plt.close(fig)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data(tokenizer):
    """Load all datasets"""
    print_header("LOADING DATASETS", "=")

    # Load original data
    original_df = pd.read_csv(ORIGINAL_DATA_PATH)
    print(f"✓ Original data: {len(original_df)} samples")
    print(f"  - Class distribution: {original_df['label'].value_counts().to_dict()}")

    # Load new data
    new_df = pd.read_csv(NEW_DATA_PATH)
    print(f"✓ New data: {len(new_df)} samples")
    print(f"  - Class distribution: {new_df['label'].value_counts().to_dict()}")

    # Load test sets
    print("\n📋 Test sets:")
    test_datasets = {}
    test_dfs = {}

    for name, path in TEST_SET_PATHS.items():
        df = pd.read_csv(path)
        test_dfs[name] = df

        dataset = Dataset.from_pandas(df[['text', 'label']])
        dataset = dataset.map(
            lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128),
            batched=True
        )
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        test_datasets[name] = dataset

        class_dist = df['label'].value_counts().to_dict()
        print(f"  ✓ {name}: {len(df)} samples (Class 0: {class_dist.get(0, 0)}, Class 1: {class_dist.get(1, 0)})")

    return original_df, new_df, test_datasets, test_dfs

# ============================================================================
# EWC IMPLEMENTATION
# ============================================================================

class EWCTrainer(Trainer):
    """Trainer with Elastic Weight Consolidation"""

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
    """Compute Fisher Information Matrix"""
    model.eval()
    fisher_dict = {}

    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param)

    num_samples = min(num_samples, len(dataset))
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    for idx in indices:
        sample = dataset[int(idx)]

        # Handle both tensor and list formats (FIX)
        if isinstance(sample['input_ids'], torch.Tensor):
            input_ids = sample['input_ids'].unsqueeze(0).to(device)
            attention_mask = sample['attention_mask'].unsqueeze(0).to(device)
            if sample['label'].dim() == 0:
                labels = sample['label'].unsqueeze(0).to(device)
            else:
                labels = sample['label'].to(device)
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
    """Callback to record loss history per epoch"""

    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.current_train_loss = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            self.current_train_loss.append(logs['loss'])

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.current_train_loss:
            self.train_losses.append(np.mean(self.current_train_loss))
            self.current_train_loss = []

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and 'eval_loss' in metrics:
            self.val_losses.append(metrics['eval_loss'])

# ============================================================================
# EVALUATION
# ============================================================================

def compute_metrics(eval_pred):
    """Compute evaluation metrics"""
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
        'f1_class_0': f1_per_class[0],
        'f1_class_1': f1_per_class[1],
        'precision_class_0': precision_per_class[0],
        'precision_class_1': precision_per_class[1],
        'recall_class_0': recall_per_class[0],
        'recall_class_1': recall_per_class[1],
    }

def evaluate_on_test_sets(model, test_datasets, device):
    """Evaluate model on all test sets"""
    model.eval()
    results = {}
    predictions_dict = {}

    for name, dataset in test_datasets.items():
        trainer = Trainer(
            model=model,
            compute_metrics=compute_metrics,
        )

        eval_results = trainer.evaluate(dataset)
        results[name] = {
            'f1_macro': eval_results['eval_f1_macro'],
            'accuracy': eval_results['eval_accuracy'],
            'precision': eval_results['eval_precision_macro'],
            'recall': eval_results['eval_recall_macro'],
            'f1_class_0': eval_results['eval_f1_class_0'],
            'f1_class_1': eval_results['eval_f1_class_1'],
        }

        # Get predictions for confusion matrix
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
    use_ewc: bool = False,
    ewc_dataset=None,
    device='cuda'
):
    """Train a model with specified configuration"""

    set_seed(seed)

    print(f"\n{'='*70}")
    print(f"Training: {method_name} (Seed: {seed})")
    print(f"LoRA: {use_lora} (r={lora_r}), EWC: {use_ewc}")
    print(f"GPU: {get_gpu_memory()}")
    print(f"{'='*70}")

    # Apply LoRA if needed
    trainable_params = 0
    total_params = 0

    if use_lora:
        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0.1,
            target_modules=["query", "key", "value"],
        )
        model = get_peft_model(model, lora_config)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"✓ LoRA applied (r={lora_r}, alpha={lora_r * 2})")
        print(f"  Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
    else:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = trainable_params
        print(f"✓ Full fine-tuning: {trainable_params:,} parameters")

    # Compute Fisher for EWC
    fisher_dict = None
    optimal_params = None

    if use_ewc and ewc_dataset is not None:
        print("Computing Fisher Information for EWC...")
        fisher_dict = compute_fisher(model, ewc_dataset, device, EWC_SAMPLES)
        optimal_params = {name: param.clone() for name, param in model.named_parameters() if param.requires_grad}
        print("✓ EWC Fisher computed")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=f'{OUTPUT_DIR}/checkpoints/{method_name}_{seed}',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        logging_steps=50,
        seed=seed,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    # Create loss callback
    loss_callback = LossHistoryCallback()

    # Create trainer
    if use_ewc:
        trainer = EWCTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            ewc_lambda=EWC_LAMBDA,
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
    print(f"✓ Training completed in {training_time:.2f} seconds")

    # Evaluate
    results, predictions = evaluate_on_test_sets(model, test_datasets, device)

    # Print results
    print(f"\n📊 Evaluation Results for {method_name}:")
    print(f"   {'Test Set':<15} | {'F1-macro':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
    print(f"   {'-'*65}")
    for test_name, metrics in results.items():
        print(f"   {test_name:<15} | {metrics['f1_macro']:.4f}     | {metrics['accuracy']:.4f}     | {metrics['precision']:.4f}     | {metrics['recall']:.4f}")

    # Collect loss history
    loss_history = {
        'train_losses': loss_callback.train_losses,
        'val_losses': loss_callback.val_losses,
    }

    return results, predictions, training_time, loss_history, trainable_params

# ============================================================================
# BWT/FWT CALCULATION
# ============================================================================

def calculate_bwt_fwt(results, original_f1_historical, original_f1_contemporary):
    """
    Calculate Backward Transfer (BWT) and Forward Transfer (FWT)

    BWT = Performance on old task AFTER learning - Performance BEFORE
    FWT = Performance on new task AFTER learning - Performance BEFORE
    """
    f1_historical_after = results['unused_old']['f1_macro']
    f1_contemporary_after = results['balanced_500']['f1_macro']

    bwt = f1_historical_after - original_f1_historical
    fwt = f1_contemporary_after - original_f1_contemporary

    return bwt, fwt

# ============================================================================
# PHASE 1: ORIGINAL MODEL EVALUATION
# ============================================================================

def run_phase1_original_model(test_datasets, device):
    """Evaluate original model to get baseline"""
    print_header("PHASE 1: ORIGINAL MODEL EVALUATION", "#")

    model = load_local_model(MODEL_PATH).to(device)
    results, predictions = evaluate_on_test_sets(model, test_datasets, device)

    print(f"\n📊 Original Model Results:")
    print(f"   {'Test Set':<15} | {'F1-macro':<10} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10}")
    print(f"   {'-'*65}")
    for test_name, metrics in results.items():
        print(f"   {test_name:<15} | {metrics['f1_macro']:.4f}     | {metrics['accuracy']:.4f}     | {metrics['precision']:.4f}     | {metrics['recall']:.4f}")

    original_f1_historical = results['unused_old']['f1_macro']
    original_f1_contemporary = results['balanced_500']['f1_macro']

    del model
    clear_memory()

    # Save intermediate
    with open(f'{OUTPUT_DIR}/phase1_original_intermediate.json', 'w') as f:
        json.dump({'results': results}, f, indent=2)

    return results, predictions, original_f1_historical, original_f1_contemporary

# ============================================================================
# PHASE 2: LORA ABLATION (FIRST!)
# ============================================================================

def run_phase2_lora_ablation(original_df, new_df, test_datasets, tokenizer,
                             original_f1_historical, original_f1_contemporary, device):
    """
    Run LoRA ablation study FIRST to determine optimal rank.
    """
    print_header("PHASE 2: LoRA ABLATION STUDY (DETERMINING OPTIMAL RANK)", "#")

    print(f"\n🔬 Testing LoRA ranks: {LORA_RANKS_TO_TEST}")
    print(f"📊 Seeds per rank: {len(ABLATION_SEEDS)}")
    print(f"🎯 Selection criterion: {LORA_SELECTION_CRITERION}")

    # Prepare data
    new_train_df, new_val_df = train_test_split(new_df, test_size=0.2, random_state=42, stratify=new_df['label'])

    def create_dataset(df):
        dataset = Dataset.from_pandas(df[['text', 'label']])
        dataset = dataset.map(
            lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128),
            batched=True
        )
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return dataset

    train_dataset = create_dataset(new_train_df)
    val_dataset = create_dataset(new_val_df)

    print(f"  → Training on new data only: {len(new_train_df)} samples")

    # Store results
    ablation_results = {}
    ablation_times = {}
    ablation_losses = {}
    ablation_predictions = {}
    ablation_bwt_fwt = {}

    for lora_r in LORA_RANKS_TO_TEST:
        print(f"\n{'-'*60}")
        print(f"Testing LoRA r={lora_r}")
        print(f"{'-'*60}")

        method_key = f'lora_r{lora_r}'
        rank_results = []
        rank_times = []
        rank_losses = []
        rank_bwt_list = []
        rank_fwt_list = []
        rank_predictions = None

        for seed in ABLATION_SEEDS:
            print(f"\n--- Seed {seed} ---")

            model = load_local_model(MODEL_PATH).to(device)

            results, predictions, training_time, loss_history, trainable = train_model(
                method_name=method_key,
                model=model,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                test_datasets=test_datasets,
                tokenizer=tokenizer,
                seed=seed,
                use_lora=True,
                lora_r=lora_r,
                use_ewc=False,
                device=device
            )

            # Calculate BWT and FWT
            bwt, fwt = calculate_bwt_fwt(results, original_f1_historical, original_f1_contemporary)

            rank_results.append(results)
            rank_times.append(training_time)
            rank_losses.append(loss_history)
            rank_bwt_list.append(bwt)
            rank_fwt_list.append(fwt)
            rank_predictions = predictions

            print(f"   BWT: {bwt:+.4f}, FWT: {fwt:+.4f}")

            del model
            clear_memory()

        # Store results
        ablation_results[method_key] = rank_results
        ablation_times[method_key] = rank_times
        ablation_losses[method_key] = rank_losses
        ablation_predictions[method_key] = rank_predictions

        # Aggregate BWT/FWT
        ablation_bwt_fwt[method_key] = {
            'rank': lora_r,
            'trainable_params': trainable,
            'mean_bwt': np.mean(rank_bwt_list),
            'std_bwt': np.std(rank_bwt_list),
            'mean_fwt': np.mean(rank_fwt_list),
            'std_fwt': np.std(rank_fwt_list),
            'bwt_list': rank_bwt_list,
            'fwt_list': rank_fwt_list,
            'mean_historical': np.mean([r['unused_old']['f1_macro'] for r in rank_results]),
            'std_historical': np.std([r['unused_old']['f1_macro'] for r in rank_results]),
            'mean_contemporary': np.mean([r['balanced_500']['f1_macro'] for r in rank_results]),
            'std_contemporary': np.std([r['balanced_500']['f1_macro'] for r in rank_results]),
        }

        print(f"\n📊 Summary for LoRA r={lora_r}:")
        print(f"   BWT: {ablation_bwt_fwt[method_key]['mean_bwt']:+.4f} ± {ablation_bwt_fwt[method_key]['std_bwt']:.4f}")
        print(f"   FWT: {ablation_bwt_fwt[method_key]['mean_fwt']:+.4f} ± {ablation_bwt_fwt[method_key]['std_fwt']:.4f}")

        # Save intermediate
        with open(f'{OUTPUT_DIR}/phase2_{method_key}_intermediate.json', 'w') as f:
            json.dump({'results': rank_results, 'times': rank_times, 'bwt_fwt': ablation_bwt_fwt[method_key]}, f, indent=2, default=str)

    # ========================================
    # SELECT BEST RANK
    # ========================================
    print_header("SELECTING OPTIMAL LoRA RANK", "-")

    print("\n📋 LoRA Ablation Summary:")
    print(f"{'Rank':<10} | {'BWT':<20} | {'FWT':<20} | {'Historical F1':<20} | {'Contemporary F1':<20}")
    print(f"{'-'*95}")

    best_rank = None
    best_score = float('-inf')

    for lora_r in LORA_RANKS_TO_TEST:
        key = f'lora_r{lora_r}'
        data = ablation_bwt_fwt[key]

        print(f"r={lora_r:<7} | {data['mean_bwt']:+.4f} ± {data['std_bwt']:.4f}  | "
              f"{data['mean_fwt']:+.4f} ± {data['std_fwt']:.4f}  | "
              f"{data['mean_historical']:.4f} ± {data['std_historical']:.4f}  | "
              f"{data['mean_contemporary']:.4f} ± {data['std_contemporary']:.4f}")

        # Calculate score
        if LORA_SELECTION_CRITERION == 'bwt':
            score = data['mean_bwt']
        elif LORA_SELECTION_CRITERION == 'fwt':
            score = data['mean_fwt']
        elif LORA_SELECTION_CRITERION == 'balanced':
            score = data['mean_bwt'] + data['mean_fwt']
        elif LORA_SELECTION_CRITERION == 'historical':
            score = data['mean_historical']
        elif LORA_SELECTION_CRITERION == 'contemporary':
            score = data['mean_contemporary']
        else:
            score = data['mean_bwt'] + data['mean_fwt']

        if score > best_score:
            best_score = score
            best_rank = lora_r

    print(f"\n{'='*60}")
    print(f"🏆 OPTIMAL LoRA RANK SELECTED: r={best_rank}")
    print(f"   Selection criterion: {LORA_SELECTION_CRITERION}")
    print(f"   Score: {best_score:.4f}")
    print(f"{'='*60}")

    # Print detailed ablation comparison
    print(f"\n{'='*80}")
    print(f"📊 LORA ABLATION - DETAILED COMPARISON")
    print(f"{'='*80}")
    print(f"\n{'Rank':<8} | {'BWT':<20} | {'FWT':<20} | {'BWT+FWT (Balance)':<18}")
    print(f"{'-'*75}")
    for lora_r in LORA_RANKS_TO_TEST:
        key = f'lora_r{lora_r}'
        data = ablation_bwt_fwt[key]
        balance = data['mean_bwt'] + data['mean_fwt']
        marker = " ⭐ BEST" if lora_r == best_rank else ""
        print(f"r={lora_r:<5} | {data['mean_bwt']:+.4f} ± {data['std_bwt']:.4f} | "
              f"{data['mean_fwt']:+.4f} ± {data['std_fwt']:.4f} | "
              f"{balance:+.4f}{marker}")

    # Save ablation summary
    with open(f'{OUTPUT_DIR}/lora_ablation_summary.json', 'w') as f:
        json.dump({
            'ablation_bwt_fwt': {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in ablation_bwt_fwt.items()},
            'optimal_rank': best_rank,
            'selection_criterion': LORA_SELECTION_CRITERION,
        }, f, indent=2)

    return ablation_results, ablation_times, ablation_losses, ablation_predictions, ablation_bwt_fwt, best_rank

# ============================================================================
# PHASE 3: MAIN METHODS
# ============================================================================

def run_phase3_main_methods(original_df, new_df, test_datasets, tokenizer,
                            original_f1_historical, original_f1_contemporary,
                            optimal_lora_rank, device):
    """Run main continual learning methods with optimal LoRA rank"""

    print_header(f"PHASE 3: MAIN METHODS (LoRA using r={optimal_lora_rank})", "#")

    # Prepare datasets
    new_train_df, new_val_df = train_test_split(new_df, test_size=0.2, random_state=42, stratify=new_df['label'])

    def create_dataset(df):
        dataset = Dataset.from_pandas(df[['text', 'label']])
        dataset = dataset.map(
            lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128),
            batched=True
        )
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return dataset

    new_train_dataset = create_dataset(new_train_df)
    new_val_dataset = create_dataset(new_val_df)

    # Create replay buffer
    replay_size = int(len(new_train_df) * REPLAY_RATIO / (1 - REPLAY_RATIO))
    replay_df = original_df.sample(n=min(replay_size, len(original_df)), random_state=42)
    combined_df = pd.concat([replay_df, new_train_df], ignore_index=True)
    combined_train_dataset = create_dataset(combined_df)

    print(f"  → New data only: {len(new_train_df)} samples")
    print(f"  → Replay buffer: {len(replay_df)} old + {len(new_train_df)} new = {len(combined_df)} samples")

    # Create EWC dataset
    ewc_df = original_df.sample(n=min(EWC_SAMPLES, len(original_df)), random_state=42)
    ewc_dataset = create_dataset(ewc_df)
    print(f"  → EWC dataset: {len(ewc_df)} samples")

    # Define methods
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
        print(f"Method: {method_name.upper()}")
        print(f"{'-'*60}")

        method_results = []
        method_times = []
        method_losses = []
        method_predictions = None
        method_params = 0

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")

            if config['use_replay']:
                train_dataset = combined_train_dataset
                print(f"  → Training on replay buffer: {len(combined_df)} samples")
            else:
                train_dataset = new_train_dataset
                print(f"  → Training on new data only: {len(new_train_df)} samples")

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
                use_ewc=config['use_ewc'],
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
        all_params[method_name] = method_params

        # Save intermediate
        with open(f'{OUTPUT_DIR}/phase3_{method_name}_intermediate.json', 'w') as f:
            json.dump({'results': method_results, 'times': method_times}, f, indent=2)

    # Print Phase 3 Summary
    print_phase_summary("PHASE 3: MAIN METHODS RESULTS", all_results, original_f1_historical, original_f1_contemporary)

    return all_results, all_times, all_losses, all_predictions, all_params

# ============================================================================
# PHASE 4: HYBRID METHODS
# ============================================================================

def run_phase4_hybrid_methods(original_df, new_df, test_datasets, tokenizer,
                              original_f1_historical, original_f1_contemporary,
                              optimal_lora_rank, device):
    """Run hybrid continual learning methods with optimal LoRA rank"""

    print_header(f"PHASE 4: HYBRID METHODS (LoRA using r={optimal_lora_rank})", "#")

    # Prepare datasets
    new_train_df, new_val_df = train_test_split(new_df, test_size=0.2, random_state=42, stratify=new_df['label'])

    def create_dataset(df):
        dataset = Dataset.from_pandas(df[['text', 'label']])
        dataset = dataset.map(
            lambda x: tokenizer(x['text'], truncation=True, padding='max_length', max_length=128),
            batched=True
        )
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return dataset

    new_train_dataset = create_dataset(new_train_df)
    new_val_dataset = create_dataset(new_val_df)

    # Create replay buffer
    replay_size = int(len(new_train_df) * REPLAY_RATIO / (1 - REPLAY_RATIO))
    replay_df = original_df.sample(n=min(replay_size, len(original_df)), random_state=42)
    combined_df = pd.concat([replay_df, new_train_df], ignore_index=True)
    combined_train_dataset = create_dataset(combined_df)

    # Create EWC dataset
    ewc_df = original_df.sample(n=min(EWC_SAMPLES, len(original_df)), random_state=42)
    ewc_dataset = create_dataset(ewc_df)

    # Define hybrid methods
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
        print(f"Method: {method_name.upper()}")
        print(f"{'-'*60}")

        method_results = []
        method_times = []
        method_losses = []
        method_predictions = None
        method_params = 0

        for seed in SEEDS:
            print(f"\n--- Seed {seed} ---")

            if config['use_replay']:
                train_dataset = combined_train_dataset
            else:
                train_dataset = new_train_dataset

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
                use_ewc=config['use_ewc'],
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
        all_params[method_name] = method_params

        # Save intermediate
        with open(f'{OUTPUT_DIR}/phase4_{method_name}_intermediate.json', 'w') as f:
            json.dump({'results': method_results, 'times': method_times}, f, indent=2)

    # Print Phase 4 Summary
    print_phase_summary("PHASE 4: HYBRID METHODS RESULTS", all_results, original_f1_historical, original_f1_contemporary)

    return all_results, all_times, all_losses, all_predictions, all_params

# ============================================================================
# TABLES GENERATION
# ============================================================================

def generate_all_tables(original_results, main_results, hybrid_results,
                        main_times, hybrid_times, ablation_bwt_fwt,
                        original_f1_historical, original_f1_contemporary,
                        optimal_lora_rank):
    """Generate all 6 tables"""

    print_header("GENERATING TABLES", "=")

    # Combine all results
    all_results = {'original': [original_results]}
    all_results.update(main_results)
    all_results.update(hybrid_results)

    all_times = {}
    all_times.update(main_times)
    all_times.update(hybrid_times)

    # ========================================
    # TABLE 1: F1-Macro
    # ========================================
    print("\n" + "="*80)
    print("TABLE 1: OVERALL PERFORMANCE (F1-Macro)")
    print("="*80)

    print(f"\n{'Method':<15} | {'Historical':<18} | {'Contemporary':<18} | {'Mixed 80-20':<18} | {'Mixed 40-60':<18}")
    print(f"{'-'*95}")

    table1_data = []
    for method in METHODS_ORDER:
        if method in all_results:
            results_list = all_results[method]
            row = {'Method': METHOD_LABELS_SHORT.get(method, method)}

            for test_set in ['unused_old', 'balanced_500', 'mixed_2080', 'mixed_4060']:
                if len(results_list) > 1:
                    f1s = [r[test_set]['f1_macro'] for r in results_list]
                    row[test_set] = f"{np.mean(f1s):.3f} ± {np.std(f1s):.3f}"
                else:
                    row[test_set] = f"{results_list[0][test_set]['f1_macro']:.3f} ± 0.000"

            table1_data.append(row)
            print(f"{row['Method']:<15} | {row['unused_old']:<18} | {row['balanced_500']:<18} | {row['mixed_2080']:<18} | {row['mixed_4060']:<18}")

    # Save
    pd.DataFrame(table1_data).to_csv(f'{TABLES_DIR}/table1_f1_macro.csv', index=False)

    # ========================================
    # TABLE 2: Accuracy
    # ========================================
    print("\n" + "="*80)
    print("TABLE 2: OVERALL PERFORMANCE (Accuracy)")
    print("="*80)

    print(f"\n{'Method':<15} | {'Historical':<18} | {'Contemporary':<18} | {'Mixed 80-20':<18} | {'Mixed 40-60':<18}")
    print(f"{'-'*95}")

    table2_data = []
    for method in METHODS_ORDER:
        if method in all_results:
            results_list = all_results[method]
            row = {'Method': METHOD_LABELS_SHORT.get(method, method)}

            for test_set in ['unused_old', 'balanced_500', 'mixed_2080', 'mixed_4060']:
                if len(results_list) > 1:
                    accs = [r[test_set]['accuracy'] for r in results_list]
                    row[test_set] = f"{np.mean(accs):.3f} ± {np.std(accs):.3f}"
                else:
                    row[test_set] = f"{results_list[0][test_set]['accuracy']:.3f} ± 0.000"

            table2_data.append(row)
            print(f"{row['Method']:<15} | {row['unused_old']:<18} | {row['balanced_500']:<18} | {row['mixed_2080']:<18} | {row['mixed_4060']:<18}")

    pd.DataFrame(table2_data).to_csv(f'{TABLES_DIR}/table2_accuracy.csv', index=False)

    # ========================================
    # TABLE 3: BWT & FWT
    # ========================================
    print("\n" + "="*80)
    print("TABLE 3: BACKWARD & FORWARD TRANSFER ANALYSIS")
    print("="*80)

    print(f"\n{'Method':<15} | {'BWT':<20} | {'FWT':<20} | {'Historical F1':<18} | {'Contemporary F1':<18}")
    print(f"{'-'*100}")

    table3_data = []
    for method in METHODS_ORDER:
        if method == 'original':
            continue
        if method in all_results:
            results_list = all_results[method]

            if len(results_list) > 1:
                bwts = [r['unused_old']['f1_macro'] - original_f1_historical for r in results_list]
                fwts = [r['balanced_500']['f1_macro'] - original_f1_contemporary for r in results_list]
                hist_f1s = [r['unused_old']['f1_macro'] for r in results_list]
                cont_f1s = [r['balanced_500']['f1_macro'] for r in results_list]

                row = {
                    'Method': METHOD_LABELS_SHORT.get(method, method),
                    'BWT': f"{np.mean(bwts):+.3f} ± {np.std(bwts):.3f}",
                    'FWT': f"{np.mean(fwts):+.3f} ± {np.std(fwts):.3f}",
                    'Historical F1': f"{np.mean(hist_f1s):.3f} ± {np.std(hist_f1s):.3f}",
                    'Contemporary F1': f"{np.mean(cont_f1s):.3f} ± {np.std(cont_f1s):.3f}",
                }
            else:
                bwt = results_list[0]['unused_old']['f1_macro'] - original_f1_historical
                fwt = results_list[0]['balanced_500']['f1_macro'] - original_f1_contemporary
                row = {
                    'Method': METHOD_LABELS_SHORT.get(method, method),
                    'BWT': f"{bwt:+.3f} ± 0.000",
                    'FWT': f"{fwt:+.3f} ± 0.000",
                    'Historical F1': f"{results_list[0]['unused_old']['f1_macro']:.3f} ± 0.000",
                    'Contemporary F1': f"{results_list[0]['balanced_500']['f1_macro']:.3f} ± 0.000",
                }

            table3_data.append(row)
            print(f"{row['Method']:<15} | {row['BWT']:<20} | {row['FWT']:<20} | {row['Historical F1']:<18} | {row['Contemporary F1']:<18}")

    pd.DataFrame(table3_data).to_csv(f'{TABLES_DIR}/table3_bwt_fwt.csv', index=False)

    # ========================================
    # TABLE 4: Per-Class F1
    # ========================================
    print("\n" + "="*80)
    print("TABLE 4: PER-CLASS F1 SCORES (Contemporary Test Set)")
    print("="*80)

    print(f"\n{'Method':<15} | {'F1 (NOT_OFF)':<18} | {'F1 (OFF)':<18} | {'F1-macro':<18}")
    print(f"{'-'*75}")

    table4_data = []
    for method in METHODS_ORDER:
        if method in all_results:
            results_list = all_results[method]

            if len(results_list) > 1:
                f1_0 = [r['balanced_500']['f1_class_0'] for r in results_list]
                f1_1 = [r['balanced_500']['f1_class_1'] for r in results_list]
                f1_m = [r['balanced_500']['f1_macro'] for r in results_list]

                row = {
                    'Method': METHOD_LABELS_SHORT.get(method, method),
                    'F1_NOT_OFF': f"{np.mean(f1_0):.3f} ± {np.std(f1_0):.3f}",
                    'F1_OFF': f"{np.mean(f1_1):.3f} ± {np.std(f1_1):.3f}",
                    'F1_macro': f"{np.mean(f1_m):.3f} ± {np.std(f1_m):.3f}",
                }
            else:
                row = {
                    'Method': METHOD_LABELS_SHORT.get(method, method),
                    'F1_NOT_OFF': f"{results_list[0]['balanced_500']['f1_class_0']:.3f} ± 0.000",
                    'F1_OFF': f"{results_list[0]['balanced_500']['f1_class_1']:.3f} ± 0.000",
                    'F1_macro': f"{results_list[0]['balanced_500']['f1_macro']:.3f} ± 0.000",
                }

            table4_data.append(row)
            print(f"{row['Method']:<15} | {row['F1_NOT_OFF']:<18} | {row['F1_OFF']:<18} | {row['F1_macro']:<18}")

    pd.DataFrame(table4_data).to_csv(f'{TABLES_DIR}/table4_per_class_f1.csv', index=False)

    # ========================================
    # TABLE 5: Training Time
    # ========================================
    print("\n" + "="*80)
    print("TABLE 5: TRAINING TIME SUMMARY")
    print("="*80)

    print(f"\n{'Method':<15} | {'Avg Time (s)':<15} | {'Std (s)':<15} | {'Num Seeds':<15}")
    print(f"{'-'*65}")

    table5_data = []
    for method in METHODS_ORDER:
        if method == 'original':
            continue
        if method in all_times:
            times = all_times[method]
            row = {
                'Method': METHOD_LABELS_SHORT.get(method, method),
                'Avg Time': f"{np.mean(times):.1f}",
                'Std': f"{np.std(times):.1f}",
                'Num Seeds': len(times),
            }
            table5_data.append(row)
            print(f"{row['Method']:<15} | {row['Avg Time']:<15} | {row['Std']:<15} | {row['Num Seeds']:<15}")

    pd.DataFrame(table5_data).to_csv(f'{TABLES_DIR}/table5_training_time.csv', index=False)

    # ========================================
    # TABLE 6: LoRA Ablation
    # ========================================
    print("\n" + "="*80)
    print("TABLE 6: LORA ABLATION STUDY")
    print("="*80)

    print(f"\n{'Config':<12} | {'Params':<12} | {'Historical F1':<18} | {'Contemporary F1':<18} | {'BWT':<15}")
    print(f"{'-'*85}")

    table6_data = []
    for lora_r in LORA_RANKS_TO_TEST:
        key = f'lora_r{lora_r}'
        if key in ablation_bwt_fwt:
            data = ablation_bwt_fwt[key]

            # Format params
            params = data.get('trainable_params', 0)
            if params > 1000000:
                params_str = f"{params/1000000:.1f}M"
            else:
                params_str = f"{params/1000:.0f}K"

            row = {
                'Config': f"r={lora_r}",
                'Params': params_str,
                'Historical F1': f"{data['mean_historical']:.3f} ± {data['std_historical']:.3f}",
                'Contemporary F1': f"{data['mean_contemporary']:.3f} ± {data['std_contemporary']:.3f}",
                'BWT': f"{data['mean_bwt']:+.3f}",
            }

            # Mark optimal
            if lora_r == optimal_lora_rank:
                row['Config'] = f"r={lora_r} ⭐"

            table6_data.append(row)
            print(f"{row['Config']:<12} | {row['Params']:<12} | {row['Historical F1']:<18} | {row['Contemporary F1']:<18} | {row['BWT']:<15}")

    pd.DataFrame(table6_data).to_csv(f'{TABLES_DIR}/table6_lora_ablation.csv', index=False)

    print(f"\n{'='*80}")
    print(f"✅ ALL 6 TABLES GENERATED AND DISPLAYED ABOVE")
    print(f"✅ Tables saved to: {TABLES_DIR}/")
    print(f"{'='*80}")

    return table1_data, table2_data, table3_data, table4_data, table5_data, table6_data

# ============================================================================
# FIGURES GENERATION
# ============================================================================

def generate_all_figures(original_results, main_results, hybrid_results,
                        main_times, hybrid_times, main_losses, hybrid_losses,
                        main_predictions, hybrid_predictions,
                        ablation_results, ablation_bwt_fwt,
                        original_f1_historical, original_f1_contemporary,
                        optimal_lora_rank, original_predictions):
    """Generate all 8 figures"""

    print_header("GENERATING ALL VISUALIZATIONS (8 Figures)", "=")

    # Combine results
    all_results = {'original': [original_results]}
    all_results.update(main_results)
    all_results.update(hybrid_results)

    all_times = {}
    all_times.update(main_times)
    all_times.update(hybrid_times)

    all_losses = {}
    all_losses.update(main_losses)
    all_losses.update(hybrid_losses)

    all_predictions = {'original': original_predictions}
    all_predictions.update(main_predictions)
    all_predictions.update(hybrid_predictions)

    # ========================================
    # FIGURE 1: F1-Macro Bar Chart
    # ========================================
    print("\n📊 [1/8] Creating Figure 1: F1-Macro Bar Chart...")

    fig, ax = plt.subplots(figsize=(14, 6))

    methods = METHODS_ORDER
    test_sets = ['unused_old', 'balanced_500', 'mixed_2080', 'mixed_4060']
    x = np.arange(len(methods))
    width = 0.2

    for i, test_set in enumerate(test_sets):
        means = []
        stds = []
        for method in methods:
            if method in all_results:
                results_list = all_results[method]
                if len(results_list) > 1:
                    f1s = [r[test_set]['f1_macro'] for r in results_list]
                    means.append(np.mean(f1s))
                    stds.append(np.std(f1s))
                else:
                    means.append(results_list[0][test_set]['f1_macro'])
                    stds.append(0)
            else:
                means.append(0)
                stds.append(0)

        bars = ax.bar(x + i*width, means, width, yerr=stds,
                     label=TEST_SET_LABELS_SHORT[test_set],
                     color=COLORBLIND_PALETTE[i],
                     capsize=3, alpha=0.85)

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Macro', fontsize=12, fontweight='bold')
    ax.set_title('F1-Macro Performance Across Test Sets (mean ± std)', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels([METHOD_LABELS_SHORT.get(m, m) for m in methods], rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0.6, 1.0)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_figure(fig, 'Figure_1_F1_Comparison')

    # ========================================
    # FIGURE 2: BWT vs FWT Scatter
    # ========================================
    print("\n📊 [2/8] Creating Figure 2: BWT vs FWT Scatter Plot...")

    fig, ax = plt.subplots(figsize=(10, 8))

    for i, method in enumerate(METHODS_ORDER):
        if method == 'original':
            continue
        if method in all_results:
            results_list = all_results[method]
            if len(results_list) > 1:
                bwts = [r['unused_old']['f1_macro'] - original_f1_historical for r in results_list]
                fwts = [r['balanced_500']['f1_macro'] - original_f1_contemporary for r in results_list]
                mean_bwt, std_bwt = np.mean(bwts), np.std(bwts)
                mean_fwt, std_fwt = np.mean(fwts), np.std(fwts)
            else:
                mean_bwt = results_list[0]['unused_old']['f1_macro'] - original_f1_historical
                mean_fwt = results_list[0]['balanced_500']['f1_macro'] - original_f1_contemporary
                std_bwt, std_fwt = 0, 0

            ax.errorbar(mean_bwt, mean_fwt, xerr=std_bwt, yerr=std_fwt,
                       fmt='o', markersize=12, capsize=5,
                       color=COLORBLIND_PALETTE[i-1],
                       label=METHOD_LABELS_SHORT.get(method, method))

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('BWT (Backward Transfer) →', fontsize=12, fontweight='bold')
    ax.set_ylabel('FWT (Forward Transfer) →', fontsize=12, fontweight='bold')
    ax.set_title('Knowledge Retention vs. Adaptation Trade-off', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    # Add quadrant labels
    ax.text(0.02, 0.98, 'Ideal\n(High FWT, Low Forgetting)', transform=ax.transAxes,
            fontsize=10, verticalalignment='top', color='green', fontweight='bold')
    ax.text(0.02, 0.02, 'Poor\n(Low FWT, High Forgetting)', transform=ax.transAxes,
            fontsize=10, verticalalignment='bottom', color='red', fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'Figure_2_BWT_FWT')

    # ========================================
    # FIGURE 3: Parameter Efficiency
    # ========================================
    print("\n📊 [3/8] Creating Figure 3: Parameter Efficiency...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Params data (approximate)
    params_data = {
        'naive_ft': 135194882,
        'er': 135194882,
        'ewc': 135194882,
        'full_er_ewc': 135194882,
        'lora': 591362,  # r=16
        'lora_er': 591362,
        'lora_ewc': 591362,
        'lora_er_ewc': 591362,
    }

    methods_for_params = [m for m in METHODS_ORDER if m != 'original' and m in all_results]

    # Left: Params vs F1
    ax1 = axes[0]
    for method in methods_for_params:
        if method in all_results:
            results_list = all_results[method]
            f1s = [r['balanced_500']['f1_macro'] for r in results_list]
            mean_f1 = np.mean(f1s)
            params = params_data.get(method, 135194882)

            color_idx = METHODS_ORDER.index(method) - 1
            ax1.scatter(params/1e6, mean_f1, s=150,
                       color=COLORBLIND_PALETTE[color_idx % len(COLORBLIND_PALETTE)],
                       label=METHOD_LABELS_SHORT.get(method, method), zorder=5)

    ax1.set_xlabel('Trainable Parameters (Millions)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1-Macro (Contemporary)', fontsize=12, fontweight='bold')
    ax1.set_title('Parameter Efficiency vs Performance', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_xscale('log')
    ax1.grid(True, alpha=0.3)

    # Right: Params vs Time
    ax2 = axes[1]
    for method in methods_for_params:
        if method in all_times:
            times = all_times[method]
            mean_time = np.mean(times)
            params = params_data.get(method, 135194882)

            color_idx = METHODS_ORDER.index(method) - 1
            ax2.scatter(params/1e6, mean_time, s=150,
                       color=COLORBLIND_PALETTE[color_idx % len(COLORBLIND_PALETTE)],
                       label=METHOD_LABELS_SHORT.get(method, method), zorder=5)

    ax2.set_xlabel('Trainable Parameters (Millions)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax2.set_title('Parameter Efficiency vs Training Time', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_xscale('log')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'Figure_3_Parameter_Efficiency')

    # ========================================
    # FIGURE 4: Training Time
    # ========================================
    print("\n📊 [4/8] Creating Figure 4: Training Time Comparison...")

    fig, ax = plt.subplots(figsize=(12, 6))

    methods_for_time = [m for m in METHODS_ORDER if m != 'original' and m in all_times]
    times_means = [np.mean(all_times[m]) for m in methods_for_time]
    times_stds = [np.std(all_times[m]) for m in methods_for_time]

    colors = [COLORBLIND_PALETTE[METHODS_ORDER.index(m) - 1] for m in methods_for_time]

    bars = ax.bar(range(len(methods_for_time)), times_means, yerr=times_stds,
                 color=colors, capsize=5, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Comparison (mean ± std)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(methods_for_time)))
    ax.set_xticklabels([METHOD_LABELS_SHORT.get(m, m) for m in methods_for_time], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mean in zip(bars, times_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
               f'{mean:.1f}s', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, 'Figure_4_Training_Time')

    # ========================================
    # FIGURE 5: Loss Curves
    # ========================================
    print("\n📊 [5/8] Creating Figure 5: Loss Curves...")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    methods_for_loss = [m for m in METHODS_ORDER if m != 'original' and m in all_losses]

    # Left: Training Loss
    ax1 = axes[0]
    for method in methods_for_loss:
        if method in all_losses:
            losses_list = all_losses[method]
            # Aggregate across seeds
            all_train = [l['train_losses'] for l in losses_list if l['train_losses']]
            if all_train:
                min_len = min(len(l) for l in all_train)
                if min_len > 0:
                    train_array = np.array([l[:min_len] for l in all_train])
                    mean_train = np.mean(train_array, axis=0)
                    std_train = np.std(train_array, axis=0)
                    epochs = range(1, min_len + 1)

                    color_idx = METHODS_ORDER.index(method) - 1
                    ax1.plot(epochs, mean_train, marker='o',
                            label=METHOD_LABELS_SHORT.get(method, method),
                            color=COLORBLIND_PALETTE[color_idx % len(COLORBLIND_PALETTE)],
                            linewidth=2)
                    ax1.fill_between(epochs, mean_train - std_train, mean_train + std_train, alpha=0.2)

    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)

    # Right: Validation Loss
    ax2 = axes[1]
    for method in methods_for_loss:
        if method in all_losses:
            losses_list = all_losses[method]
            all_val = [l['val_losses'] for l in losses_list if l['val_losses']]
            if all_val:
                min_len = min(len(l) for l in all_val)
                if min_len > 0:
                    val_array = np.array([l[:min_len] for l in all_val])
                    mean_val = np.mean(val_array, axis=0)
                    std_val = np.std(val_array, axis=0)
                    epochs = range(1, min_len + 1)

                    color_idx = METHODS_ORDER.index(method) - 1
                    ax2.plot(epochs, mean_val, marker='s',
                            label=METHOD_LABELS_SHORT.get(method, method),
                            color=COLORBLIND_PALETTE[color_idx % len(COLORBLIND_PALETTE)],
                            linewidth=2)
                    ax2.fill_between(epochs, mean_val - std_val, mean_val + std_val, alpha=0.2)

    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax2.set_title('Validation Loss Convergence', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8, ncol=2)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_figure(fig, 'Figure_5_Loss_Curves')

    # ========================================
    # FIGURE 6: LoRA Ablation
    # ========================================
    print("\n📊 [6/8] Creating Figure 6: LoRA Ablation Study...")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ranks = LORA_RANKS_TO_TEST
    x = np.arange(len(ranks))

    hist_means = [ablation_bwt_fwt[f'lora_r{r}']['mean_historical'] for r in ranks]
    hist_stds = [ablation_bwt_fwt[f'lora_r{r}']['std_historical'] for r in ranks]
    cont_means = [ablation_bwt_fwt[f'lora_r{r}']['mean_contemporary'] for r in ranks]
    cont_stds = [ablation_bwt_fwt[f'lora_r{r}']['std_contemporary'] for r in ranks]

    colors = COLORBLIND_PALETTE[:len(ranks)]

    # Historical
    ax1 = axes[0]
    bars1 = ax1.bar(x, hist_means, yerr=hist_stds, capsize=8, color=colors, edgecolor='black', linewidth=2)

    # Highlight optimal
    optimal_idx = ranks.index(optimal_lora_rank)
    bars1[optimal_idx].set_edgecolor('red')
    bars1[optimal_idx].set_linewidth(4)

    ax1.set_ylabel('F1-Macro', fontsize=12, fontweight='bold')
    ax1.set_title('Historical Test Set\n(Knowledge Retention)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'r={r}' for r in ranks], fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    for bar, mean in zip(bars1, hist_means):
        ax1.annotate(f'{mean:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    # Contemporary
    ax2 = axes[1]
    bars2 = ax2.bar(x, cont_means, yerr=cont_stds, capsize=8, color=colors, edgecolor='black', linewidth=2)

    bars2[optimal_idx].set_edgecolor('red')
    bars2[optimal_idx].set_linewidth(4)

    ax2.set_ylabel('F1-Macro', fontsize=12, fontweight='bold')
    ax2.set_title('Contemporary Test Set\n(Adaptation)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'r={r}' for r in ranks], fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    for bar, mean in zip(bars2, cont_means):
        ax2.annotate(f'{mean:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 5), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    plt.suptitle(f'LoRA Rank Ablation Study (Optimal: r={optimal_lora_rank}, highlighted in red)',
                fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'Figure_6_LoRA_Ablation')

    # ========================================
    # FIGURE 7 (A1): Confusion Matrices 9×4
    # ========================================
    print("\n📊 [7/8] Creating Figure A1: Comprehensive Confusion Matrices (9×4)...")

    fig, axes = plt.subplots(9, 4, figsize=(18, 28))

    cm_methods = METHODS_ORDER
    cm_test_sets = ['unused_old', 'mixed_2080', 'mixed_4060', 'balanced_500']

    for row_idx, method in enumerate(cm_methods):
        for col_idx, test_set in enumerate(cm_test_sets):
            ax = axes[row_idx, col_idx]

            preds_key = f'{test_set}_preds'
            labels_key = f'{test_set}_labels'

            if method in all_predictions and preds_key in all_predictions[method]:
                preds = all_predictions[method][preds_key]
                labels = all_predictions[method][labels_key]
                cm = confusion_matrix(labels, preds)

                acc = (cm[0,0] + cm[1,1]) / cm.sum() * 100

                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['NOT_OFF', 'OFF'], yticklabels=['NOT_OFF', 'OFF'],
                           cbar=False, annot_kws={'size': 10, 'fontweight': 'bold'})

                ax.set_title(f'Acc: {acc:.1f}%', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')

            if row_idx == 0:
                ax.set_title(f'{TEST_SET_LABELS_SHORT[test_set]}\nAcc: {acc:.1f}%' if method in all_predictions else TEST_SET_LABELS_SHORT[test_set],
                           fontsize=10, fontweight='bold')

            if col_idx == 0:
                ax.set_ylabel(f'{METHOD_LABELS_SHORT.get(method, method)}\n\nActual', fontsize=9, fontweight='bold')

            if row_idx == len(cm_methods) - 1:
                ax.set_xlabel('Predicted', fontsize=9)

    plt.suptitle('Confusion Matrix Analysis: All Methods × All Test Sets', fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    save_figure(fig, 'Figure_A1_Confusion_Matrix_9x4')

    # ========================================
    # FIGURE 8 (A2): All Methods on Contemporary
    # ========================================
    print("\n📊 [8/8] Creating Figure A2: All Methods Confusion Matrices (3×3)...")

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    axes = axes.flatten()

    test_set = 'balanced_500'

    for idx, method in enumerate(METHODS_ORDER):
        ax = axes[idx]

        preds_key = f'{test_set}_preds'
        labels_key = f'{test_set}_labels'

        if method in all_predictions and preds_key in all_predictions[method]:
            preds = all_predictions[method][preds_key]
            labels = all_predictions[method][labels_key]
            cm = confusion_matrix(labels, preds)

            acc = (cm[0,0] + cm[1,1]) / cm.sum() * 100
            rec = cm[1,1] / (cm[1,0] + cm[1,1]) * 100 if (cm[1,0] + cm[1,1]) > 0 else 0

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['NOT_OFF', 'OFF'], yticklabels=['NOT_OFF', 'OFF'],
                       annot_kws={'size': 14, 'fontweight': 'bold'},
                       cbar=False)

            ax.set_title(f"{METHOD_LABELS_SHORT.get(method, method)}\nAcc: {acc:.1f}%, Rec: {rec:.1f}%",
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Predicted', fontsize=10, fontweight='bold')
            ax.set_ylabel('Actual', fontsize=10, fontweight='bold')

    plt.suptitle(f'Confusion Matrices - All Methods\n({TEST_SET_LABELS_SHORT[test_set]} Test Set)',
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    save_figure(fig, 'Figure_A2_Confusion_Matrix_All9')

    print(f"\n{'='*80}")
    print(f"✅ ALL 8 FIGURES GENERATED AND DISPLAYED ABOVE")
    print(f"✅ Figures saved to: {FIGURES_DIR}/")
    print(f"{'='*80}")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""

    print("\n" + "="*80)
    print("   COMPLETE CONTINUAL LEARNING EXPERIMENTS")
    print("   LoRA Ablation First → Optimal Rank Selection → All Methods")
    print("   WITH ALL TABLES AND FIGURES")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"GPU: {get_gpu_memory()}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    create_directories()

    print("\n📥 Loading tokenizer...")
    tokenizer = load_local_tokenizer(MODEL_PATH)
    print("✓ Tokenizer loaded")

    original_df, new_df, test_datasets, test_dfs = load_all_data(tokenizer)

    # Phase 1
    original_results, original_predictions, original_f1_historical, original_f1_contemporary = \
        run_phase1_original_model(test_datasets, device)

    # Phase 2: LoRA Ablation FIRST
    ablation_results, ablation_times, ablation_losses, ablation_predictions, ablation_bwt_fwt, optimal_lora_rank = \
        run_phase2_lora_ablation(original_df, new_df, test_datasets, tokenizer,
                                original_f1_historical, original_f1_contemporary, device)

    # Phase 3: Main Methods
    main_results, main_times, main_losses, main_predictions, main_params = \
        run_phase3_main_methods(original_df, new_df, test_datasets, tokenizer,
                               original_f1_historical, original_f1_contemporary,
                               optimal_lora_rank, device)

    # Phase 4: Hybrid Methods
    hybrid_results, hybrid_times, hybrid_losses, hybrid_predictions, hybrid_params = \
        run_phase4_hybrid_methods(original_df, new_df, test_datasets, tokenizer,
                                 original_f1_historical, original_f1_contemporary,
                                 optimal_lora_rank, device)

    # Generate Tables
    generate_all_tables(original_results, main_results, hybrid_results,
                       main_times, hybrid_times, ablation_bwt_fwt,
                       original_f1_historical, original_f1_contemporary,
                       optimal_lora_rank)

    # Generate Figures
    generate_all_figures(original_results, main_results, hybrid_results,
                        main_times, hybrid_times, main_losses, hybrid_losses,
                        main_predictions, hybrid_predictions,
                        ablation_results, ablation_bwt_fwt,
                        original_f1_historical, original_f1_contemporary,
                        optimal_lora_rank, original_predictions)

    # Save final results
    final_results = {
        'config': {
            'optimal_lora_rank': optimal_lora_rank,
            'selection_criterion': LORA_SELECTION_CRITERION,
            'seeds': SEEDS,
            'lora_ranks_tested': LORA_RANKS_TO_TEST,
        },
        'original_f1_historical': original_f1_historical,
        'original_f1_contemporary': original_f1_contemporary,
        'timestamp': datetime.now().isoformat(),
    }

    with open(f'{OUTPUT_DIR}/all_results_final.json', 'w') as f:
        json.dump(final_results, f, indent=2)

    # ========================================
    # PRINT COMPREHENSIVE FINAL SUMMARY
    # ========================================
    print("\n" + "="*80)
    print("   ✅ ALL EXPERIMENTS COMPLETED SUCCESSFULLY!")
    print("="*80)

    print(f"\n⏱️ Finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    print(f"\n{'='*80}")
    print("🏆 FINAL RESULTS SUMMARY")
    print(f"{'='*80}")

    print(f"\n🔧 Configuration:")
    print(f"   • Optimal LoRA Rank: r={optimal_lora_rank}")
    print(f"   • Selection Criterion: {LORA_SELECTION_CRITERION}")
    print(f"   • Seeds Used: {SEEDS}")
    print(f"   • LoRA Ranks Tested: {LORA_RANKS_TO_TEST}")

    print(f"\n📊 Original Model Baseline:")
    print(f"   • Historical F1: {original_f1_historical:.4f}")
    print(f"   • Contemporary F1: {original_f1_contemporary:.4f}")

    # Combine all results for final summary
    all_final_results = {'original': [original_results]}
    all_final_results.update(main_results)
    all_final_results.update(hybrid_results)

    print(f"\n{'='*80}")
    print("📋 FINAL PERFORMANCE TABLE (F1-Macro)")
    print(f"{'='*80}")
    print(f"\n{'Method':<15} | {'Historical':<12} | {'Contemporary':<12} | {'BWT':<12} | {'FWT':<12}")
    print(f"{'-'*70}")

    for method in METHODS_ORDER:
        if method in all_final_results:
            results_list = all_final_results[method]
            if len(results_list) > 1:
                hist_f1 = np.mean([r['unused_old']['f1_macro'] for r in results_list])
                cont_f1 = np.mean([r['balanced_500']['f1_macro'] for r in results_list])
                bwt = hist_f1 - original_f1_historical
                fwt = cont_f1 - original_f1_contemporary
            else:
                hist_f1 = results_list[0]['unused_old']['f1_macro']
                cont_f1 = results_list[0]['balanced_500']['f1_macro']
                bwt = hist_f1 - original_f1_historical
                fwt = cont_f1 - original_f1_contemporary

            if method == 'original':
                print(f"{METHOD_LABELS_SHORT.get(method, method):<15} | {hist_f1:<12.4f} | {cont_f1:<12.4f} | {'--':<12} | {'--':<12}")
            else:
                print(f"{METHOD_LABELS_SHORT.get(method, method):<15} | {hist_f1:<12.4f} | {cont_f1:<12.4f} | {bwt:<+12.4f} | {fwt:<+12.4f}")

    # Find best methods
    print(f"\n{'='*80}")
    print("🥇 BEST METHODS BY METRIC")
    print(f"{'='*80}")

    best_bwt_method = None
    best_bwt = float('-inf')
    best_fwt_method = None
    best_fwt = float('-inf')
    best_balance_method = None
    best_balance = float('-inf')

    for method in METHODS_ORDER:
        if method == 'original' or method not in all_final_results:
            continue
        results_list = all_final_results[method]
        if len(results_list) > 1:
            hist_f1 = np.mean([r['unused_old']['f1_macro'] for r in results_list])
            cont_f1 = np.mean([r['balanced_500']['f1_macro'] for r in results_list])
        else:
            hist_f1 = results_list[0]['unused_old']['f1_macro']
            cont_f1 = results_list[0]['balanced_500']['f1_macro']

        bwt = hist_f1 - original_f1_historical
        fwt = cont_f1 - original_f1_contemporary
        balance = bwt + fwt

        if bwt > best_bwt:
            best_bwt = bwt
            best_bwt_method = method
        if fwt > best_fwt:
            best_fwt = fwt
            best_fwt_method = method
        if balance > best_balance:
            best_balance = balance
            best_balance_method = method

    print(f"\n   🏆 Best Knowledge Retention (BWT): {METHOD_LABELS_SHORT.get(best_bwt_method, best_bwt_method)} ({best_bwt:+.4f})")
    print(f"   🏆 Best Adaptation (FWT): {METHOD_LABELS_SHORT.get(best_fwt_method, best_fwt_method)} ({best_fwt:+.4f})")
    print(f"   🏆 Best Balance (BWT+FWT): {METHOD_LABELS_SHORT.get(best_balance_method, best_balance_method)} ({best_balance:+.4f})")

    print(f"\n{'='*80}")
    print("📁 OUTPUT FILES")
    print(f"{'='*80}")
    print(f"\n   📊 Figures (8): {FIGURES_DIR}/")
    print(f"      • Figure_1_F1_Comparison.png/.pdf")
    print(f"      • Figure_2_BWT_FWT.png/.pdf")
    print(f"      • Figure_3_Parameter_Efficiency.png/.pdf")
    print(f"      • Figure_4_Training_Time.png/.pdf")
    print(f"      • Figure_5_Loss_Curves.png/.pdf")
    print(f"      • Figure_6_LoRA_Ablation.png/.pdf")
    print(f"      • Figure_A1_Confusion_Matrix_9x4.png/.pdf")
    print(f"      • Figure_A2_Confusion_Matrix_All9.png/.pdf")
    print(f"\n   📋 Tables (6): {TABLES_DIR}/")
    print(f"      • table1_f1_macro.csv")
    print(f"      • table2_accuracy.csv")
    print(f"      • table3_bwt_fwt.csv")
    print(f"      • table4_per_class_f1.csv")
    print(f"      • table5_training_time.csv")
    print(f"      • table6_lora_ablation.csv")
    print(f"\n   📄 JSON Results: {OUTPUT_DIR}/")
    print(f"      • all_results_final.json")
    print(f"      • lora_ablation_summary.json")

    print(f"\n{'='*80}")
    print("   🎉 EXPERIMENT COMPLETE! 🎉")
    print(f"{'='*80}\n")

    return {
        'optimal_rank': optimal_lora_rank,
        'original_results': original_results,
        'main_results': main_results,
        'hybrid_results': hybrid_results,
        'ablation_bwt_fwt': ablation_bwt_fwt,
    }

if __name__ == "__main__":
    results = main()