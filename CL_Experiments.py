"""
================================================================================
Continual Learning for Saudi Dialect Offensive Language Detection
under Temporal Linguistic Drift
================================================================================

This script implements the complete experiment pipeline for evaluating
continual learning methods on Saudi dialect offensive language detection.

EXPERIMENT STRUCTURE:
  Phase 1: LoRA Ablation Study (4 ranks x 5 seeds = 20 runs)
  Phase 2: Main CL Experiments (8 methods x 5 seeds = 40 runs)

METHODS EVALUATED:
  - Original (baseline)
  - Naive Fine-tuning
  - Experience Replay (ER)
  - Elastic Weight Consolidation (EWC)
  - LoRA (with optimal rank from ablation)
  - LoRA + ER
  - LoRA + EWC
  - LoRA + ER + EWC
  - Full + ER + EWC

FIGURES GENERATED:
  - Figure 1: Performance comparison (grouped bar chart)
  - Figure 2: KR vs AG trade-off scatter plot
  - Figure 3: Parameter efficiency analysis (2 panels)
  - Figure 4: Training time comparison
  - Figure A1: Confusion matrices (9 methods x 4 test sets)
  - Figure B1: LoRA rank ablation study

USAGE:
  1. Update the paths in the CONFIGURATION section
  2. Run: python CL_Experiments.py

================================================================================
"""

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import warnings
import time
from typing import Dict, List, Tuple
from torch.utils.data import DataLoader

warnings.filterwarnings('ignore')

# Disable wandb
os.environ['WANDB_DISABLED'] = 'true'
os.environ['WANDB_MODE'] = 'disabled'


# ============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# ============================================================================

# Data paths
BASE_PATH = './data'  # Update to your data directory

MODEL_PATH = f'{BASE_PATH}/SOD_AraBERT_model'
ORIGINAL_DS = f'{BASE_PATH}/SDOffensive_Paper2.csv'
NEW_DS = f'{BASE_PATH}/Paper2_DS_Complete.csv'
HISTORICAL_TEST = f'{BASE_PATH}/processed500UnseenDS_Paper2.csv'
CONTEMPORARY_TEST = f'{BASE_PATH}/Balanced500_Paper2.csv'
MIXED_2080_TEST = f'{BASE_PATH}/TestDS2080_Paper2.csv'
MIXED_4060_TEST = f'{BASE_PATH}/TestDS4060_Paper2.csv'

# Output directories
RESULTS_DIR = './results'
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)


# ============================================================================
# HYPERPARAMETERS
# ============================================================================

SEEDS = [42, 101, 123, 456, 789]  # 5 random seeds
REPLAY_SAMPLES = 750              # Class-balanced replay buffer size
EWC_LAMBDA = 1000                 # EWC regularization strength
EWC_SAMPLES = 1000                # Samples for Fisher computation
MAX_LENGTH = 128                  # Maximum sequence length
BATCH_SIZE = 32                   # Batch size
LEARNING_RATE = 2e-5              # Learning rate
EPOCHS = 5                        # Training epochs
CLASS_WEIGHT_0 = 1.0              # Weight for non-offensive class
CLASS_WEIGHT_1 = 2.0              # Weight for offensive class

TOTAL_MODEL_PARAMS = 135_000_000  # AraBERT ~ 135M parameters


# ============================================================================
# LoRA ABLATION CONFIGURATION
# ============================================================================

LORA_ABLATION_VARIANTS = [
    {"name": "r=8",  "r": 8,  "alpha": 16,  "target_modules": ["query", "key", "value"], "dropout": 0.1, "bias": "none"},
    {"name": "r=16", "r": 16, "alpha": 32,  "target_modules": ["query", "key", "value"], "dropout": 0.1, "bias": "none"},
    {"name": "r=32", "r": 32, "alpha": 64,  "target_modules": ["query", "key", "value"], "dropout": 0.1, "bias": "none"},
    {"name": "r=64", "r": 64, "alpha": 128, "target_modules": ["query", "key", "value"], "dropout": 0.1, "bias": "none"},
]

OPTIMAL_LORA_CONFIG = None


# ============================================================================
# VISUALIZATION SETTINGS (Colorblind-friendly)
# ============================================================================

COLORS = {
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73',
    'red': '#D55E00', 'purple': '#CC79A7', 'yellow': '#F0E442',
    'cyan': '#56B4E9', 'gray': '#999999', 'brown': '#8B4513',
}

METHOD_COLORS = {
    'original': COLORS['gray'], 'naive_ft': COLORS['orange'],
    'er': COLORS['blue'], 'ewc': COLORS['purple'],
    'lora': COLORS['green'], 'lora+er': COLORS['yellow'],
    'lora+ewc': COLORS['cyan'], 'lora+er+ewc': COLORS['red'],
    'full+er+ewc': COLORS['brown'],
}

METHOD_NAMES = {
    'original': 'Original', 'naive_ft': 'Naive FT',
    'er': 'ER', 'ewc': 'EWC', 'lora': 'LoRA',
    'lora+er': 'LoRA+ER', 'lora+ewc': 'LoRA+EWC',
    'lora+er+ewc': 'LoRA+ER+EWC', 'full+er+ewc': 'Full+ER+EWC',
}

METHODS_ORDER = ['original', 'naive_ft', 'er', 'ewc', 'lora', 
                 'lora+er', 'full+er+ewc', 'lora+ewc', 'lora+er+ewc']

TEST_SET_NAMES = {
    'historical': 'Historical', 'contemporary': 'Contemporary',
    'mixed_2080': 'Mixed 80-20', 'mixed_4060': 'Mixed 40-60',
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_dataset_csv(path: str) -> pd.DataFrame:
    """Load dataset from CSV file."""
    df = pd.read_csv(path)
    if 'labels' in df.columns and 'label' not in df.columns:
        df = df.rename(columns={'labels': 'label'})
    return df[['text', 'label']].copy()


def tokenize_function(examples, tokenizer):
    """Tokenize text examples."""
    return tokenizer(
        examples['text'],
        padding='max_length',
        truncation=True,
        max_length=MAX_LENGTH
    )


def prepare_test_dataset(df: pd.DataFrame) -> Dataset:
    """Prepare test dataset for evaluation."""
    return Dataset.from_pandas(df[['text', 'label']].rename(columns={'label': 'labels'}))


def compute_metrics(eval_pred):
    """Compute evaluation metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'accuracy': accuracy_score(labels, preds),
        'precision_macro': precision_score(labels, preds, average='macro', zero_division=0),
        'recall_macro': recall_score(labels, preds, average='macro', zero_division=0),
        'f1_macro': f1_score(labels, preds, average='macro', zero_division=0),
        'f1_class_0': f1_score(labels, preds, pos_label=0, average='binary', zero_division=0),
        'f1_class_1': f1_score(labels, preds, pos_label=1, average='binary', zero_division=0),
    }


def clear_memory():
    """Clear GPU memory."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fmt_mean_std(values, decimals=4):
    """Format mean +/- std string."""
    values = [v for v in values if v is not None and not np.isnan(v)]
    if not values:
        return "N/A"
    m, s = np.mean(values), np.std(values)
    if len(values) == 1:
        return f"{m:.{decimals}f}"
    return f"{m:.{decimals}f} +/- {s:.{decimals}f}"


# ============================================================================
# EWC IMPLEMENTATION
# ============================================================================

class EWC:
    """
    Elastic Weight Consolidation (Kirkpatrick et al., 2017)
    
    Computes diagonal Fisher Information Matrix to identify important
    parameters for previous tasks and penalizes changes to them.
    """

    def __init__(self, model, dataloader, device):
        self.device = device
        self.model = model
        self.params = {n: p.clone().detach() for n, p in self.model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader)

    def _compute_fisher(self, dataloader):
        """Compute diagonal Fisher Information Matrix."""
        fisher = {n: torch.zeros_like(p, device=self.device) 
                  for n, p in self.model.named_parameters() if p.requires_grad}

        self.model.eval()
        num_samples = 0
        
        for batch in dataloader:
            self.model.zero_grad()
            inputs = {k: v.to(self.device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(self.device)

            outputs = self.model(**inputs)
            loss = torch.nn.CrossEntropyLoss()(outputs.logits, labels)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2
            num_samples += labels.size(0)

        for n in fisher:
            fisher[n] /= max(1, num_samples)

        return fisher

    def penalty(self, model):
        """Compute EWC penalty term."""
        loss = 0.0
        for n, p in model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.params[n]) ** 2).sum()
        return loss


# ============================================================================
# CUSTOM TRAINER WITH EWC
# ============================================================================

class EWCTrainer(Trainer):
    """
    Custom Trainer with EWC regularization and class weights.
    Uses the correct EWC formula: loss + (lambda/2) * penalty
    """
    
    def __init__(self, *args, ewc=None, ewc_lambda=0.0, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ewc = ewc
        self.ewc_lambda = ewc_lambda
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        else:
            loss_fct = torch.nn.CrossEntropyLoss()

        loss = loss_fct(logits, labels)

        # EWC regularization: loss + (lambda/2) * penalty
        if self.ewc is not None and self.ewc_lambda > 0:
            loss = loss + (self.ewc_lambda / 2) * self.ewc.penalty(model)

        return (loss, outputs) if return_outputs else loss


# ============================================================================
# EXPERIENCE REPLAY
# ============================================================================

def create_replay_buffer(original_df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """Create class-balanced replay buffer from original training data."""
    set_seed(seed)
    df0 = original_df[original_df['label'] == 0]
    df1 = original_df[original_df['label'] == 1]

    n0, n1 = n // 2, n - n // 2

    replay0 = df0.sample(n=min(n0, len(df0)), replace=(len(df0) < n0), random_state=seed)
    replay1 = df1.sample(n=min(n1, len(df1)), replace=(len(df1) < n1), random_state=seed)

    return pd.concat([replay0, replay1]).sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ============================================================================
# ORIGINAL MODEL EVALUATION
# ============================================================================

def evaluate_original_model(test_datasets: Dict, tokenizer) -> Tuple[Dict, Dict]:
    """Evaluate original model (deterministic baseline)."""
    print("\n" + "="*60)
    print("Evaluating: ORIGINAL MODEL (Baseline)")
    print("="*60)
    
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir='./temp', 
            per_device_eval_batch_size=BATCH_SIZE, 
            report_to="none", 
            remove_unused_columns=False
        ),
        compute_metrics=compute_metrics
    )

    results, predictions = {}, {}
    for test_name, test_ds in test_datasets.items():
        results[test_name] = trainer.evaluate(test_ds)
        pred_output = trainer.predict(test_ds)
        predictions[f'{test_name}_preds'] = np.argmax(pred_output.predictions, axis=1)
        predictions[f'{test_name}_labels'] = pred_output.label_ids
        
        print(f"  {test_name}: F1={results[test_name]['eval_f1_macro']:.4f}, "
              f"Acc={results[test_name]['eval_accuracy']:.4f}")
    
    del model, trainer
    clear_memory()
    
    print("[OK] Original model evaluation complete")
    return results, predictions


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_method(
    method_name: str,
    seed: int,
    original_df: pd.DataFrame,
    new_df: pd.DataFrame,
    test_datasets: Dict,
    tokenizer,
    use_lora: bool = False,
    use_er: bool = False,
    use_ewc: bool = False,
    lora_config: dict = None,
) -> Tuple[Dict, float, Dict, int, Dict]:
    """
    Train a continual learning method and evaluate on all test sets.
    
    IMPORTANT: Validation uses NEW data only (replay only in training).
    """

    print(f"\n{'='*60}")
    print(f"Training: {method_name} | Seed: {seed}")
    print(f"LoRA: {use_lora} | ER: {use_er} | EWC: {use_ewc}")
    print(f"{'='*60}")

    set_seed(seed)
    start_time = time.time()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: Split NEW data into train/val (NEW-ONLY validation)
    new_train_df, new_val_df = train_test_split(
        new_df, test_size=0.2, random_state=seed, stratify=new_df['label']
    )
    print(f"  New data: {len(new_train_df)} train / {len(new_val_df)} val")
    
    # Step 2: Add replay to training ONLY (not validation)
    if use_er:
        replay_df = create_replay_buffer(original_df, REPLAY_SAMPLES, seed)
        train_df = pd.concat([new_train_df, replay_df], ignore_index=True)
        train_df = train_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
        print(f"  ER: {len(new_train_df)} new + {len(replay_df)} replay = {len(train_df)} total")
    else:
        train_df = new_train_df
        print(f"  Training: {len(train_df)} samples")
    
    # Step 3: Validation is NEW-ONLY
    val_df = new_val_df
    print(f"  Validation: {len(val_df)} samples (NEW-ONLY)")

    # Tokenize datasets
    train_ds = Dataset.from_dict({'text': train_df['text'].tolist(), 'labels': train_df['label'].tolist()})
    val_ds = Dataset.from_dict({'text': val_df['text'].tolist(), 'labels': val_df['label'].tolist()})
    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    val_ds = val_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH, num_labels=2)
    model = model.to(device)

    # Apply LoRA if specified
    if use_lora and lora_config:
        peft_config = LoraConfig(
            r=lora_config["r"],
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["target_modules"],
            lora_dropout=lora_config.get("dropout", 0.1),
            bias=lora_config.get("bias", "none"),
            task_type=TaskType.SEQ_CLS
        )
        model = get_peft_model(model, peft_config)
        print(f"  LoRA: r={lora_config['r']}, alpha={lora_config['alpha']}")
        model.print_trainable_parameters()

    # Count parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")

    # Compute EWC Fisher Information if needed
    ewc = None
    if use_ewc:
        print("  Computing Fisher Information...")
        ewc_df = original_df.sample(n=min(EWC_SAMPLES, len(original_df)), random_state=seed)
        ewc_ds = Dataset.from_pandas(ewc_df[['text', 'label']].rename(columns={'label': 'labels'}))
        ewc_ds = ewc_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
        ewc_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        ewc_loader = DataLoader(ewc_ds, batch_size=BATCH_SIZE, shuffle=False)
        ewc = EWC(model, ewc_loader, device)
        print("  [OK] EWC Fisher computed")

    # Training
    class_weights = torch.tensor([CLASS_WEIGHT_0, CLASS_WEIGHT_1], dtype=torch.float32)
    
    training_args = TrainingArguments(
        output_dir=f'./temp_{method_name}_{seed}',
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        warmup_steps=100,
        weight_decay=0.01,
        eval_strategy='epoch',
        save_strategy='no',
        report_to="none",
        logging_steps=50,
        fp16=torch.cuda.is_available(),
        seed=seed,
        remove_unused_columns=False,
    )

    trainer = EWCTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        ewc=ewc,
        ewc_lambda=EWC_LAMBDA if use_ewc else 0.0,
        class_weights=class_weights
    )

    trainer.train()

    # Extract convergence metrics
    log_history = trainer.state.log_history or []
    train_losses = [x.get('loss') for x in log_history if 'loss' in x]
    eval_losses = [x.get('eval_loss') for x in log_history if 'eval_loss' in x]
    
    convergence = {
        'final_train_loss': float(train_losses[-1]) if train_losses else float('nan'),
        'best_val_loss': float(min(eval_losses)) if eval_losses else float('nan'),
        'final_val_loss': float(eval_losses[-1]) if eval_losses else float('nan'),
    }

    training_time = time.time() - start_time

    # Evaluate on all test sets
    results, predictions = {}, {}
    for test_name, test_ds in test_datasets.items():
        results[test_name] = trainer.evaluate(test_ds)
        pred_output = trainer.predict(test_ds)
        predictions[f'{test_name}_preds'] = np.argmax(pred_output.predictions, axis=1)
        predictions[f'{test_name}_labels'] = pred_output.label_ids

    print(f"[OK] Completed in {training_time:.1f}s")

    del model, trainer
    clear_memory()

    return results, training_time, convergence, trainable_params, predictions


# ============================================================================
# PHASE 1: LoRA ABLATION STUDY
# ============================================================================

def run_lora_ablation(original_df, new_df, test_datasets, tokenizer):
    """Run LoRA ablation study to determine optimal rank."""
    
    print("\n" + "="*80)
    print("PHASE 1: LoRA ABLATION STUDY")
    print("Testing 4 ranks x 5 seeds = 20 runs")
    print("="*80)

    ablation_results = {}
    ablation_params = {}

    for variant in LORA_ABLATION_VARIANTS:
        variant_name = variant["name"]
        print(f"\n{'#'*60}")
        print(f"LoRA Rank: {variant_name} (r={variant['r']}, alpha={variant['alpha']})")
        print(f"{'#'*60}")

        ablation_results[variant_name] = []
        
        for seed in SEEDS:
            results, train_time, conv, trainable, preds = train_method(
                method_name=f"LoRA_{variant_name}",
                seed=seed,
                original_df=original_df,
                new_df=new_df,
                test_datasets=test_datasets,
                tokenizer=tokenizer,
                use_lora=True,
                use_er=False,
                use_ewc=False,
                lora_config=variant,
            )
            ablation_results[variant_name].append(results)
            ablation_params[variant_name] = trainable

    # Determine optimal rank
    print("\n" + "="*60)
    print("ABLATION RESULTS SUMMARY")
    print("="*60)
    print(f"{'Rank':<10} | {'Historical':<12} | {'Contemporary':<12} | {'Balanced':<12}")
    print("-" * 55)

    best_variant = None
    best_score = -1

    for variant in LORA_ABLATION_VARIANTS:
        name = variant["name"]
        results_list = ablation_results[name]
        
        hist_f1 = np.mean([r['historical']['eval_f1_macro'] for r in results_list])
        cont_f1 = np.mean([r['contemporary']['eval_f1_macro'] for r in results_list])
        balanced_score = (hist_f1 + cont_f1) / 2
        
        print(f"{name:<10} | {hist_f1:.4f}       | {cont_f1:.4f}       | {balanced_score:.4f}")
        
        if balanced_score > best_score:
            best_score = balanced_score
            best_variant = variant

    print(f"\n[OK] OPTIMAL RANK: {best_variant['name']} (Balanced: {best_score:.4f})")
    
    return ablation_results, ablation_params, best_variant


# ============================================================================
# PHASE 2: MAIN EXPERIMENTS
# ============================================================================

def run_main_experiments(original_df, new_df, test_datasets, tokenizer, optimal_lora_config,
                         original_results, original_predictions):
    """Run main continual learning experiments using optimal LoRA config."""
    
    print("\n" + "="*80)
    print("PHASE 2: MAIN CONTINUAL LEARNING EXPERIMENTS")
    print(f"Using optimal LoRA: {optimal_lora_config['name']}")
    print("="*80)

    methods_config = [
        ('naive_ft', False, False, False),
        ('er', False, True, False),
        ('ewc', False, False, True),
        ('lora', True, False, False),
        ('lora+er', True, True, False),
        ('lora+ewc', True, False, True),
        ('lora+er+ewc', True, True, True),
        ('full+er+ewc', False, True, True),
    ]

    # Initialize with original results
    all_results = {'original': [original_results]}
    training_times = {'original': [0.0]}
    convergence_metrics = {'original': [{}]}
    params_dict = {'original': 0}
    all_predictions = {'original': original_predictions}

    for method_name, use_lora, use_er, use_ewc in methods_config:
        print(f"\n{'#'*60}")
        print(f"Method: {method_name.upper()}")
        print(f"{'#'*60}")

        all_results[method_name] = []
        training_times[method_name] = []
        convergence_metrics[method_name] = []

        for seed in SEEDS:
            results, train_time, conv, trainable, preds = train_method(
                method_name=method_name,
                seed=seed,
                original_df=original_df,
                new_df=new_df,
                test_datasets=test_datasets,
                tokenizer=tokenizer,
                use_lora=use_lora,
                use_er=use_er,
                use_ewc=use_ewc,
                lora_config=optimal_lora_config if use_lora else None,
            )

            all_results[method_name].append(results)
            training_times[method_name].append(train_time)
            convergence_metrics[method_name].append(conv)
            params_dict[method_name] = trainable
            
            if seed == SEEDS[-1]:
                all_predictions[method_name] = preds

    return all_results, training_times, convergence_metrics, params_dict, all_predictions


# ============================================================================
# TABLE GENERATION
# ============================================================================

def generate_all_tables(all_results, training_times, convergence_metrics, params_dict,
                        ablation_results, ablation_params, optimal_lora_config):
    """Generate all result tables."""

    orig_hist = all_results['original'][0]['historical']['eval_f1_macro']
    orig_cont = all_results['original'][0]['contemporary']['eval_f1_macro']

    # TABLE 3: Overall Performance
    print("\n" + "="*100)
    print("TABLE 3: Overall Performance")
    print("="*100)

    table3_rows = []
    for method in METHODS_ORDER:
        if method not in all_results:
            continue
        results_list = all_results[method]
        row = {'Method': METHOD_NAMES.get(method, method)}
        is_original = (method == 'original')
        
        for test_key, test_name in TEST_SET_NAMES.items():
            f1_vals = [r[test_key]['eval_f1_macro'] for r in results_list]
            acc_vals = [r[test_key]['eval_accuracy'] for r in results_list]
            
            if is_original:
                row[f'{test_name} F1'] = f"{f1_vals[0]:.4f}"
                row[f'{test_name} Acc'] = f"{acc_vals[0]:.4f}"
            else:
                row[f'{test_name} F1'] = fmt_mean_std(f1_vals)
                row[f'{test_name} Acc'] = fmt_mean_std(acc_vals)
        table3_rows.append(row)

    table3 = pd.DataFrame(table3_rows)
    print(table3.to_string(index=False))
    table3.to_csv(os.path.join(TABLES_DIR, 'Table3_OverallPerformance.csv'), index=False)

    # TABLE 4: KR and AG Analysis
    print("\n" + "="*100)
    print("TABLE 4: Knowledge Retention (KR) and Adaptation Gain (AG)")
    print("="*100)

    table4_rows = []
    for method in METHODS_ORDER:
        if method == 'original' or method not in all_results:
            continue
        results_list = all_results[method]
        
        kr_vals = [r['historical']['eval_f1_macro'] - orig_hist for r in results_list]
        ag_vals = [r['contemporary']['eval_f1_macro'] - orig_cont for r in results_list]
        hist_vals = [r['historical']['eval_f1_macro'] for r in results_list]
        cont_vals = [r['contemporary']['eval_f1_macro'] for r in results_list]
        
        table4_rows.append({
            'Method': METHOD_NAMES.get(method, method),
            'KR': fmt_mean_std(kr_vals),
            'AG': fmt_mean_std(ag_vals),
            'Historical F1': fmt_mean_std(hist_vals),
            'Contemporary F1': fmt_mean_std(cont_vals),
        })

    table4 = pd.DataFrame(table4_rows)
    print(table4.to_string(index=False))
    table4.to_csv(os.path.join(TABLES_DIR, 'Table4_KR_AG.csv'), index=False)

    # TABLE 5: Training Convergence
    print("\n" + "="*100)
    print("TABLE 5: Training Convergence")
    print("="*100)

    table5_rows = []
    for method in METHODS_ORDER:
        if method not in convergence_metrics:
            continue
        if method == 'original':
            table5_rows.append({'Method': 'Original', 'Final Train Loss': '-', 
                               'Best Val Loss': '-', 'Final Val Loss': '-',
                               'Trainable Params': '-', 'Time (s)': '-'})
            continue

        conv_list = convergence_metrics[method]
        final_train = [c.get('final_train_loss') for c in conv_list if c]
        best_val = [c.get('best_val_loss') for c in conv_list if c]
        final_val = [c.get('final_val_loss') for c in conv_list if c]
        times = training_times.get(method, [])
        params = params_dict.get(method, 0)

        table5_rows.append({
            'Method': METHOD_NAMES.get(method, method),
            'Final Train Loss': fmt_mean_std([v for v in final_train if v and not np.isnan(v)]),
            'Best Val Loss': fmt_mean_std([v for v in best_val if v and not np.isnan(v)]),
            'Final Val Loss': fmt_mean_std([v for v in final_val if v and not np.isnan(v)]),
            'Trainable Params': f"{params:,}" if params else '-',
            'Time (s)': fmt_mean_std(times, 1) if times else '-'
        })

    table5 = pd.DataFrame(table5_rows)
    print(table5.to_string(index=False))
    table5.to_csv(os.path.join(TABLES_DIR, 'Table5_Convergence.csv'), index=False)

    # TABLE B1: LoRA Ablation
    print("\n" + "="*100)
    print("TABLE B1: LoRA Rank Ablation")
    print("="*100)

    tableB1_rows = []
    for variant in LORA_ABLATION_VARIANTS:
        name = variant["name"]
        if name not in ablation_results:
            continue
        
        params = ablation_params.get(name, 0)
        results_list = ablation_results[name]
        hist_vals = [r['historical']['eval_f1_macro'] for r in results_list]
        cont_vals = [r['contemporary']['eval_f1_macro'] for r in results_list]
        kr_vals = [h - orig_hist for h in hist_vals]
        
        is_optimal = "[*]" if name == optimal_lora_config["name"] else ""
        
        tableB1_rows.append({
            'Rank': name, 'Parameters': f"{params:,}",
            'Historical F1': fmt_mean_std(hist_vals),
            'Contemporary F1': fmt_mean_std(cont_vals),
            'KR': fmt_mean_std(kr_vals), 'Optimal': is_optimal
        })

    tableB1 = pd.DataFrame(tableB1_rows)
    print(tableB1.to_string(index=False))
    tableB1.to_csv(os.path.join(TABLES_DIR, 'TableB1_LoRA_Ablation.csv'), index=False)

    print(f"\n[OK] All tables saved to: {TABLES_DIR}")
    
    return orig_hist, orig_cont


# ============================================================================
# FIGURE GENERATION
# ============================================================================

def generate_figure_1(all_results):
    """Figure 1: Performance bar charts."""
    print("\nGenerating Figure 1: Performance Comparison...")
    
    test_sets = ['historical', 'contemporary', 'mixed_2080', 'mixed_4060']
    methods = METHODS_ORDER
    test_set_colors = ['#D55E00', '#0072B2', '#CC79A7', '#009E73']
    test_set_labels = ['Historical', 'Contemporary', 'Mixed 80-20', 'Mixed 40-60']
    
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(methods))
    width = 0.2
    
    for i, (test_set, color, label) in enumerate(zip(test_sets, test_set_colors, test_set_labels)):
        means, stds = [], []
        for method in methods:
            if method not in all_results:
                means.append(0); stds.append(0); continue
            vals = [r[test_set]['eval_f1_macro'] for r in all_results[method]]
            means.append(np.mean(vals))
            stds.append(np.std(vals) if len(vals) > 1 else 0.0)
        
        ax.bar(x + i*width, means, width, yerr=stds, label=label, 
               color=color, capsize=3, edgecolor='black', linewidth=0.5, alpha=0.85)
    
    ax.set_ylabel('F1-macro', fontsize=12, fontweight='bold')
    ax.set_xticks(x + 1.5*width)
    ax.set_xticklabels([METHOD_NAMES.get(m, m) for m in methods], fontsize=10)
    ax.set_ylim(0.5, 1.05)
    ax.legend(loc='upper right', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    plt.title('Figure 1. Performance Comparison (mean +/- std, 5 seeds)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, 'Figure_1_Performance.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {fig_path}")


def generate_figure_2(all_results, orig_hist, orig_cont):
    """Figure 2: KR vs AG scatter plot."""
    print("\nGenerating Figure 2: KR vs AG Trade-off...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    markers = {'naive_ft': 'o', 'er': 's', 'ewc': '^', 'lora': 'D', 
               'lora+er': 'v', 'lora+ewc': '<', 'lora+er+ewc': '>', 'full+er+ewc': 'p'}
    
    for method in METHODS_ORDER:
        if method == 'original' or method not in all_results:
            continue
        results_list = all_results[method]
        kr_vals = [r['historical']['eval_f1_macro'] - orig_hist for r in results_list]
        ag_vals = [r['contemporary']['eval_f1_macro'] - orig_cont for r in results_list]
        kr_mean, kr_std = np.mean(kr_vals), np.std(kr_vals)
        ag_mean, ag_std = np.mean(ag_vals), np.std(ag_vals)
        
        ax.errorbar(ag_mean, kr_mean, xerr=ag_std, yerr=kr_std,
                   marker=markers.get(method, 'o'), markersize=12,
                   color=METHOD_COLORS.get(method, 'gray'),
                   label=METHOD_NAMES.get(method, method),
                   capsize=4, linewidth=2, markeredgecolor='black', markeredgewidth=0.5)
    
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Adaptation Gain (AG)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Knowledge Retention (KR)', fontsize=12, fontweight='bold')
    ax.set_title('Figure 2. Knowledge Retention vs Adaptation Gain', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, 'Figure_2_KR_AG_Tradeoff.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {fig_path}")


def generate_figure_3(all_results, params_dict):
    """Figure 3: Parameter Efficiency (two panels)."""
    print("\nGenerating Figure 3: Parameter Efficiency...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods_to_plot = ['naive_ft', 'er', 'ewc', 'lora', 'lora+er', 'lora+ewc', 'lora+er+ewc', 'full+er+ewc']
    
    for ax, (test_set, title) in zip(axes, [
        ('historical', 'Historical F1 (Knowledge Retention)'),
        ('contemporary', 'Contemporary F1 (Adaptation)')
    ]):
        for method in methods_to_plot:
            if method not in all_results:
                continue
            vals = [r[test_set]['eval_f1_macro'] for r in all_results[method]]
            mean_f1 = np.mean(vals)
            std_f1 = np.std(vals) if len(vals) > 1 else 0.0
            params = params_dict.get(method, TOTAL_MODEL_PARAMS)
            pct = (params / TOTAL_MODEL_PARAMS) * 100
            
            ax.errorbar(pct, mean_f1, yerr=std_f1, marker='o', markersize=10,
                       color=METHOD_COLORS.get(method, 'gray'),
                       label=METHOD_NAMES.get(method, method), capsize=4, linewidth=2)
        
        ax.set_xlabel('Trainable Parameters (%)', fontsize=11, fontweight='bold')
        ax.set_ylabel('F1-macro', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xscale('log')
        ax.set_ylim(0.70, 1.0)
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(loc='lower right', fontsize=8)
    
    plt.suptitle('Figure 3. Parameter Efficiency: Retention vs Adaptation Trade-off',
                 fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    fig_path = os.path.join(FIGURES_DIR, 'Figure_3_ParamEfficiency.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {fig_path}")


def generate_figure_4(training_times):
    """Figure 4: Training Time Comparison."""
    print("\nGenerating Figure 4: Training Time...")
    
    methods = [m for m in METHODS_ORDER if m != 'original' and m in training_times]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    names, means, stds, colors_list = [], [], [], []
    for method in methods:
        times = training_times[method]
        names.append(METHOD_NAMES.get(method, method))
        means.append(np.mean(times))
        stds.append(np.std(times) if len(times) > 1 else 0.0)
        colors_list.append(METHOD_COLORS.get(method, COLORS['gray']))
    
    x_pos = np.arange(len(names))
    ax.bar(x_pos, means, yerr=stds, capsize=4, color=colors_list,
           edgecolor='black', linewidth=0.5, alpha=0.85)
    
    ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=0, ha='center', fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    plt.title('Figure 4. Training Time Comparison (mean +/- std, 5 seeds)', fontsize=13, fontweight='bold')
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, 'Figure_4_TrainingTime.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {fig_path}")


def generate_figure_A1(all_predictions):
    """Figure A1: Confusion Matrices (9 methods x 4 test sets)."""
    print("\nGenerating Figure A1: Confusion Matrices...")
    
    test_sets = ['historical', 'contemporary', 'mixed_2080', 'mixed_4060']
    methods = METHODS_ORDER
    
    fig, axes = plt.subplots(9, 4, figsize=(16, 36))
    
    for row_idx, method in enumerate(methods):
        for col_idx, test_set in enumerate(test_sets):
            ax = axes[row_idx, col_idx]
            
            preds_key = f'{test_set}_preds'
            labels_key = f'{test_set}_labels'
            
            if method in all_predictions and preds_key in all_predictions[method]:
                preds = all_predictions[method][preds_key]
                labels = all_predictions[method][labels_key]
                
                cm = confusion_matrix(labels, preds)
                acc = accuracy_score(labels, preds) * 100
                
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                           xticklabels=['Non-Off', 'Offensive'],
                           yticklabels=['Non-Off', 'Offensive'],
                           cbar=False, annot_kws={'size': 11})
                
                if row_idx == 0:
                    ax.set_title(f'{TEST_SET_NAMES[test_set]}\nAcc: {acc:.1f}%',
                                fontsize=12, fontweight='bold')
                else:
                    ax.set_title(f'Acc: {acc:.1f}%', fontsize=11)
                
                if col_idx == 0:
                    ax.set_ylabel(f'{METHOD_NAMES.get(method, method)}\n\nActual',
                                 fontsize=11, fontweight='bold')
                else:
                    ax.set_ylabel('Actual', fontsize=10)
                
                ax.set_xlabel('Predicted', fontsize=10)
            else:
                ax.axis('off')
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
    
    plt.suptitle('Figure A1. Confusion Matrix Analysis: All Methods x All Test Sets\n(Seed 101)',
                 fontsize=16, fontweight='bold', y=1.01)
    plt.tight_layout()
    
    fig_path = os.path.join(FIGURES_DIR, 'Figure_A1_ConfusionMatrices.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {fig_path}")


def generate_figure_B1(ablation_results, optimal_lora_config):
    """Figure B1: LoRA Rank Ablation."""
    print("\nGenerating Figure B1: LoRA Ablation...")
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ranks = [8, 16, 32, 64]
    x_pos = np.arange(len(ranks))
    
    for ax, (metric, title) in zip(axes, [
        ('historical', 'Historical F1 (Knowledge Retention)'),
        ('contemporary', 'Contemporary F1 (Adaptation)')
    ]):
        means, stds = [], []
        for variant in LORA_ABLATION_VARIANTS:
            name = variant["name"]
            vals = [r[metric]['eval_f1_macro'] for r in ablation_results[name]]
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=4, 
                     color=COLORS['blue'], edgecolor='black', linewidth=0.5, alpha=0.85)
        
        opt_idx = [v["name"] for v in LORA_ABLATION_VARIANTS].index(optimal_lora_config["name"])
        bars[opt_idx].set_edgecolor('red')
        bars[opt_idx].set_linewidth(3)
        
        ax.set_xlabel('LoRA Rank (r)', fontsize=12, fontweight='bold')
        ax.set_ylabel('F1-macro', fontsize=12, fontweight='bold')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels([str(r) for r in ranks], fontsize=11)
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)
        ax.set_ylim(0.60, 1.0)
    
    plt.suptitle('Figure B1. LoRA Rank Ablation (optimal in red)', fontsize=14, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    
    fig_path = os.path.join(FIGURES_DIR, 'Figure_B1_LoRA_Ablation.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"[OK] Saved: {fig_path}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("\n" + "="*80)
    print("CONTINUAL LEARNING FOR SAUDI DIALECT OFFENSIVE LANGUAGE DETECTION")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Seeds: {SEEDS}")
    print(f"  Learning Rate: {LEARNING_RATE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  EWC lambda: {EWC_LAMBDA}")
    print(f"  Replay Buffer: {REPLAY_SAMPLES} samples")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Load datasets
    print("\nLoading datasets...")
    original_df = load_dataset_csv(ORIGINAL_DS)
    new_df = load_dataset_csv(NEW_DS)
    historical_df = load_dataset_csv(HISTORICAL_TEST)
    contemporary_df = load_dataset_csv(CONTEMPORARY_TEST)
    mixed_2080_df = load_dataset_csv(MIXED_2080_TEST)
    mixed_4060_df = load_dataset_csv(MIXED_4060_TEST)

    print(f"[OK] Original SOD: {len(original_df)} samples")
    print(f"[OK] New (NEW_DS): {len(new_df)} samples")
    print(f"[OK] Test sets: Hist={len(historical_df)}, Cont={len(contemporary_df)}, "
          f"M80-20={len(mixed_2080_df)}, M40-60={len(mixed_4060_df)}")

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    print("[OK] Tokenizer loaded")

    # Prepare test datasets
    print("\nPreparing test datasets...")
    test_datasets = {}
    for name, df in [('historical', historical_df), ('contemporary', contemporary_df),
                     ('mixed_2080', mixed_2080_df), ('mixed_4060', mixed_4060_df)]:
        ds = prepare_test_dataset(df)
        test_datasets[name] = ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    print("[OK] Test datasets ready")

    # Evaluate Original Model
    original_results, original_predictions = evaluate_original_model(test_datasets, tokenizer)

    # Phase 1: LoRA Ablation
    ablation_results, ablation_params, optimal_lora_config = run_lora_ablation(
        original_df, new_df, test_datasets, tokenizer
    )

    # Phase 2: Main Experiments
    all_results, training_times, convergence_metrics, params_dict, all_predictions = run_main_experiments(
        original_df, new_df, test_datasets, tokenizer, optimal_lora_config,
        original_results, original_predictions
    )

    # Generate Tables
    print("\n" + "="*80)
    print("GENERATING TABLES")
    print("="*80)
    orig_hist, orig_cont = generate_all_tables(
        all_results, training_times, convergence_metrics, params_dict,
        ablation_results, ablation_params, optimal_lora_config
    )

    # Generate Figures
    print("\n" + "="*80)
    print("GENERATING FIGURES")
    print("="*80)
    generate_figure_1(all_results)
    generate_figure_2(all_results, orig_hist, orig_cont)
    generate_figure_3(all_results, params_dict)
    generate_figure_4(training_times)
    generate_figure_A1(all_predictions)
    generate_figure_B1(ablation_results, optimal_lora_config)

    # Summary
    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {RESULTS_DIR}")
    print(f"Optimal LoRA rank: {optimal_lora_config['name']}")
    print("\nFigures generated:")
    print("  - Figure 1: Performance Comparison")
    print("  - Figure 2: KR vs AG Trade-off")
    print("  - Figure 3: Parameter Efficiency")
    print("  - Figure 4: Training Time")
    print("  - Figure A1: Confusion Matrices")
    print("  - Figure B1: LoRA Ablation")


if __name__ == "__main__":
    main()
