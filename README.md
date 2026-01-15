# Continual Learning for Saudi Dialect Offensive Language Detection under Temporal Linguistic Drift

This repository provides the implementation code for our paper on **continual learning** for **Saudi dialect offensive language detection** under **temporal linguistic drift**.

## Abstract

Offensive language detection systems degrade over time as linguistic patterns evolve, particularly in dialectal Arabic social media where new terms emerge and familiar expressions shift in meaning. This study investigates temporal linguistic drift in Saudi-dialect offensive language detection through a systematic evaluation of continual learning approaches including **Experience Replay (ER)**, **Elastic Weight Consolidation (EWC)**, and **Low-Rank Adaptation (LoRA)**.

## Repository Contents

```
├── CL_Experiments.ipynb    # Main experiment notebook
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── LICENSE                 # License file
```

## Test Scenarios

| Test Set | Description |
|----------|-------------|
| **Historical** | 500 samples from SOD corpus (2019–2022), no drift terms |
| **Contemporary** | 500 samples with newly emerged terms and context-shifting expressions |
| **Mixed 20-80** | 20% contemporary + 80% historical |
| **Mixed 40-60** | 40% contemporary + 60% historical |
| **SimuReal** | Realistic distribution: 80% non-offensive, 20% offensive (5% drift terms) |

## Experiment Structure

```
Phase 1: Original Model (Baseline)

Phase 2: LoRA Ablation Study
├── Standard modules (Q, K, V) × ranks {8, 16, 32}
├── Extended modules (Q, K, V, O) × ranks {8, 16, 32}
└── 3 seeds each
    → Select optimal configuration based on weighted criterion (0.6×KR + 0.4×AG)

Phase 2.5: EWC Lambda Ablation
├── λ ∈ {100, 500, 1000, 5000, 10000}
└── 3 seeds each

Phase 2.6: ER Buffer Size Ablation
├── Replay ratio ∈ {0.1, 0.2, 0.4, 0.6, 0.8}
└── 3 seeds each

Phase 3: Main Continual Learning Methods (5 seeds each)
├── Original (baseline)
├── Naïve Fine-tuning
├── Experience Replay (ER)
├── Elastic Weight Consolidation (EWC)
└── LoRA

Phase 4: Hybrid Methods (5 seeds each)
├── LoRA + ER
├── LoRA + EWC
├── LoRA + ER + EWC
└── Full + ER + EWC
```

## Methods Implemented

| Method | Description |
|--------|-------------|
| **Experience Replay (ER)** | Combines samples from original SOD training split with new data |
| **EWC** | Elastic Weight Consolidation with Fisher Information Matrix |
| **LoRA** | Low-Rank Adaptation applied to attention layers |
| **Hybrid methods** | Combinations of ER, EWC, and LoRA |

## Evaluation Metrics

- **Knowledge Retention (KR)**: `F1_historical^after - F1_historical^before` (measures forgetting)
- **Adaptation Gain (AG)**: `F1_contemporary^adapted - F1_contemporary^original` (measures learning)
- **F1-macro, Accuracy**: Standard classification metrics
- **Per-class F1**: F1-OFF (offensive) and F1-NOT (non-offensive)

## Data Availability

The datasets used in this study contain culturally sensitive Saudi dialect offensive language content and are **not publicly released**. They are available from the corresponding author upon reasonable request for legitimate research purposes.

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.20+
- PEFT 0.4+
- See `requirements.txt` for full list

## Citation

If you use this code in your research, please cite:

```bibtex
@article{asiri2026continual,
  title={Continual Learning for Saudi Dialect Offensive Language Detection under Temporal Linguistic Drift},
  author={Asiri, Afefa and Saleh, Mostafa},
  journal={Information},
  year={2026},
  publisher={MDPI}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
