# Continual Learning for Saudi Dialect Offensive Language Detection under Temporal Linguistic Drift

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Afefa-Asiri/Continual-Learning-for-Saudi-Dialect-Offensive-Language-Detection-under-Temporal-Linguistic-Drift/blob/main/CL_Experiments.ipynb)

This repository provides the implementation code for our paper on **continual learning** for **Saudi dialect offensive language detection** under **temporal linguistic drift**.

## Abstract

Offensive language detection systems degrade over time as linguistic patterns evolve, particularly in dialectal Arabic social media where new terms emerge and familiar expressions shift in meaning. This study investigates temporal linguistic drift in Saudi-dialect offensive language detection through a systematic evaluation of continual learning approaches including **Experience Replay (ER)**, **Elastic Weight Consolidation (EWC)**, and **Low-Rank Adaptation (LoRA)**.

## Repository Contents

```
├── CL_Experiments.ipynb    # Main experiment notebook (Google Colab)
├── requirements.txt        # Python dependencies
├── README.md              # This file
└── LICENSE                # License file
```

## Quick Start (Google Colab)

1. **Click the "Open in Colab" badge** above
2. **Upload your data files** to Google Drive:
   ```
   /content/drive/MyDrive/CL_Experiment/data/
   ├── SOD_AraBERT_model/          # Pre-trained SOD_AraBERT checkpoint
   ├── SDOffensive_Paper2.csv      # Original SOD training data (2019-2022)
   ├── Paper2_DS_Complete.csv      # New training data (2024-2025)
   ├── processed500UnseenDS_Paper2.csv  # Historical test set
   ├── Balanced500_Paper2.csv      # Contemporary test set
   ├── TestDS2080_Paper2.csv       # Mixed 80-20 test set
   └── TestDS4060_Paper2.csv       # Mixed 40-60 test set
   ```
3. **Update the paths** in the Configuration section
4. **Run all cells**

## Experiment Structure

```
Phase 1: LoRA Ablation Study
├── r=8  (5 seeds)
├── r=16 (5 seeds)  ← Optimal rank
├── r=32 (5 seeds)
└── r=64 (5 seeds)
    → Select optimal rank based on balanced (KR + AG) score

Phase 2: Main Continual Learning Experiments (using optimal LoRA rank)
├── Original (baseline)
├── Naïve Fine-tuning
├── Experience Replay (ER)
├── Elastic Weight Consolidation (EWC)
├── LoRA
├── LoRA + ER
├── LoRA + EWC
├── LoRA + ER + EWC
└── Full + ER + EWC
```

## Methods Implemented

| Method | Description |
|--------|-------------|
| **Experience Replay (ER)** | Combines samples from original SOD training split with new data |
| **EWC** | Elastic Weight Consolidation with Fisher Information Matrix (λ=1000) |
| **LoRA** | Low-Rank Adaptation applied to Q, K, V attention layers |
| **Hybrid methods** | Combinations of ER, EWC, and LoRA |

## Evaluation Metrics

- **Knowledge Retention (KR)**: `F1_historical^after - F1_historical^before` (measures forgetting)
- **Adaptation Gain (AG)**: `F1_contemporary^adapted - F1_contemporary^original` (measures learning)
- **F1-macro, Accuracy**: Standard classification metrics

## Data Availability

The datasets used in this study contain culturally sensitive Saudi dialect offensive language content and are **not publicly released**. They are available from the corresponding author upon reasonable request for legitimate research purposes.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- Transformers 4.20+
- PEFT 0.4+
- See `requirements.txt` for full list

## Citation

If you use this code in your research, please cite:

```bibtex
@article{asiri2025continual,
  title={Continual Learning for Saudi Dialect Offensive Language Detection under Temporal Linguistic Drift},
  author={Asiri, Afefa and Saleh, Mostafa},
  journal={Information},
  year={2025},
  publisher={MDPI}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Afefa Asiri** - aasiri0410@stu.kau.edu.sa
- Faculty of Computing and Information Technology, King Abdulaziz University
