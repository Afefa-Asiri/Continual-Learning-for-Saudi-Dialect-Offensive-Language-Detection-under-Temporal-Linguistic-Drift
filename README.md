# Continual Learning for Saudi Dialect Offensive Language Detection under Temporal Linguistic Drift

This repository provides the **full experiment code** used in our paper on continual learning (CL) for **Saudi dialect offensive language detection** under **temporal linguistic drift**.

The script reproduces the training/evaluation pipeline and automatically generates the main **tables** and **figures** used in the paper, including:
- Baseline (original model) evaluation
- Continual learning methods (e.g., **Experience Replay (ER)**, **Elastic Weight Consolidation (EWC)**)
- **LoRA** and LoRA variants, including **LoRA rank ablation** (performed first, then the chosen rank is used for subsequent runs)
- Confusion matrices and summary visualizations
- Result tables (F1/Accuracy, per-class metrics, BWT/FWT analysis, timing, and ablation summary)

> Note: This repo contains **code only**. Datasets and checkpoints are not included.

---

## Repository contents

- `cl_complete_experiments.py` — main runnable script
- `requirements.txt` — Python dependencies

---

## Data and model layout (expected folders)

Place your resources using the following **relative** structure (GitHub-friendly):

```text
./data/
  Paper2_DS_Complete.csv
  SDOffensive_Paper2.csv
  processed500UnseenDS_Paper2.csv
  Balanced500_Paper2.csv
  TestDS2080_Paper2.csv
  TestDS4060_Paper2.csv

./models/
  SOD_AraBERT_model/   # the base/original AraBERT checkpoint folder
```

The script writes all generated outputs to:

```text
./results/
  figures/
  tables/
  (and other run artifacts)
```

If your filenames differ, update the corresponding path variables near the top of `cl_complete_experiments.py`.

---

## Installation

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -U pip
pip install -r requirements.txt
```

---

## Run the experiments

```bash
python cl_complete_experiments.py
```

### Notes
- Running the full pipeline can be computationally expensive. A GPU is strongly recommended.
- Results can vary slightly based on GPU type and library versions. For reproducibility, keep your environment stable.

---

## Outputs

After the run finishes, you will find:
- Figures in: `./results/figures/`
- Tables in:  `./results/tables/`

---

## Data availability

The datasets used in this study are derived from Saudi social-media posts and may contain sensitive/offensive content.
Therefore, the datasets are **not publicly released**.

They are available from the corresponding author **upon reasonable request**.

---

## Citation

If you use this code in academic work, please cite the paper associated with this repository.

---

## License

Add a license file (e.g., MIT) if you plan to make this repository publicly reusable.
