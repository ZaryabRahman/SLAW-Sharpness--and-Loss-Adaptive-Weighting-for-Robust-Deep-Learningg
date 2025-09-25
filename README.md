# SLAW: Sharpness- and Loss-Adaptive Weighting for Robust Deep Learning

This repository contains the official PyTorch implementation for the paper "SLAW: Sharpness- and Loss-Adaptive Weighting for Robust Deep Learning".

## Abstract

The generalization performance of deep neural networks is critically undermined by their convergence to sharp minima and their vulnerability to noisy labels. We propose SLAW (Sharpness- and Loss-Adaptive Weighting), a novel training algorithm that jointly tackles these challenges. SLAW integrates two adaptive mechanisms: a sharpness-aware regularizer that steers the optimizer towards flatter regions of the loss landscape, and a statistical loss reweighting scheme that dynamically filters out noisy samples during training. Our experiments demonstrate that SLAW not only matches the performance of standard methods on clean data but also confers exceptional robustness against significant label noise, preventing catastrophic overfitting where baseline models fail.

## Setup

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/your-username/SLAW-Robust-Deep-Learning.git
    cd SLAW-Robust-Deep-Learning
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

## How to Run the Experiments

All experiments are run from the command line using `main.py`. The script saves model checkpoints and training history to the `results/` directory.

### Experiment 1: Clean Data (0% Noise)

```bash
# Train Baseline (CE)
python main.py --run baseline --noise 0.0

# Train SLAW (Full)
python main.py --run slaw --noise 0.0
```

### Experiment 2 & Ablation Study (20% Noise)

This will run all models needed for both the main 20% noise comparison and the full ablation study.

```bash
# Train Baseline (CE)
python main.py --run baseline --noise 0.2

# Train SALS-only
python main.py --run sals_only --noise 0.2

# Train LAW-only
python main.py --run law_only --noise 0.2

# Train SLAW (Full)
python main.py --run slaw --noise 0.2
```

### Experiment 3: Extreme Noise (40%)

```bash
# Train Baseline (CE)
python main.py --run baseline --noise 0.4

# Train SLAW (Full)
python main.py --run slaw --noise 0.4
```

## How to Generate Plots

After running the training experiments, use `plot.py` to generate all figures from the paper. Figures will be saved to the `plots/` directory.

```bash
# Generate plots for the clean data experiment (Figures 1, 2)
python plot.py --exp clean

# Generate plots for the 20% noise experiment (Figures 3, 4)
python plot.py --exp noise20

# Generate plots for the 40% noise experiment (Figures 5, 6, 7)
python plot.py --exp noise40

# Generate plots for the ablation study (Figures 9, 10)
python plot.py --exp ablation20
```
