# SLAW: Sharpness- and Loss-Adaptive Weighting for Robust Deep Learning

This repository contains the official PyTorch implementation and experimental results for the paper **"SLAW: Sharpness- and Loss-Adaptive Weighting for Robust Deep Learning"**. All figures from the paper are included in the `/plots` directory of this repository for reference.

![SLAW vs Baseline on 20% Noise](/plots/acc-noisy-50.png)
*A key result from the paper (Figure 3): When trained on data with 20% label noise, a standard ResNet-18 suffers from catastrophic overfitting. In contrast, our proposed method, SLAW, maintains stable generalization and resists memorizing incorrect labels.*

---

## Abstract

> The generalization performance of deep neural networks is critically undermined by their convergence to sharp minima and their vulnerability to noisy labels. We propose SLAW (Sharpness- and Loss-Adaptive Weighting), a novel training algorithm that jointly tackles these challenges. SLAW integrates two adaptive mechanisms: a sharpness-aware regularizer that steers the optimizer towards flatter regions of the loss landscape, and a statistical loss reweighting scheme that dynamically filters out noisy samples during training. Our experiments demonstrate that SLAW not only matches the performance of standard methods on clean data but also confers exceptional robustness against significant label noise, preventing catastrophic overfitting where baseline models fail. By unifying landscape-aware optimization with robust learning from corrupted data, SLAW provides a practical and effective method for improving the reliability and generalization of deep learning models in the wild.

---

## The SLAW Algorithm: Core Concepts

Deep learning models face two fundamental reliability challenges: finding solutions that generalize well (i.e., avoiding sharp minima in the loss landscape) and learning from imperfect, real-world data (i.e., handling noisy labels). SLAW introduces a unified framework to address both problems simultaneously through two synergistic components:

1.  **Sharpness-Adaptive Label Smoothing (SALS):** This mechanism regularizes the model by preventing it from making overconfident predictions. Critically, the strength of this regularization is not fixed; it is **dynamically adapted** based on the local geometry of the loss landscape. We use the L2 norm of the gradient of the loss (`||∇L||₂`) as an efficient proxy for local sharpness.
    *   In **sharp regions** (high gradient norm, typically early in training), SALS applies stronger label smoothing to discourage premature convergence to unstable solutions.
    *   In **flat regions** (low gradient norm), the smoothing effect is automatically annealed, allowing the model to fine-tune its predictions with higher confidence.

2.  **Loss-Adaptive Reweighting (LAW):** This component acts as an on-the-fly statistical filter to combat noisy labels. The core intuition is that as a model learns the true data distribution, mislabeled samples will consistently yield higher loss values than correctly labeled ones. LAW identifies these samples as statistical outliers within each mini-batch and down-weights their contribution to the gradient update.
    *   It works by standardizing the per-sample losses within a batch to get a z-score.
    *   A generalized sigmoid function then maps these scores to a weight between `(1 - γ)` and `1`, effectively reducing the influence of high-loss (likely noisy) and very low-loss (already learned) samples, forcing the model to focus on informative examples.

By combining these two mechanisms, SLAW can navigate the loss landscape towards wide, generalizable minima while simultaneously cleaning the training signal from corrupting noise.

---

## Experimental Setup

All experiments were conducted using a consistent setup to ensure fair comparisons.

-   **Dataset:** CIFAR-10 (with 0%, 20%, and 40% symmetric label noise introduced to the training set).
-   **Model Architecture:** ResNet-18, pre-trained on ImageNet.
-   **Optimizer:** AdamW with a learning rate of `1e-3` and weight decay of `5e-4`.
-   **Scheduler:** Cosine Annealing Learning Rate Scheduler over 50 epochs.
-   **Hardware:** All models were trained on a single NVIDIA T4 GPU.

---

## Summary of Results and Key Achievements

Our empirical evaluation demonstrates SLAW's effectiveness across various levels of data quality.

### **Experiment 1: Performance on Clean Data**

**Objective:** To establish that SLAW does not harm performance on clean, ideal datasets.

**Achievement:** SLAW acts as an effective regularizer. It achieves a final validation accuracy (87.85%) nearly identical to the baseline (87.86%) while **significantly improving model calibration**. The Expected Calibration Error (ECE) was reduced by **14.6%** (from 7.55% to 6.45%), indicating that SLAW produces more reliable and trustworthy probability estimates without sacrificing raw performance.
![SLAW vs Baseline Training curves on clean CIFAR-10](/plots/acc-50.png)
![Internal dynamics of the SLAW algorithm during training. on clean CIFAR-10](/plots/param-50.png)

### **Experiment 2: Robustness to Moderate (20%) Label Noise**

**Objective:** To test SLAW's core ability to mitigate overfitting from corrupted labels.

**Achievement:** This experiment showcases SLAW's primary strength. While the baseline model suffers **catastrophic overfitting** after 25 epochs (losing ~3% accuracy), SLAW maintains **remarkable stability and robust generalization** throughout training. Analysis of SLAW's internal dynamics (Figure 4) provides a "smoking gun": the average sample weight assigned by the LAW component steadily decreases, providing direct evidence of its **successful, on-the-fly identification and suppression of noisy samples**.


### **Experiment 3: Probing the Limits with Extreme (40%) Noise**

**Objective:** To evaluate SLAW's behavior under severe data corruption.

**Achievement:** The baseline model's performance completely collapses in this high-noise regime, with chaotic training dynamics. SLAW, in contrast, demonstrates profound training stability, maintaining a controlled and smooth learning trajectory. This experiment revealed a fascinating insight we term **"principled under-confidence"**. While SLAW's ECE is technically higher than the baseline's, the cause is entirely different. The baseline is poorly calibrated due to aggressive *overconfidence* in its incorrect, memorized predictions. SLAW becomes systematically *under-confident* because it has correctly learned to distrust a massive portion (40%) of the training data. This learned skepticism is crucial for its robust accuracy but results in a cautious model, a predictable trade-off at the heart of SLAW's design.

### **Ablation Study**

**Objective:** To deconstruct SLAW and quantify the contribution of each component.

**Achievement:** Our ablation study confirmed the synergistic design of SLAW.
*   **LAW is the primary driver of robustness.** The LAW-only model successfully mitigates the catastrophic overfitting seen in the baseline.
*   **SALS provides a crucial performance boost.** While SALS-only is insufficient to handle noise, when combined with LAW, it helps the model converge to a better solution, achieving a higher peak accuracy (82.95%) than the LAW-only model (82.43%).

---

## How to Reproduce the Results

The following commands allow for the complete reproduction of all experiments. The `main.py` script handles training, and results are saved to the `/results` directory.

**1. Clean Data Experiments (Table I, Figs 1-2):**
```bash
python main.py --run baseline --noise 0.0
python main.py --run slaw --noise 0.0
```

**2. 20% Noise & Ablation Study (Tables II & IV, Figs 3-4, 9-10):**
```bash
python main.py --run baseline --noise 0.2
python main.py --run sals_only --noise 0.2
python main.py --run law_only --noise 0.2
python main.py --run slaw --noise 0.2
```

**3. 40% Extreme Noise Experiments (Table III, Figs 5-7):**
```bash
python main.py --run baseline --noise 0.4
python main.py --run slaw --noise 0.4```

---

## generating Figures

After the training experiments are complete, you can generate all figures from the paper using `plot.py`. The script reads the saved logs from the `/results` directory and outputs high-resolution images to `/plots`.

```bash
# generate plots for the clean data experiment (Figures 1, 2)
python plot.py --exp clean

# generate plots for the 20% noise experiment (Figures 3, 4)
python plot.py --exp noise20

# generate plots for the 40% noise experiment (Figures 5, 6, 7)
python plot.py --exp noise40

# generate plots for the ablation study (Figures 9, 10)
python plot.py --exp ablation20
```

---

## How to Cite

If you find this work useful in your research, please cite our work (the detials will be updated upon accpetance):

```bibtex
@inproceedings{rahman2025slaw,
  title={{SLAW}: {S}harpness- and {L}oss-{A}daptive {W}eighting for {R}obust {D}eep {L}earning},
  author={Rahman, Zaryab},
  booktitle={Proceedings of the International Conference on Machine Learning (ICML)},
  year={2025},
  note={Code available at: https://github.com/ZaryabRahman/SLAW-Sharpness--and-Loss-Adaptive-Weighting-for-Robust-Deep-Learningg}
}
```
