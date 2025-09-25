import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os
import torch

from utils import History, get_calibration_data, calculate_ece
from dataset import get_dataloaders
from main import get_model

def plot_acc_loss(history: History, run_names: list[str], legend_names: list[str], title_suffix: str, filename: str):
    """
    gnerates and saves validation accuracy and loss curves over epochs.

    :param history: The History object containing the training logs.
    :type history: History
    :param run_names: A list of internal run names to plot from the history object.
    :type run_names: list[str]
    :param legend_names: A list of display names for the legend, corresponding to `run_names`.
    :type legend_names: list[str]
    :param title_suffix: A string to append to the plot titles (e.g., "on 20% Noise").
    :type title_suffix: str
    :param filename: The base name for the saved plot image file.
    :type filename: str
    """
    sns.set_style("whitegrid")
    palette = sns.color_palette("deep", len(run_names))
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    for i, run_name in enumerate(run_names):
        if run_name in history.history:
            df = history.get_epoch_df(run_name)
            label = legend_names[i]
            axs[0].plot(df['epoch'], df['val_acc'], label=label, marker='o', linestyle='-', markersize=4, color=palette[i])
            axs[1].plot(df['epoch'], df['val_loss'], label=label, marker='o', linestyle='-', markersize=4, color=palette[i])

    axs[0].set_title(f"Validation Accuracy vs. Epochs {title_suffix}", fontsize=16)
    axs[0].set_xlabel("Epoch"); axs[0].set_ylabel("Accuracy (%)"); axs[0].legend(); axs[0].grid(True)
    axs[1].set_title(f"Validation Loss vs. Epochs {title_suffix}", fontsize=16)
    axs[1].set_xlabel("Epoch"); axs[1].set_ylabel("Loss"); axs[1].legend(); axs[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', dpi=300)
    print(f"Saved plot to plots/{filename}.png")
    plt.show()

def plot_slaw_dynamics(history: History, run_name: str, filename: str):
    """
    visuaalizes the internal dynamics of the SLAW algorithm over training steps.

    this function plots the sharpness proxy (batch gradient norm and its EMA) and
    the adaptive parameters (average epsilon and average weight).

    :param history: The History object with logged step metrics.
    :type history: History
    :param run_name: The specific SLAW run name to visualize.
    :type run_name: str
    :param filename: The base name for the saved plot image file.
    :type filename: str
    """
    if run_name not in history.history:
        print(f"Run '{run_name}' not found in history for dynamics plot.")
        return
        
    sns.set_style("whitegrid")
    palette = sns.color_palette("deep")
    slaw_step_df = history.get_step_df(run_name)
    if slaw_step_df.empty or 's_batch' not in slaw_step_df.columns:
        print(f"No step data or missing columns for {run_name}.")
        return

    slaw_step_df_smooth = slaw_step_df.rolling(window=100, min_periods=1).mean()
    fig, axs = plt.subplots(1, 2, figsize=(16, 6))

    # sharpness proxy
    axs[0].plot(slaw_step_df_smooth.index, slaw_step_df_smooth['s_batch'], label='Batch Grad Norm (Smoothed)', alpha=0.8, color=palette[0])
    axs[0].plot(slaw_step_df_smooth.index, slaw_step_df_smooth['sharp_ema'], label='EMA of Grad Norm', linestyle='--', color='red')
    axs[0].set_title("SLAW: Batch Sharpness Proxy", fontsize=16)
    axs[0].set_xlabel("Training Step"); axs[0].set_ylabel("L2 Norm of Gradient"); axs[0].legend(); axs[0].grid(True); axs[0].set_yscale('log')

    # adaptive params
    ax2 = axs[1].twinx()
    p1, = axs[1].plot(slaw_step_df_smooth.index, slaw_step_df_smooth['eps_mean'], label='Avg. $\\epsilon_i$', color=palette[0])
    p2, = ax2.plot(slaw_step_df_smooth.index, slaw_step_df_smooth['weight_mean'], label='Avg. $w_i$', color=palette[1])
    axs[1].set_title("SLAW: Adaptive Parameters", fontsize=16); axs[1].set_xlabel("Training Step")
    axs[1].set_ylabel("Avg. Epsilon", color=palette[0]); ax2.set_ylabel("Avg. Weight", color=palette[1])
    axs[1].tick_params(axis='y', labelcolor=palette[0]); ax2.tick_params(axis='y', labelcolor=palette[1])
    axs[1].legend(handles=[p1, p2], loc='best')

    plt.tight_layout()
    plt.savefig(f'plots/{filename}.png', dpi=300)
    print(f"Saved plot to plots/{filename}.png")
    plt.show()

def plot_reliability_diagrams(run_names: list[str], legend_names: list[str], filename: str, n_bins: int = 15, device: str = 'cuda'):
    """
    generates and saves reliability diagrams for one or more models.

    :param run_names: A list of internal run names to plot.
    :type run_names: list[str]
    :param legend_names: A list of display names for the titles.
    :type legend_names: list[str]
    :param filename: The base name for the saved plot image file.
    :type filename: str
    :param n_bins: The number of confidence bins to use.
    :type n_bins: int
    :param device: The device to run model inference on.
    :type device: str
    """
    num_plots = len(run_names)
    fig, axs = plt.subplots(1, num_plots, figsize=(8 * num_plots, 7), squeeze=False)
    axs = axs.flatten()

    for i, run_name in enumerate(run_names):
        model = get_model()
        model_path = f"results/{run_name}_best_model.pth"
        if not os.path.exists(model_path):
            print(f"Model file not found at {model_path}. Skipping reliability diagram.")
            continue
            
        model.load_state_dict(torch.load(model_path, map_location=device))
        _, testloader = get_dataloaders(batch_size=128)
        
        accuracies, avg_confs, _ = get_calibration_data(model, testloader, n_bins=n_bins, device=device)
        ece = calculate_ece(model, testloader, n_bins=n_bins, device=device)
        title = f'{legend_names[i]}\nECE = {ece:.4f}'

        ax = axs[i]
        bin_centers = np.linspace(0, 1, n_bins + 1)[:-1] + (1/(2*n_bins))
        ax.bar(bin_centers, accuracies, width=1/n_bins, edgecolor='black', color='blue', alpha=0.6, label='Accuracy')
        ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfect Calibration')

        # add gap viusal
        for j, (conf, acc) in enumerate(zip(avg_confs, accuracies)):
            if conf > acc: # overconfident
                ax.add_patch(plt.Rectangle((bin_centers[j] - 1/(2*n_bins), acc), 1/n_bins, conf-acc, color='red', alpha=0.4, hatch='/', edgecolor='black'))
            else: # underconfidenet
                ax.add_patch(plt.Rectangle((bin_centers[j] - 1/(2*n_bins), conf), 1/n_bins, acc-conf, color='blue', alpha=0.4, hatch='\\', edgecolor='black'))

        ax.set_title(title, fontsize=16)
        ax.set_xlabel('Confidence', fontsize=12); ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(); ax.grid(True, linestyle=':')

    plt.tight_layout(pad=3.0)
    plt.savefig(f'plots/{filename}.png', dpi=300)
    print(f"Saved plot to plots/{filename}.png")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate plots for SLAW experiments.")
    parser.add_argument('--exp', type=str, required=True, choices=['clean', 'noise20', 'noise40', 'ablation20'], help="Which experiment to plot.")
    args = parser.parse_args()

    if args.exp == 'clean':
        history = History(); history.load('results/history_noise_0.pkl')
        runs = ['baseline_noise0', 'slaw_noise0']
        legends = ['Baseline (CE)', 'SLAW (ours)']
        plot_acc_loss(history, runs, legends, "on Clean Data", "Figure_1_acc_loss_clean")
        plot_slaw_dynamics(history, 'slaw_noise0', "Figure_2_dynamics_clean")
        
    elif args.exp == 'noise20':
        history = History(); history.load('results/history_noise_20.pkl')
        runs = ['baseline_noise20', 'slaw_noise20']
        legends = ['Baseline (CE) on 20% Noise', 'SLAW on 20% Noise']
        plot_acc_loss(history, runs, legends, "on 20% Noise", "Figure_3_acc_loss_noise20")
        plot_slaw_dynamics(history, 'slaw_noise20', "Figure_4_dynamics_noise20")

    elif args.exp == 'noise40':
        history = History(); history.load('results/history_noise_40.pkl')
        runs = ['baseline_noise40', 'slaw_noise40']
        legends = ['Baseline (CE) on 40% Noise', 'SLAW on 40% Noise']
        plot_acc_loss(history, runs, legends, "on 40% Noise", "Figure_5_acc_loss_noise40")
        plot_slaw_dynamics(history, 'slaw_noise40', "Figure_6_dynamics_noise40")
        plot_reliability_diagrams(runs, legends, "Figure_7_reliability_noise40")

    elif args.exp == 'ablation20':
        history = History(); history.load('results/history_noise_20.pkl')
        runs = ['baseline_noise20', 'sals_only_noise20', 'law_only_noise20', 'slaw_noise20']
        legends = ['Baseline (CE)', 'SALS-only', 'LAW-only', 'SLAW (Full)']
        plot_acc_loss(history, runs, legends, " (Ablation Study on 20% Noise)", "Figure_9_ablation_acc_loss")
        plot_reliability_diagrams(runs, legends, "Figure_10_reliability_ablation")
