# Simulation of Diff-LR vs Learning Rate Finder (LRF)
# This code is intended to produce reproducible, transparent simulation results for the blog-style research note.
# It generates synthetic "epochs-to-target" numbers using a parametric growth model and algorithmic assumptions.
# Results will be displayed as tables and plotted using matplotlib. DataFrame shown with caas_jupyter_tools.display_dataframe_to_user.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import exp
import io, os, json

# Display utility (provided by the environment)
from caas_jupyter_tools import display_dataframe_to_user

np.random.seed(42)

# --- Configuration / assumptions (documented in paper) ---
datasets = {
    "MNIST": {"target_acc": 0.98, "difficulty": 0.6, "max_acc": 0.995},
    "CIFAR-10": {"target_acc": 0.80, "difficulty": 1.0, "max_acc": 0.90},
    "ImageNet-subset": {"target_acc": 0.60, "difficulty": 1.6, "max_acc": 0.75}
}

# Models (representing major families); parameter counts will be adjusted to match buckets
model_families = {
    "MLP": {"base_k": 0.30},          # base learning speed coefficient
    "Small-CNN": {"base_k": 0.25},
    "ResNet-ish": {"base_k": 0.22},
    "Tiny-Transformer": {"base_k": 0.20}
}

param_buckets = {
    "Small (~10k)": 10_000,
    "Medium (~100k)": 100_000,
    "Large (~1M)": 1_000_000,
    "XL (~10M)": 10_000_000
}

# Algorithmic assumptions for LR-selection methods (these are explicit assumptions used by the simulation)
# Success: probability that the method finds a "near-optimal" LR (interpreted as quality factor q ~ 1)
# If it fails, the chosen LR has quality q sampled from a distribution that slows convergence (lower q) or leads to divergence.
assumptions = {
    "LRF": {
        "exploration_epochs": 0.5,  # approximate effective cost in epochs for the LR range test
        "base_success_prob": {"Small (~10k)": 0.75, "Medium (~100k)": 0.65, "Large (~1M)": 0.55, "XL (~10M)": 0.45},
        "success_q_dist": {"mean": 0.98, "std": 0.03},
        "failure_q_dist": {"mean": 0.5, "std": 0.15},
        "divergence_prob_when_failure": 0.10  # when failure, small chance of divergence (no convergence)
    },
    "Diff-LR": {
        "exploration_epochs": 1.0,  # more short independent runs; higher overhead but more reliable
        "base_success_prob": {"Small (~10k)": 0.88, "Medium (~100k)": 0.80, "Large (~1M)": 0.72, "XL (~10M)": 0.60},
        "success_q_dist": {"mean": 0.995, "std": 0.02},
        "failure_q_dist": {"mean": 0.65, "std": 0.12},
        "divergence_prob_when_failure": 0.06
    }
}

# Growth model: accuracy(t) = max_acc * (1 - exp(-k * q * t))
# where k = base_k * capacity_factor / difficulty
# capacity_factor increases with parameter count but with diminishing returns.
def capacity_factor(params):
    # Diminishing returns: log-scale mapping, normalized so ~10k->0.7, 10M->1.3
    return 0.7 + 0.6 * (np.log10(params) - 4) / 6  # log10(10k)=4, log10(10M)=7 -> maps 0.7..1.3

def epochs_to_target(dataset, model_family, params, q, seed=None, max_epochs=500):
    # returns number of training epochs needed to reach target acc, or np.inf if not reached
    if seed is not None:
        np.random.seed(seed)
    ds = datasets[dataset]
    base_k = model_families[model_family]["base_k"]
    cap = capacity_factor(params)
    difficulty = ds["difficulty"]
    k = base_k * cap / difficulty
    max_acc = ds["max_acc"]
    target = ds["target_acc"]
    # If q is extremely small or negative, training stalls or diverges
    if q <= 0.05:
        return np.inf
    # Effective learning speed scalar depends on q (q in (0, ~1]). If q>1 it's slightly faster but risky; we'll cap
    effective_speed = k * q
    # Invert growth model to solve for t: target = max_acc * (1 - exp(-effective_speed * t))
    if target >= max_acc:
        # target unreachable with this model-dataset pairing
        return np.inf
    try:
        t = -np.log(1 - target / max_acc) / (effective_speed + 1e-12)
    except Exception:
        return np.inf
    # Add a small stochastic jitter to reflect training variance
    jitter = np.random.normal(loc=1.0, scale=0.07)
    t = t * jitter
    if t > max_epochs:
        return np.inf
    return t

# Simulation loop: run many trials across combinations
n_trials = 200  # seeds per combo to estimate mean and std
results = []

for dataset in datasets:
    for model in model_families:
        for bucket_name, params in param_buckets.items():
            for method in ["LRF", "Diff-LR"]:
                cfg = assumptions[method]
                expl_epochs = cfg["exploration_epochs"]
                success_prob = cfg["base_success_prob"][bucket_name]
                success_q_mu = cfg["success_q_dist"]["mean"]
                success_q_sigma = cfg["success_q_dist"]["std"]
                fail_q_mu = cfg["failure_q_dist"]["mean"]
                fail_q_sigma = cfg["failure_q_dist"]["std"]
                div_prob = cfg["divergence_prob_when_failure"]
                epochs_list = []
                failures = 0
                divergences = 0
                for trial in range(n_trials):
                    # determine if method finds good LR
                    if np.random.rand() < success_prob:
                        q = np.random.normal(success_q_mu, success_q_sigma)
                    else:
                        q = np.random.normal(fail_q_mu, fail_q_sigma)
                        # divergence can happen
                        if np.random.rand() < div_prob:
                            q = 0.02  # near-zero causing divergence
                    # clip q to sensible range
                    q = float(np.clip(q, 0.01, 1.5))
                    train_epochs = epochs_to_target(dataset, model, params, q, seed=None, max_epochs=1000)
                    # If diverged or didn't reach target, mark as inf -> we'll cap at max cost proxy later
                    if np.isinf(train_epochs):
                        divergences += 1
                        # represent as a large penalty (we'll also report divergence count)
                        total_epochs = np.inf
                    else:
                        total_epochs = expl_epochs + train_epochs
                    epochs_list.append(total_epochs)
                # summarize
                finite = [e for e in epochs_list if np.isfinite(e)]
                mean_epochs = np.mean(finite) if len(finite)>0 else np.inf
                std_epochs = np.std(finite) if len(finite)>0 else np.inf
                divergence_rate = divergences / n_trials
                results.append({
                    "Dataset": dataset,
                    "Model": model,
                    "Param Bucket": bucket_name,
                    "Params": params,
                    "Method": method,
                    "Mean Epochs (to target, excl. diverged)": mean_epochs,
                    "Std Epochs": std_epochs,
                    "Divergence Rate": divergence_rate,
                    "Trials": n_trials,
                    "Exploration Epochs": expl_epochs
                })

df = pd.DataFrame(results)

# Pivot table for easier reading: mean epochs for each (dataset, model, param bucket, method)
pivot = df.pivot_table(index=["Dataset","Model","Param Bucket","Params"], columns="Method", values=["Mean Epochs (to target, excl. diverged)","Std Epochs","Divergence Rate","Exploration Epochs"])
pivot = pivot.reset_index()

# Display readable DataFrame to user
display_dataframe_to_user("Diff-LR_vs_LRF_sim_results", df)

# Save figures: epochs vs param bucket plot per dataset+model family
plots_dir = "/Your_directory/data/sim_plots" #If you intend on reproducing this code make sure that you update this to the directory wherein you want to save the plots_dir.
os.makedirs(plots_dir, exist_ok=True)

for dataset in datasets:
    for model in model_families:
        sub = df[(df["Dataset"]==dataset)&(df["Model"]==model)]
        if sub.empty:
            continue
        # Prepare x as log(params)
        x = [np.log10(p) for p in sub["Params"]]
        methods = ["LRF","Diff-LR"]
        plt.figure(figsize=(8,5))
        for method in methods:
            s = sub[sub["Method"]==method].sort_values("Params")
            y = s["Mean Epochs (to target, excl. diverged)"].values
            # Plot with markers; don't specify colors (tooling rule)
            plt.plot(x[:len(y)], y, marker='o', label=method)
        plt.xlabel("log10(Parameters)")
        plt.ylabel("Mean epochs to target (excl. diverged)")
        plt.title(f"{dataset} — {model}: Epochs to Target vs Model Size")
        plt.legend()
        fname = f"{plots_dir}/{dataset}_{model}_epochs_vs_params.png".replace(" ","_")
        plt.savefig(fname, bbox_inches='tight', dpi=150)
        plt.close()

# Save CSV of results
csv_path = "/Your_directory/data/diff_lr_sim_results.csv" #If you intend on reproducing this code make sure that you update this to the directory wherein you want to save this.
df.to_csv(csv_path, index=False)

# Provide a small summary printed out for quick glance
summary = df.groupby(["Dataset","Param Bucket","Method"]).agg({
    "Mean Epochs (to target, excl. diverged)": "mean",
    "Std Epochs": "mean",
    "Divergence Rate": "mean"
}).reset_index().sort_values(["Dataset","Param Bucket"])

print("Simulation complete. Results CSV saved to:", csv_path)
print("Plots saved to:", plots_dir)

# Return a short JSON with paths for the assistant to reference in the writeup
output = {
    "csv": csv_path,
    "plots_dir": plots_dir,
    "n_trials": n_trials
}
output_json = json.dumps(output)
output_json

