# Dynamic-LR: An Intelligent Method for Dynamic Learning Rate Optimization

This repository provides a method for dynamic learning rate optimization. It contains a Python implementation of an intelligent search algorithm that efficiently discovers an optimal initial learning rate, setting a strong foundation for stable and effective model training.

-----

## 💡 The Core Idea

The core challenge in training deep models is setting a proper learning rate. This method automates the discovery of an optimal starting rate, which is the crucial first step in any dynamic learning rate strategy (e.g., when using schedulers like `ReduceLROnPlateau`).

The algorithm performs a series of short, independent test runs to determine if the model is:

  * 💥 **Overshooting**: The learning rate is too high, causing the training loss to explode or diverge.
  * 📉 **Viable**: The learning rate leads to stable training and is considered a potential candidate.

By intelligently narrowing the search space, the algorithm quickly converges on a robust initial learning rate.

-----

## 🔬 How It Works: A Two-Phase Approach

The "Dynamic Learning Rate" (DLR) algorithm is implemented in two main phases.

### Phase 1: Binary Search for Initial Learning Rate

This phase uses a binary search to find a good starting learning rate. It tests for overshooting (where the validation loss explodes) versus undershooting or balanced training.

1.  **Set Bounds**: Start with a low learning rate (e.g., `1e-6`) and a high one (e.g., `1e-1`).
2.  **Test Midpoint**: Train the model for a few epochs with the midpoint learning rate.
3.  **Adjust Bounds**:
      * If the model **overshoots**, the midpoint becomes the new `high`.
      * Otherwise, the midpoint becomes the new `low`.
4.  **Repeat** until the range is sufficiently narrow.

### Phase 2: Dynamic Adjustments During Full Training

With a good initial rate, the second phase monitors training and dynamically adjusts the learning rate based on performance.

  * **Overshoot Detection**: If validation loss increases while training loss decreases, the learning rate is reduced.
  * **Undershoot Detection**: If both losses get stuck at a high value, the learning rate is increased.

-----

## 📊 Simulation: Diff-LR vs. Learning Rate Finder (LRF)

A simulation was conducted to compare the performance of **Diff-LR** against a traditional **Learning Rate Finder (LRF)**. The simulation generated synthetic "epochs-to-target" numbers across various datasets, model architectures, and model sizes.

### Key Findings from the Simulation

The results consistently show that **Diff-LR requires fewer epochs to reach the target accuracy** compared to LRF across all tested configurations. This indicates that Diff-LR is a more efficient and reliable method for finding an optimal learning rate.

Below are some of the plots from the simulation that illustrate these findings:

  * **CIFAR-10 — Tiny-Transformer**:

\<img src="joshuaplrj/diff-lr/Diff-LR-Ben\_Sim/diff\_lr\_sim\_plots/CIFAR-10\_Tiny-Transformer\_epochs\_vs\_params.png" alt="CIFAR-10 Tiny-Transformer Plot"\>

  * **ImageNet-subset — Small-CNN**:

\<img src="joshuaplrj/diff-lr/Diff-LR-Ben\_Sim/diff\_lr\_sim\_plots/ImageNet-subset\_Small-CNN\_epochs\_vs\_params.png" alt="ImageNet-subset Small-CNN Plot"\>

  * **MNIST — Small-CNN**:

\<img src="joshuaplrj/diff-lr/Diff-LR-Ben\_Sim/diff\_lr\_sim\_plots/MNIST\_Small-CNN\_epochs\_vs\_params.png" alt="MNIST Small-CNN Plot"\>

The full simulation results can be found in `diff_lr_sim_results.csv`.

-----

## 🚀 The Journey Behind the Idea

The concept for this dynamic learning rate finder was inspired by the desire to automate and improve upon traditional methods. For a detailed narrative of how this idea was developed, from understanding overfitting and underfitting to the "aha\!" moment of applying binary search, check out [The\_Discovery.md](https://www.google.com/search?q=joshuaplrj/diff-lr/Diff-LR-Ben_Sim/The_Discovery.md).

-----

## ✨ Features

  * **Automated Rate Finding**: Eliminates manual guesswork and trial-and-error.
  * **Foundation for Dynamic Rates**: Provides a strong, validated starting point for any learning rate schedule.
  * **Efficient Search Logic**: Converges on a solution quickly, minimizing the computational cost.

-----

## Implementation

A minimal, self-contained PyTorch implementation of the two-phase "Dynamic Learning Rate" algorithm can be found in `DLR.py`.

-----

## 📜 License

This repository is licensed under the Apache License, Version 2.0. See the [LICENSE](https://www.google.com/search?q=joshuaplrj/diff-lr/Diff-LR-Ben_Sim/LICENSE) file for details.
