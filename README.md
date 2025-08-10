# Dynamic-LR
This repository provides a method for dynamic learning rate optimization. It contains a Python implementation of an intelligent search algorithm that efficiently discovers an optimal initial learning rate, setting a strong foundation for stable and effective model training.

## 💡 The Core Idea
The core challenge in training deep models is setting a proper learning rate. This method automates the discovery of an optimal starting rate, which is the crucial first step in any dynamic learning rate strategy (e.g., when using schedulers like ReduceLROnPlateau).

To achieve this, the algorithm performs a series of short, independent test runs within a given range. For each learning rate tested, it determines if the model is:

## 💥 Overshooting: 
The learning rate is too high, causing the training loss to explode or diverge. This rate is discarded, and the search continues in the lower half.

## 📉 Viable (Balanced or Undershooting): 
The learning rate leads to stable training. This rate is considered a potential candidate, and the search continues in the upper half to see if a better rate exists.

By intelligently narrowing the search space, the algorithm quickly converges on a robust initial learning rate.

## ✨ Features
1. Automated Rate Finding: Eliminates manual guesswork and trial-and-error.

2. Foundation for Dynamic Rates: Provides a strong, validated starting point for any learning rate schedule.

3. Efficient Search Logic: Converges on a solution quickly, minimizing the computational cost of finding a good rate.
