# Theoretical Foundations of the Weighted Naive Bayes Framework

This document explains the theoretical concepts and algorithms used in this framework that are not typical in standard machine learning implementations.

## Table of Contents

1. [Differential Evolution (DE)](#differential-evolution-de)
2. [PR-AUC Optimization](#pr-auc-optimization)
3. [Weighted Naive Bayes Multiplication](#weighted-naive-bayes-multiplication)
4. [Probability Calibration Methods](#probability-calibration-methods)
5. [Three-Way Data Split](#three-way-data-split)
6. [Gradient-Free Optimization](#gradient-free-optimization)
7. [Early Stopping with Best Weight Restoration](#early-stopping-with-best-weight-restoration)

## Differential Evolution (DE)

### What is it?

Differential Evolution is a population-based metaheuristic optimization algorithm that belongs to the class of evolutionary algorithms. Unlike gradient-based methods (like SGD or Adam), DE doesn't require the objective function to be differentiable.

### How it works:

1. **Initialize Population**: Create a random population of candidate solutions (weight vectors)
2. **Mutation**: For each member, create a mutant vector by adding weighted differences between population members
3. **Crossover**: Mix the mutant vector with the original to create a trial vector
4. **Selection**: Keep the better solution between trial and original

### Mathematical formulation:

```
Mutation: v_i = x_r1 + F * (x_r2 - x_r3)
Crossover: u_i,j = v_i,j if rand() < CR, else x_i,j
Selection: x_i(t+1) = u_i if f(u_i) < f(x_i), else x_i
```

Where:
- F = mutation factor (typically 0.5-2.0)
- CR = crossover rate (typically 0.1-1.0)

### Where it's used in this framework:

- **File**: `pkg/optimizer/differential_evolution.go`
- **Purpose**: Optimizing ensemble weights without requiring gradient information
- **Advantage**: Handles non-convex, non-differentiable objective functions (PR-AUC)

## PR-AUC Optimization

### What is it?

Precision-Recall Area Under Curve (PR-AUC) is a performance metric particularly useful for imbalanced datasets. Unlike ROC-AUC, PR-AUC focuses on the performance with respect to the positive class.

### Why PR-AUC instead of accuracy or ROC-AUC?

1. **Imbalanced data**: When negative class >> positive class, accuracy and ROC-AUC can be misleading
2. **Focus on positives**: PR curves show the trade-off between precision (correct positive predictions) and recall (finding all positives)
3. **Non-differentiable**: PR-AUC is not smoothly differentiable, making gradient-based optimization challenging

### Mathematical formulation:

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
PR-AUC = ∫ Precision(Recall) d(Recall)
```

### Where it's used in this framework:

- **File**: `pkg/metrics/metrics.go`
- **Purpose**: Primary optimization objective for ensemble weights
- **Implementation**: Trapezoidal integration over precision-recall curve

## Weighted Naive Bayes Multiplication

### What is it?

A novel ensemble combination method that uses weighted multiplication instead of weighted averaging. Each model's prediction is raised to a power (its weight) before multiplication.

### Mathematical formulation:

```
P(y=1|x) = ∏(p_i^w_i) for i = 1 to n
```

Where:
- p_i = prediction from model i
- w_i = weight for model i
- n = number of models

### Why this approach?

1. **Probabilistic interpretation**: Follows Naive Bayes assumption of conditional independence
2. **Weight semantics**: w_i = 0 excludes model (p^0 = 1), w_i = 1 gives normal contribution
3. **Non-linear combination**: Can capture complex interactions between models

### Calibration necessity:

Raw multiplication produces very small values (e.g., 0.7^5 ≈ 0.168), requiring calibration to map back to [0,1].

### Where it's used in this framework:

- **File**: `pkg/framework/trainer.go` (prediction logic)
- **Purpose**: Core ensemble combination method
- **Related**: Calibration methods to handle score distribution

## Probability Calibration Methods

### What are they?

Methods to transform model scores into well-calibrated probabilities where P(y=1|score=0.7) ≈ 0.7.

### 1. Beta Calibration (Default)

Maps scores to maintain relative ordering while adjusting distribution:

```
1. Find positive/negative class score means
2. Apply linear transformation to map means to [0.2, 0.8]
3. Clip to [0, 1]
```

**Advantages**: Preserves score distribution shape, simple and robust

### 2. Isotonic Regression

Non-parametric method that finds monotonic mapping:

```
minimize ∑(y_i - f(x_i))^2
subject to: f(x_1) ≤ f(x_2) ≤ ... ≤ f(x_n)
```

**Advantages**: Handles complex non-linear patterns, no assumptions about distribution

### 3. Platt Scaling

Fits sigmoid function to scores:

```
P(y=1|score) = 1 / (1 + exp(A*score + B))
```

**Advantages**: Simple parametric form, good for small calibration sets

### Where they're used in this framework:

- **File**: `pkg/framework/calibration.go`
- **Purpose**: Transform ensemble scores to calibrated probabilities
- **Usage**: Applied after ensemble prediction, before threshold application

## Three-Way Data Split

### What is it?

Dividing data into three disjoint sets instead of the traditional train/test split:

1. **Training Set (60%)**: Train individual models and ensemble weights
2. **Calibration Set (20%)**: Fit calibration function, prevent overfitting
3. **Test Set (20%)**: Final unbiased evaluation

### Why three-way?

1. **Prevents data leakage**: Calibration parameters not influenced by test data
2. **Unbiased threshold selection**: Find optimal threshold on calibration set
3. **Better generalization**: Each component trained on appropriate data

### Where it's used in this framework:

- **File**: `pkg/data/splitter.go`
- **Configuration**: `use_three_way_split` in DataConfig
- **Purpose**: Proper separation of training, calibration, and evaluation

## Gradient-Free Optimization

### What is it?

Optimization methods that don't require computing gradients of the objective function. Essential when:

1. Objective is non-differentiable (like PR-AUC)
2. Objective has many local optima
3. Gradient computation is expensive or noisy

### Methods in this framework:

1. **Differential Evolution**: Main optimizer
2. **Random Search**: Baseline comparison

### Advantages:

- Handles discrete metrics (precision at fixed recall)
- Works with non-convex objectives
- No need for backpropagation infrastructure

### Where it's used in this framework:

- **Files**: `pkg/optimizer/differential_evolution.go`, `pkg/optimizer/random_search.go`
- **Purpose**: Optimize weights for non-differentiable PR-AUC objective
- **Interface**: `optimizer.Optimizer` allows pluggable optimization methods

## Early Stopping with Best Weight Restoration

### What is it?

A regularization technique that:
1. Monitors validation metric during training
2. Stops when metric stops improving
3. Restores weights from best epoch

### Implementation details:

```go
type EarlyStopping struct {
    patience     int       // Epochs without improvement
    minDelta     float64   // Minimum change to be improvement
    bestScore    float64   // Track best score
    bestWeights  []float64 // Save best weights
    counter      int       // Patience counter
}
```

### Algorithm:

1. After each epoch, evaluate validation metric
2. If improvement > minDelta: save weights, reset counter
3. Else: increment counter
4. If counter >= patience: stop and restore best weights

### Where it's used in this framework:

- **File**: `pkg/framework/early_stopping.go`
- **Purpose**: Prevent overfitting, ensure best model is returned
- **Configuration**: `early_stopping` section in config

## Summary of Non-Typical Concepts

| Concept | Why Non-Typical | Advantage in This Framework |
|---------|----------------|---------------------------|
| Differential Evolution | Most ML uses gradient descent | Handles non-differentiable PR-AUC |
| PR-AUC Optimization | Usually optimize cross-entropy | Better for imbalanced data |
| Weighted Multiplication | Standard is weighted averaging | Probabilistic interpretation |
| Three-Way Split | Usually just train/test | Prevents calibration leakage |
| Beta Calibration | Often use Platt or isotonic | Preserves score distribution |
| Gradient-Free | Deep learning dominates | No backprop needed |

These concepts work together to create a framework optimized for ensemble weight learning on imbalanced binary classification tasks where precision-recall trade-offs are critical.