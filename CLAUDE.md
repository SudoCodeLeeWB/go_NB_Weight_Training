# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Weighted Naive Bayes Training Framework for Go that optimizes ensemble weights for multiple binary classifiers using PR-AUC (Precision-Recall Area Under Curve) optimization. The framework uses gradient-free optimization (Differential Evolution) and combines predictions using weighted multiplication: `∏(prediction_i ^ weight_i)`.

## Key Commands

### Testing & Development
```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run specific package tests
go test ./pkg/metrics -v
go test ./pkg/data -v

# Run benchmarks
go test -bench=. -benchtime=10s ./pkg/metrics

# Download dependencies
go mod download
```

### Building & Running
```bash
# Build CLI
go build -o train_spam cmd/train/main.go

# Run CLI training
go run cmd/train/main.go -data predictions.csv
go run cmd/train/main.go -data predictions.csv -config config.json -output ./results

# Run demos
./run_demo.sh
./test_framework.sh

# Run examples
go run examples/simple/main.go
go run examples/advanced/main.go
go run demo.go
```

## Architecture

### Core Structure
```
pkg/
├── framework/      # Core trainer, config, AggregatedModel interface
├── optimizer/      # Differential Evolution implementation
├── metrics/        # PR-AUC, ROC-AUC calculations
├── data/          # Dataset loading, stratified splitting
└── visualization/ # Plot generation with gonum/plot
```

### Key Interface - AggregatedModel
The framework only knows about ONE interface:
```go
type AggregatedModel interface {
    Predict(samples [][]float64) ([]float64, error)
    GetWeights() []float64
    SetWeights(weights []float64) error
    GetNumModels() int
    GetModelNames() []string  // Optional, for reporting
}
```

### Design Philosophy
- **Black Box Approach**: Framework treats your model as a black box
- **Weight Optimization Only**: Framework ONLY optimizes weights, nothing else
- **User Freedom**: You implement aggregation however you want
- **Gradient-Free Optimization**: Uses Differential Evolution
- **PR-AUC Focus**: Optimizes for Precision-Recall AUC by default

## Current State & Known Issues

### Production Readiness
- ✅ Fixed: Early stopping, model persistence, probability calibration, input validation, memory efficiency
- ✅ Added: Three-way split for calibration, multiple calibration methods, PR-specific threshold optimization
- ✅ Added: Single prediction API, caching support, weight enforcement options
- ⚠️ API server removed per user request (local use only)

### Recent Major Improvements
1. **Early Stopping Fixed**: Now properly saves and restores best weights
2. **Calibration System**: Multiple methods (beta, isotonic, platt, none) to handle score distributions
3. **Three-Way Split**: Prevents data leakage (train/calibration/test)
4. **Threshold Optimization**: PR-specific metrics (precision, MCC, PR-distance)
5. **Weight Control**: Option to prevent model exclusion (enforce_non_zero)

## How It Works

1. **You Implement AggregatedModel**: Create your ensemble with any internal structure
2. **Framework Optimizes Weights**: Uses Differential Evolution to find best weights
3. **You Control Aggregation**: Implement Predict() however you want (Naive Bayes, voting, etc.)

Example:
```go
type MyEnsemble struct {
    // Your internal models - framework doesn't see these
    model1, model2, model3 interface{}
    weights []float64
}

func (m *MyEnsemble) Predict(samples [][]float64) ([]float64, error) {
    // YOUR aggregation logic (e.g., weighted Naive Bayes)
    // Get predictions from your models
    // Combine using weights: ∏(prediction_i ^ weight_i)
}

// Train
result, _ := framework.TrainAggregatedModel(dataset, myEnsemble, config)
```

## Data Requirements
- Labels must be binary (0 or 1)
- Features: Whatever your internal models expect
- Example format:
  ```csv
  feature1,feature2,feature3,label
  0.8,0.7,0.9,1
  0.2,0.3,0.1,0
  ```

## Configuration

### Core Configuration
- `DataConfig`: 
  - `validation_split`: Train/test split ratio
  - `k_folds`: Number of cross-validation folds (1 = simple split)
  - `use_three_way_split`: Enable train/calibration/test split
  - `calibration_split`: Size of calibration set (with three-way)
- `TrainingConfig`:
  - `optimization_metric`: "pr_auc" (default), "roc_auc", "precision", "recall"
  - `enable_calibration`: Apply probability calibration
  - `calibration_method`: "beta" (default), "isotonic", "platt", "none"
  - `threshold_metric`: "precision" (default), "f1", "recall", "mcc", "pr_distance"
- `OptimizerConfig`:
  - `min_weight`: Minimum weight (0.01 default to avoid model exclusion)
  - `max_weight`: Maximum weight (2.0 default)
  - `enforce_non_zero`: Prevent models from being excluded (false default)
- `EarlyStopping`: 
  - `patience`: Number of epochs without improvement before stopping
  - `monitor`: Metric to monitor (e.g., "val_pr_auc")

### Calibration Methods
1. **Beta** (default): Preserves score distribution, maps class means to [0.2, 0.8]
2. **Isotonic**: Non-parametric, handles complex patterns
3. **Platt**: Sigmoid transformation (can be too aggressive)
4. **None**: Simple min-max scaling

### Weight Initialization & Zero Weights
- Initial weights: Random from [min_weight, max_weight]
- Zero weight means model^0 = 1 (model excluded)
- Set `enforce_non_zero: true` to keep all models active
- Optimizer may set weights to 0 if model hurts performance

Default configuration: `framework.DefaultConfig()`

## Key Improvements Summary

### 1. Calibration System
- **Problem**: Naive Bayes multiplication creates tiny scores (e.g., 0.5^8 = 0.0039)
- **Solution**: Multiple calibration methods to map scores to proper probabilities
- **Result**: Threshold 0.5 becomes meaningful, predictions spread across [0,1]

### 2. Three-Way Split
- **Problem**: Data leakage when calibrating and finding threshold on same validation set
- **Solution**: Train (60%) → Calibration (20%) → Test (20%)
- **Result**: Unbiased threshold selection, better generalization

### 3. PR-Specific Threshold Optimization
- **Problem**: Default 0.5 threshold often suboptimal for imbalanced data
- **Solution**: Find optimal threshold based on precision, F1, MCC, or PR-distance
- **Result**: Better precision-recall trade-offs for production use

### 4. Weight Control
- **Problem**: Models might be excluded (weight=0) when needed for business reasons
- **Solution**: `enforce_non_zero` option to keep all models active
- **Result**: Full control over ensemble composition

### 5. Results Output
- Shows calibration method used
- Reports optimal threshold and metric used
- Displays performance at optimal threshold
- Saves all information in JSON and summary files
