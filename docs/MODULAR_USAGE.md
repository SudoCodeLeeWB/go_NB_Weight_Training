# Modular Usage Guide

This guide explains how to use the framework in a modular way with your own models.

## Directory Structure

```
go_NB_Weight_Training/
├── models/                 # Your model directories
│   └── spam_ensemble/      # Example model
│       ├── wrapper.go      # Implements ModelWrapper interface
│       ├── aggregated_model.go  # Implements AggregatedModel interface
│       └── model_config.json    # Model configuration
├── datasets/              # Your datasets (CSV or JSON)
│   ├── example_spam.csv
│   └── example_spam.json
├── config/                # Training configurations
│   ├── quick_training.json
│   ├── default_config.json
│   └── production_config.json
└── scripts/              # Shell scripts
    └── train.sh          # Main training script
```

## Quick Start

### 1. Use the Shell Script

```bash
# Basic usage
./scripts/train.sh models/spam_ensemble datasets/example_spam.csv

# With custom config
./scripts/train.sh models/spam_ensemble datasets/example_spam.json config/quick_training.json
```

### 2. Use the CLI Directly

```bash
# Build the modular CLI
go build -o train_modular cmd/train_modular/main.go

# Run training
./train_modular -model models/spam_ensemble -data datasets/example_spam.csv
```

## Creating Your Own Model

### Step 1: Create Model Directory

```
models/your_model/
├── wrapper.go           # Required: Implements ModelWrapper
├── aggregated_model.go  # Required: Implements AggregatedModel
└── model_config.json    # Required: Configuration
```

### Step 2: Implement wrapper.go

```go
package main

import (
    "github.com/iwonbin/go-nb-weight-training/pkg/framework"
)

func init() {
    framework.RegisterModelWrapper("your_model", &YourWrapper{})
}

type YourWrapper struct {
    // Your fields
}

func (w *YourWrapper) LoadModel(modelDir string) error {
    // Load your models here
    return nil
}

func (w *YourWrapper) GetAggregatedModel() framework.AggregatedModel {
    // Return your aggregated model
}

func (w *YourWrapper) GetInfo() framework.ModelInfo {
    return framework.ModelInfo{
        Name:        "Your Model",
        Version:     "1.0.0",
        Description: "Description",
        Models:      []string{"Model1", "Model2", "Model3"},
    }
}
```

### Step 3: Implement aggregated_model.go

```go
type YourAggregatedModel struct {
    // Your actual models (sklearn, tensorflow, etc.)
    model1 interface{}
    model2 interface{}
    model3 interface{}
    weights []float64
}

func (m *YourAggregatedModel) Predict(samples [][]float64) ([]float64, error) {
    // Get predictions from each model
    pred1 := m.model1.Predict(samples)  // Your logic
    pred2 := m.model2.Predict(samples)
    pred3 := m.model3.Predict(samples)
    
    // Combine using weights (Naive Bayes multiplication)
    results := make([]float64, len(samples))
    for i := range results {
        results[i] = math.Pow(pred1[i], m.weights[0]) *
                     math.Pow(pred2[i], m.weights[1]) *
                     math.Pow(pred3[i], m.weights[2])
    }
    return results, nil
}

// Implement other 4 methods...
```

### Step 4: Create model_config.json

```json
{
  "model_paths": {
    "model1": "path/to/model1.pkl",
    "model2": "path/to/model2.h5",
    "model3": "path/to/model3.json"
  },
  "initial_weights": [1.0, 1.0, 1.0]
}
```

## Dataset Formats

### CSV Format
```csv
feature1,feature2,feature3,label
0.8,0.7,0.9,1
0.2,0.3,0.1,0
```

### JSON Format
```json
{
  "features": [
    [0.8, 0.7, 0.9],
    [0.2, 0.3, 0.1]
  ],
  "labels": [1, 0],
  "metadata": {
    "feature_names": ["feature1", "feature2", "feature3"],
    "description": "My dataset"
  }
}
```

## Configuration

Training configurations control the optimization process:

```json
{
  "data_config": {
    "validation_split": 0.2,
    "k_folds": 5,
    "stratified": true
  },
  "training_config": {
    "max_epochs": 100,
    "optimization_metric": "pr_auc"
  },
  "optimizer_config": {
    "min_weight": 0.01,
    "max_weight": 2.0
  }
}
```

## Output

Results are automatically saved to `./output/results_TIMESTAMP/`:
- `best_weights.json` - Optimized weights for your models
- `report.html` - Interactive visualization
- `training_result.json` - Complete metrics
- `pr_curve.png` / `roc_curve.png` - Performance curves

## Important Notes

1. **The framework only optimizes weights** - it doesn't know about your model internals
2. **Each model must return percentage values (0-1)** from Predict()
3. **You control the aggregation logic** in your Predict() method
4. **Models can be anything** - sklearn, TensorFlow, API calls, etc.

## Example: Using Existing Models

```go
// In your aggregated_model.go
import "os/exec"

func (m *YourAggregatedModel) getSklearnPredictions(samples [][]float64) []float64 {
    // Call Python script
    cmd := exec.Command("python", "predict.py", "--model", m.modelPath)
    // ... send samples, get predictions
    return predictions
}
```

The framework treats your model as a black box and only cares about optimizing the weights!