# Weighted Naive Bayes Training Framework

A Go framework for training weighted naive bayes ensemble models with PR-AUC optimization. This framework is designed to find optimal weights for combining predictions from multiple binary classifiers using a naive bayes aggregation approach.

## Features

- **PR-AUC Optimization**: Optimizes weights to maximize Precision-Recall AUC without setting a fixed threshold
- **Gradient-Free Optimization**: Uses Differential Evolution algorithm for robust weight optimization
- **Stratified Cross-Validation**: Maintains class balance across folds for reliable evaluation
- **Early Stopping**: Prevents overfitting with configurable patience and monitoring
- **Comprehensive Metrics**: Calculates PR-AUC, ROC-AUC, precision, recall, and F1-score
- **Visualization**: Generates PR curves, ROC curves, and HTML reports with gonum/plot
- **Modular Design**: Clean architecture with interfaces for easy extension
- **Production Ready**: Comprehensive test suite and benchmarks

## Installation

```bash
go get github.com/iwonbin/go-nb-weight-training
```

## Quick Start

### 1. Implement Your Model

To use this framework, you need to implement the `Model` interface for each of your classifiers:

```go
package main

import "github.com/iwonbin/go-nb-weight-training/pkg/framework"

// Step 1: Define your model struct
type MyLogisticRegression struct {
    // Your model parameters
    coefficients []float64
    intercept    float64
}

// Step 2: Implement the Predict method
// Input: samples [][]float64 - each row is a feature vector
// Output: []float64 - probability scores (0-1) for positive class
func (m *MyLogisticRegression) Predict(samples [][]float64) ([]float64, error) {
    predictions := make([]float64, len(samples))
    
    for i, features := range samples {
        // Your prediction logic here
        // Example: logistic regression
        logit := m.intercept
        for j, coef := range m.coefficients {
            if j < len(features) {
                logit += coef * features[j]
            }
        }
        // Convert to probability
        predictions[i] = 1.0 / (1.0 + math.Exp(-logit))
    }
    
    return predictions, nil
}

// Step 3: Implement GetName method
func (m *MyLogisticRegression) GetName() string {
    return "LogisticRegression"
}
```

### 2. Prepare Your Data

Your CSV file should contain raw features (NOT predictions) with a binary label:

```csv
feature1,feature2,feature3,feature4,label
5.2,3.1,4.5,1.8,1
2.1,1.2,1.3,0.4,0
6.7,3.0,5.2,2.3,1
1.5,0.8,1.1,0.3,0
```

### 3. Train the Ensemble

```go
package main

import (
    "fmt"
    "log"
    
    "github.com/iwonbin/go-nb-weight-training/pkg/framework"
    "github.com/iwonbin/go-nb-weight-training/pkg/data"
)

func main() {
    // Step 1: Load your data
    loader := data.NewCSVLoader()
    dataset, err := loader.Load("train_data.csv")
    if err != nil {
        log.Fatal(err)
    }
    
    // Step 2: Create your pre-trained models
    models := []framework.Model{
        &MyLogisticRegression{
            // Initialize with your trained parameters
            coefficients: []float64{0.5, -0.3, 0.8, 0.2},
            intercept:    -0.1,
        },
        &MyRandomForest{
            // Your random forest implementation
        },
        &MyNeuralNetwork{
            // Your neural network implementation
        },
    }
    
    // Step 3: Configure training (or load from config file)
    config := framework.DefaultConfig()
    config.TrainingConfig.MaxEpochs = 100
    config.DataConfig.ValidationSplit = 0.2
    config.DataConfig.KFolds = 1  // Use simple split
    
    // Enable early stopping
    config.EarlyStopping = &framework.EarlyStoppingConfig{
        Patience: 5,      // Stop after 5 epochs with no improvement
        MinDelta: 0.001,  // Minimum improvement threshold
        Monitor:  "pr_auc",
        Mode:     "max",
    }
    
    // Step 4: Train to find optimal weights
    trainer := framework.NewTrainer(config)
    result, err := trainer.Train(dataset, models)
    if err != nil {
        log.Fatal(err)
    }
    
    // Step 5: Display results
    fmt.Printf("Training completed!\n")
    fmt.Printf("Number of models: %d\n", len(models))
    fmt.Printf("Best weights found: %v\n", result.BestWeights)
    fmt.Printf("Final PR-AUC: %.4f\n", result.FinalMetrics["pr_auc"])
    
    // Step 6: Create the final ensemble for predictions
    ensemble := &framework.EnsembleModel{
        Models:  models,
        Weights: result.BestWeights,
    }
    
    // Make predictions on new data
    newSamples := [][]float64{
        {5.1, 3.0, 4.7, 1.9},
        {2.0, 1.1, 1.2, 0.5},
    }
    predictions, _ := ensemble.Predict(newSamples)
    fmt.Printf("Predictions: %v\n", predictions)
}
```

### 4. Using Configuration Files

Instead of hardcoding configuration, you can load from JSON files in the `config/` directory:

```go
// Load configuration from file
config, err := framework.LoadConfig("config/training_config.json")
if err != nil {
    log.Fatal(err)
}

// Train with loaded config
trainer := framework.NewTrainer(config)
result, _ := trainer.Train(dataset, models)
```

## Configuration

The framework provides extensive configuration options:

```go
config := &framework.Config{
    DataConfig: framework.DataConfig{
        ValidationSplit: 0.2,    // Train-validation split ratio
        KFolds:          5,      // Number of CV folds
        Stratified:      true,   // Use stratified splitting
        RandomSeed:      42,     // For reproducibility
    },
    TrainingConfig: framework.TrainingConfig{
        MaxEpochs:          100,      // Maximum optimization iterations
        OptimizationMetric: "pr_auc", // Metric to optimize
        Verbose:            true,      // Enable logging
    },
    OptimizerConfig: framework.OptimizerConfig{
        Type:           "differential_evolution",
        PopulationSize: 50,
        MutationFactor: 0.8,
        CrossoverProb:  0.9,
        MinWeight:      0.0,
        MaxWeight:      2.0,
    },
    EarlyStopping: &framework.EarlyStoppingConfig{
        Patience: 10,
        MinDelta: 0.001,
        Monitor:  "val_pr_auc",
        Mode:     "max",
    },
    Visualization: framework.VisualizationConfig{
        Enabled:        true,
        OutputDir:      "./output",
        GenerateReport: true,
    },
}
```

## How It Works

### Overview

This framework finds optimal weights for combining multiple pre-trained classifiers using a Weighted Naive Bayes approach. Instead of training the models themselves, it learns how much to "trust" each model's predictions.

### Data Flow

```
Raw Features → Model 1 → Prediction 1 ─┐
             → Model 2 → Prediction 2 ─┼─→ Weighted Ensemble → Final Score
             → Model 3 → Prediction 3 ─┘   (using learned weights)
```

### The Training Process

#### 1. **What Gets Trained**
- **Input**: Multiple pre-trained models that can make predictions
- **Output**: Optimal weight for each model
- **Formula**: `final_score = prediction1^weight1 × prediction2^weight2 × prediction3^weight3`

#### 2. **Differential Evolution Algorithm**

The framework uses an evolutionary algorithm to find optimal weights:

**Population Initialization**
```
Population = [
    [0.5, 1.2, 0.8],  // Candidate 1: weights for 3 models
    [0.7, 0.9, 1.1],  // Candidate 2
    [1.0, 0.6, 1.3],  // Candidate 3
    ...               // 50 candidates total
]
```

**Evolution Process**
For each generation:
1. **Mutation**: For each candidate, create a mutant by combining other candidates
   ```
   Select 3 random candidates: r1, r2, r3
   mutant = r1 + 0.8 × (r2 - r3)
   ```

2. **Crossover**: Mix mutant with original candidate
   ```
   For each weight position:
   - 90% chance: use mutant value
   - 10% chance: keep original value
   ```

3. **Selection**: Evaluate both candidates
   ```
   If mutant performs better (higher PR-AUC):
       keep mutant
   Else:
       keep original
   ```

#### 3. **Example Evolution Step**

```
Current weights: [0.5, 1.2, 0.8]
After mutation:  [0.7, 0.9, 1.1]

Evaluation:
- Current: PR-AUC = 0.85
- Mutant:  PR-AUC = 0.87  ← Better! Replace current with mutant

Next generation starts with [0.7, 0.9, 1.1]
```

#### 4. **Optimization Objective**

The algorithm optimizes for PR-AUC (Precision-Recall Area Under Curve):
1. Apply current weights to ensemble
2. Make predictions on validation data
3. Calculate PR-AUC score
4. Higher score = better weight combination

#### 5. **Convergence**

Training stops when:
- Maximum iterations reached (default: 100)
- No improvement for 10 iterations
- Early stopping triggered (no improvement for patience epochs)

### Input Data Format

Your CSV should contain **raw features**, not predictions:

```csv
feature1,feature2,feature3,feature4,label
5.0,2.0,3.0,0.8,1
1.0,8.0,0.0,0.1,0
7.0,1.0,5.0,0.9,1
```

Each model receives these features and produces its own predictions, which are then combined using the learned weights.

## Visualization

The framework generates comprehensive visualizations:

- **PR Curve**: Precision-Recall curve with AUC value
- **ROC Curve**: ROC curve with AUC value  
- **Weight Distribution**: Bar chart showing learned weights
- **Training History**: Metric evolution over epochs
- **HTML Report**: Complete training summary with all metrics and plots

## Examples

See the `examples/` directory for complete examples:

- `simple/`: Basic usage with minimal configuration
- `advanced/`: Advanced features including cross-validation and custom callbacks

## Command Line Usage

```bash
# Basic training
go run cmd/train/main.go -data predictions.csv

# With custom configuration
go run cmd/train/main.go -data predictions.csv -config config.json -output ./results

# Flags:
#   -data:    Path to training data (required)
#   -config:  Path to configuration file (optional)
#   -output:  Output directory for results (default: ./output)
#   -verbose: Enable verbose logging (default: true)
```

## Architecture

```
├── pkg/
│   ├── framework/      # Core training framework
│   ├── optimizer/      # Weight optimization algorithms
│   ├── metrics/        # PR-AUC, ROC-AUC implementations
│   ├── data/          # Data loading and splitting
│   └── visualization/ # Plotting and reporting
├── cmd/               # CLI applications
├── models/            # Example model implementations
└── examples/          # Usage examples
```

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run benchmarks
go test -bench=. ./...
```

## License

MIT License - see LICENSE file for details

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{weighted_naive_bayes_go,
  title = {Weighted Naive Bayes Training Framework for Go},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/iwonbin/go-nb-weight-training}
}
```