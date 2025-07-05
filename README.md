# Weighted Naive Bayes Training Framework for Go

A weight optimization framework that finds the best way to combine your binary classifiers. The framework treats your model as a black box and uses gradient-free optimization (Differential Evolution) to find optimal weights that maximize your chosen metric (PR-AUC by default).

## Key Features

- **Black Box Model Treatment**: Framework only cares about weights, not your model internals
- **Simple Interface**: Implement just 5 methods to use the framework
- **Gradient-Free Optimization**: Differential Evolution finds optimal weights without gradients
- **PR-AUC Focus**: Optimizes for precision-recall by default (configurable)
- **Automatic Output**: Results, visualizations, and reports saved to `./output`
- **Cross-Validation**: Built-in k-fold and stratified sampling
- **Early Stopping**: Prevents overfitting automatically
- **HTML Reports**: Automatic generation of interactive reports with graphs

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/go_NB_Weight_Training.git
cd go_NB_Weight_Training

# Download dependencies
go mod download

# Build the CLI tool
go build -o train_ensemble cmd/train/main.go
```

## Quick Start

### 1. Implement the AggregatedModel Interface

```go
type AggregatedModel interface {
    Predict(samples [][]float64) ([]float64, error)
    GetWeights() []float64
    SetWeights(weights []float64) error
    GetNumModels() int
    GetModelNames() []string
}
```

### 2. Train Your Model

```go
// Your implementation
ensemble := &MyCustomEnsemble{...}

// Train with framework
result, err := framework.TrainAggregatedModel(dataset, ensemble, config)

// Your model now has optimized weights!
```

### 3. Run Examples

```bash
# See how to implement AggregatedModel
go run examples/custom_aggregated_model/main.go

# Simple example with mock models
go run examples/simple/main.go

# Run interactive demo
./demo/run_demo.sh
```

## How It Works

1. **You Control Everything**: The framework doesn't know or care about your model structure
2. **Weight Optimization**: Framework tests different weight combinations to maximize performance
3. **Black Box Approach**: Your `Predict()` method can do anything - Naive Bayes, voting, stacking, etc.

## Data Format

Your data format is up to you! The framework just needs:
- Binary labels (0 or 1)
- Features that your model understands

Example CSV:
```csv
feature1,feature2,feature3,label
0.8,0.7,0.9,1
0.2,0.3,0.1,0
```

## Configuration

### Configuration Structure

```json
{
  "data_config": {
    "validation_split": 0.2,
    "k_folds": 5,
    "stratified": true,
    "use_three_way_split": true,
    "calibration_split": 0.25
  },
  "training_config": {
    "max_epochs": 100,
    "optimization_metric": "pr_auc",
    "enable_calibration": true,
    "calibration_method": "beta",
    "threshold_metric": "precision"
  },
  "optimizer_config": {
    "type": "differential_evolution",
    "min_weight": 0.01,
    "max_weight": 2.0,
    "enforce_non_zero": false
  },
  "early_stopping": {
    "enabled": true,
    "patience": 10,
    "min_delta": 0.001,
    "monitor": "val_pr_auc"
  }
}
```

### Pre-configured Profiles

1. **Quick Training** (`config/quick_training.json`)
   - Fast prototyping with simple train/test split
   - 50 epochs, no cross-validation

2. **Default** (`config/default_config.json`)
   - Balanced settings with 5-fold cross-validation
   - 100 epochs, early stopping

3. **Production** (`config/production_config.json`)
   - Robust configuration with 10-fold cross-validation
   - 200 epochs, three-way split, comprehensive validation

## Architecture

```
pkg/
├── framework/      # Core training framework
│   ├── trainer.go      # Main training orchestrator
│   ├── config.go       # Configuration management
│   ├── model.go        # Model interfaces
│   └── persistence.go  # Model saving/loading
├── optimizer/      # Optimization algorithms
│   ├── differential_evolution.go
│   └── random_search.go
├── metrics/        # Evaluation metrics
│   ├── metrics.go      # PR-AUC, ROC-AUC calculations
│   └── threshold.go    # Optimal threshold finding
├── data/          # Data handling
│   ├── loader.go       # CSV data loading
│   ├── splitter.go     # Stratified splitting
│   └── generators/     # Data generation utilities
├── ensemble/      # Ensemble implementation
│   └── weighted_ensemble.go
└── visualization/ # Plotting and reports
    ├── plots.go        # PR/ROC curve generation
    └── report.go       # HTML report generation
```

## Key Concepts

### 1. Weighted Naive Bayes Aggregation
Combines model predictions using: `∏(prediction_i ^ weight_i)`

### 2. Calibration Methods
- **Beta**: Preserves distribution, maps class means to [0.2, 0.8]
- **Isotonic**: Non-parametric, handles complex patterns
- **Platt**: Sigmoid transformation
- **None**: Simple min-max scaling

### 3. Three-Way Split
- Train (60%): Model training
- Calibration (20%): Probability calibration
- Test (20%): Final evaluation

### 4. Weight Control
- Zero weight excludes a model (model^0 = 1)
- `enforce_non_zero`: Keeps all models active
- Weight bounds: [min_weight, max_weight]

## Testing

```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run specific package tests
go test ./pkg/metrics -v

# Run benchmarks
go test -bench=. -benchtime=10s ./pkg/metrics

# Run integration tests
go test ./test/integration/... -v

# Run comprehensive test suite
./scripts/test_framework.sh
```

## Example Implementation

```go
type MySpamEnsemble struct {
    spamDetector  *MySpamDetector  // Your models
    neuralNet     *MyNeuralNet     // Framework doesn't
    randomForest  *MyRandomForest  // see these
    weights       []float64
}

func (e *MySpamEnsemble) Predict(samples [][]float64) ([]float64, error) {
    // Get predictions from each model
    spam_conf := e.spamDetector.Predict(samples)
    nn_conf := e.neuralNet.Predict(samples)
    rf_conf := e.randomForest.Predict(samples)
    
    // Combine using weights (Naive Bayes multiplication)
    results := make([]float64, len(samples))
    for i := range results {
        results[i] = math.Pow(spam_conf[i], e.weights[0]) *
                     math.Pow(nn_conf[i], e.weights[1]) *
                     math.Pow(rf_conf[i], e.weights[2])
    }
    return results, nil
}

// Implement other 4 methods...
```

## Output Structure

All outputs are automatically saved to `./output/results_YYYY-MM-DD_HH-MM-SS/`:

```
./output/
└── results_2025-07-05_14-30-45/
    ├── best_weights.json        # Optimized weights for your models
    ├── training_result.json     # Complete training metrics
    ├── summary.txt              # Human-readable summary
    ├── config.json              # Configuration used
    ├── report.html             # Interactive HTML report
    ├── pr_curve.png            # Precision-Recall curve
    └── roc_curve.png           # ROC curve
```

## Why Use This Framework?

1. **You Already Have Models**: Don't retrain - optimize how to combine them
2. **Black Box Approach**: Use models from any source (sklearn, TensorFlow, custom)
3. **Automatic Weight Optimization**: Let the framework find the best combination
4. **Production Ready**: Get optimized weights you can deploy immediately

## Production Considerations

### ✅ Production Ready Features
- Model persistence and loading
- Probability calibration
- Input validation
- Memory-efficient batch processing
- Early stopping with best weight restoration
- Comprehensive error handling

### ⚠️ Limitations
- Local use only (no API server)
- Single-machine processing
- Binary classification only

## Examples

See the `examples/` directory for:
- `simple/`: Basic usage
- `advanced/`: Cross-validation and advanced features
- `production/`: Production-ready implementation
- `calibration_demo.go`: Calibration methods demonstration
- `precision_analysis.go`: Precision optimization

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [gonum](https://www.gonum.org/) for numerical computations
- Visualization powered by [gonum/plot](https://github.com/gonum/plot)
- Inspired by ensemble learning and Naive Bayes theory