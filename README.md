# Weighted Naive Bayes Training Framework for Go

A sophisticated ensemble learning framework that optimizes weights for multiple binary classifiers using PR-AUC (Precision-Recall Area Under Curve) optimization. The framework employs gradient-free optimization algorithms and combines predictions using weighted Naive Bayes multiplication.

## Features

- **PR-AUC Optimization**: Focuses on precision-recall trade-offs, ideal for imbalanced datasets
- **Gradient-Free Optimization**: Uses Differential Evolution for robust weight optimization
- **Multiple Calibration Methods**: Beta, Isotonic, Platt, and None calibration for probability adjustment
- **Three-Way Data Split**: Prevents data leakage with separate train/calibration/test sets
- **Flexible Model Interface**: Easy integration of custom binary classifiers
- **Cross-Validation Support**: K-fold and stratified sampling for robust evaluation
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Comprehensive Visualization**: Generates PR/ROC curves and HTML reports
- **Production Ready**: Includes model persistence, batch processing, and extensive validation

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

### 1. Basic Training

```bash
# Train with default configuration
go run cmd/train/main.go -data your_data.csv

# Or use the built binary
./train_ensemble -data your_data.csv
```

### 2. Advanced Training with Configuration

```bash
# Use production configuration
go run cmd/train/main.go \
  -data your_data.csv \
  -config config/production_config.json \
  -output ./results
```

### 3. Run Examples

```bash
# Simple example
go run examples/simple/main.go

# Advanced example with cross-validation
go run examples/advanced/main.go

# Production-ready example
go run examples/production/main.go

# Run all demos
./demo/run_demo.sh
```

## Data Format

The framework expects CSV data with features and binary labels:

```csv
feature1,feature2,feature3,label
0.8,0.7,0.9,1
0.2,0.3,0.1,0
```

**Important**: 
- Features are raw input features, NOT classifier predictions
- Labels must be binary (0 or 1)
- Models implementing the Model interface process these features

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

## CLI Usage

```bash
# Basic usage
./train_ensemble -data data.csv

# Full options
./train_ensemble \
  -data data.csv \
  -config config.json \
  -output results/ \
  -verbose

# Flags:
#   -data    Path to training data (required)
#   -config  Path to configuration file
#   -output  Output directory (default: ./output)
#   -verbose Enable verbose logging (default: true)
```

## Output

The framework generates:
- `ensemble_weights.json`: Trained model weights
- `training_results.json`: Detailed training metrics
- `training_summary.txt`: Human-readable summary
- `pr_curve.png`: Precision-Recall curve
- `roc_curve.png`: ROC curve
- `report.html`: Comprehensive HTML report

## Creating Custom Models

Implement the `Model` interface:

```go
type Model interface {
    Predict(features []float64) (float64, error)
    GetName() string
}

// Optional extended interface
type ExtendedModel interface {
    Model
    Train(X [][]float64, y []int) error
    Save(path string) error
    Load(path string) error
}
```

See `models/example_model.go` for a template.

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