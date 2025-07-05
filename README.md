# Weighted Naive Bayes Training Framework for Go

A gradient-free weight optimization framework that finds the best way to combine your binary classifiers using Differential Evolution. The framework treats your model as a black box and optimizes weights to maximize your chosen metric (PR-AUC by default).

## 🚀 Key Features

- **Black Box Model Treatment**: Framework only cares about weights, not your model internals
- **Simple Interface**: Implement just 5 methods to use the framework
- **Gradient-Free Optimization**: Differential Evolution finds optimal weights without gradients
- **PR-AUC Focus**: Optimizes for precision-recall by default (configurable)
- **Calibration Comparison**: Compare multiple calibration methods to handle tiny Naive Bayes scores
- **Automatic Output**: Results, visualizations, and reports saved to `./output`
- **Cross-Validation**: Built-in k-fold and stratified sampling
- **Early Stopping**: Prevents overfitting automatically
- **HTML Reports**: Automatic generation of interactive reports with graphs

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/go_NB_Weight_Training.git
cd go_NB_Weight_Training

# Download dependencies
go mod download

# Build the modular CLI
go build -o train_modular cmd/train_modular/main.go
```

## 🏃 Quick Start

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

### 2. Use the Modular System

```bash
# Using the shell script (recommended)
./scripts/train.sh models/spam_ensemble datasets/spam_data.csv

# With custom configuration
./scripts/train.sh models/spam_ensemble datasets/spam_data.csv config/production.json

# Direct CLI usage
./train_modular -model models/spam_ensemble -data datasets/data.csv
```

### 3. Check Results

Results are automatically saved to `./output/results_YYYY-MM-DD_HH-MM-SS/`:
- `best_weights.json` - Optimized weights for your models
- `training_result.json` - Complete training metrics
- `report.html` - Interactive visualization report
- `pr_curve.png` / `roc_curve.png` - Performance curves

## 🔬 How It Works

1. **You provide**: An aggregated model that combines multiple classifiers
2. **Framework optimizes**: Weights using Differential Evolution to maximize PR-AUC
3. **You get**: Optimal weights, visualizations, and performance metrics

The framework uses weighted Naive Bayes multiplication:
```
P(y=1|x) = ∏(p_i^w_i) for i = 1 to n
```

## 🎯 Example Implementation

```go
type MySpamEnsemble struct {
    bayesFilter  *BayesModel
    neuralNet    *NeuralNetwork
    rulesEngine  *RulesEngine
    weights      []float64
}

func (e *MySpamEnsemble) Predict(samples [][]float64) ([]float64, error) {
    // Get predictions from each model
    bayes_conf := e.bayesFilter.Predict(samples)
    nn_conf := e.neuralNet.Predict(samples)
    rules_conf := e.rulesEngine.Predict(samples)
    
    // Combine using weighted Naive Bayes
    results := make([]float64, len(samples))
    for i := range results {
        results[i] = math.Pow(bayes_conf[i], e.weights[0]) *
                     math.Pow(nn_conf[i], e.weights[1]) *
                     math.Pow(rules_conf[i], e.weights[2])
    }
    return results, nil
}

// Implement other 4 methods...
```

## ✨ Calibration Comparison (New Feature!)

When using Naive Bayes multiplication, raw scores can become tiny (e.g., 0.5^8 = 0.0039). The framework now supports automatic calibration comparison:

### Optional Interface

```go
type CalibratedAggregatedModel interface {
    AggregatedModel
    PredictWithCalibration(samples [][]float64) (raw, calibrated []float64, err error)
    GetCalibrationMethod() string
}
```

### What You Get

- **Automatic comparison** of 4 calibration methods (Beta, Isotonic, Platt, Min-Max)
- **Score distribution plots** showing how each method transforms scores
- **Performance metrics** for each calibration at optimal thresholds
- **Recommendations** based on your optimization metric

See [CALIBRATION_COMPARISON.md](docs/CALIBRATION_COMPARISON.md) for details.

## ⚙️ Configuration

### Quick Configurations

- `config/quick_training.json` - Fast prototyping (50 epochs, no cross-validation)
- `config/default_config.json` - Balanced settings (100 epochs, 5-fold CV)
- `config/production_config.json` - Robust training (200 epochs, 10-fold CV)

### Key Configuration Options

```json
{
  "data_config": {
    "validation_split": 0.2,
    "k_folds": 5,
    "stratified": true
  },
  "training_config": {
    "max_epochs": 100,
    "optimization_metric": "pr_auc",
    "threshold_metric": "precision"
  },
  "optimizer_config": {
    "min_weight": 0.01,
    "max_weight": 2.0,
    "enforce_non_zero": false
  }
}
```

## 📊 Output Structure

```
./output/
└── results_2025-07-05_14-30-45/
    ├── best_weights.json                      # Optimized weights
    ├── training_result.json                   # Complete metrics
    ├── summary.txt                            # Human-readable summary
    ├── config.json                            # Configuration used
    ├── report.html                            # Interactive report
    ├── pr_curve.png                           # Precision-Recall curve
    ├── roc_curve.png                          # ROC curve
    ├── calibration_score_distributions.png    # Score distributions
    ├── calibration_comparison.png             # Method comparison
    └── calibration_pr_curves.png              # PR curves by method
```

## 🧪 Testing

```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Test calibration comparison
./scripts/test_calibration.sh
```

## 📖 Documentation

- [THEORY.md](docs/THEORY.md) - Theoretical foundations and algorithms
- [CALIBRATION_COMPARISON.md](docs/CALIBRATION_COMPARISON.md) - Calibration feature guide
- [MODULAR_USAGE.md](docs/MODULAR_USAGE.md) - Modular system usage
- [TODO.md](docs/TODO.md) - Roadmap and future enhancements

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Differential Evolution algorithm based on Storn & Price (1997)
- PR-AUC implementation inspired by scikit-learn
- Calibration methods adapted from probability calibration literature