# Project Structure

## Directory Layout

```
go_NB_Weight_Training/
├── cmd/                    # Command-line applications
│   └── train/             # Training CLI tool
├── config/                # Configuration files
│   ├── default_config.json
│   ├── quick_training.json
│   └── production_config.json
├── demo/                  # Demo programs
│   ├── demo.go           # Main demo
│   ├── demo_simple.go    # Simple example
│   ├── demo_detailed.go  # Detailed example
│   └── demo_challenging.go # Challenging scenarios
├── examples/              # Usage examples
│   ├── simple/           # Basic usage
│   ├── advanced/         # Advanced features
│   └── production/       # Production-ready example
├── models/               # Example model implementations
│   ├── spam_detector.go  # Spam detection models
│   └── example_model.go  # Template for new models
├── pkg/                  # Core packages
│   ├── data/            # Data loading and splitting
│   ├── framework/       # Core training framework
│   ├── metrics/         # Evaluation metrics
│   ├── optimizer/       # Optimization algorithms
│   └── visualization/   # Plotting and reporting
├── scripts/             # Utility scripts
│   └── test_framework.sh # Testing script
├── test/                # Test files
│   ├── integration/     # Integration tests
│   ├── benchmarks/      # Performance benchmarks
│   └── fixtures/        # Test data
├── docs/                # Documentation
├── CLAUDE.md           # AI assistant guidance
├── README.md           # Project documentation
├── TODO.md             # Future enhancements
└── go.mod              # Go module file
```

## Key Components

### Core Framework (`pkg/framework/`)
- `trainer.go` - Main training orchestrator
- `model.go` - Model interfaces
- `config.go` - Configuration management
- `validation.go` - Input validation
- `cache.go` - Prediction caching
- `result_writer.go` - Result persistence

### Data Management (`pkg/data/`)
- `loader.go` - CSV data loading
- `splitter.go` - Train/test splitting
- `types.go` - Data structures

### Optimization (`pkg/optimizer/`)
- `differential_evolution.go` - Main optimizer
- `random_search.go` - Alternative optimizer

### Metrics (`pkg/metrics/`)
- `pr_auc.go` - Precision-Recall AUC
- `roc_auc.go` - ROC AUC
- `metrics.go` - Common metrics

## Usage Flow

1. Implement the `Model` interface for your classifiers
2. Load data using `data.LoadData()`
3. Configure training with `framework.Config`
4. Train ensemble with `trainer.Train()`
5. Results saved to timestamped directories
6. Use learned weights for predictions