# Project Structure

This document provides an overview of the Weighted Naive Bayes Training Framework directory structure.

```
go_NB_Weight_Training/
│
├── cmd/                        # Command-line applications
│   └── train/                  # Main CLI training tool
│       └── main.go            # Entry point for ensemble training CLI
│
├── pkg/                        # Core library packages
│   ├── data/                   # Data handling and processing
│   │   ├── loader.go          # CSV data loading functionality
│   │   ├── loader_test.go     # Tests for data loader
│   │   ├── splitter.go        # Data splitting (train/val/test, k-fold)
│   │   ├── splitter_test.go   # Tests for data splitter
│   │   ├── types.go           # Data structure definitions
│   │   └── generators/        # Synthetic data generation
│   │       └── complex_generator.go  # Complex pattern data generator
│   │
│   ├── framework/              # Core training framework
│   │   ├── trainer.go         # Main training orchestrator
│   │   ├── config.go          # Configuration structures and defaults
│   │   ├── model.go           # Model interface definition
│   │   ├── model_extended.go  # Extended model interface (train/save/load)
│   │   ├── calibration.go     # Probability calibration methods
│   │   ├── persistence.go     # Model persistence (save/load)
│   │   ├── result_writer.go   # Training results output
│   │   ├── batch_processor.go # Batch processing for memory efficiency
│   │   ├── cache.go           # Prediction caching
│   │   ├── callbacks.go       # Training callbacks interface
│   │   ├── errors.go          # Custom error types
│   │   ├── validation.go      # Input validation
│   │   └── weights.go         # Weight management utilities
│   │
│   ├── metrics/                # Evaluation metrics
│   │   ├── metrics.go         # Core metrics implementation
│   │   ├── metrics_test.go    # Tests for metrics
│   │   ├── pr_auc.go          # Precision-Recall AUC calculation
│   │   └── roc_auc.go         # ROC AUC calculation
│   │
│   ├── optimizer/              # Optimization algorithms
│   │   ├── optimizer.go       # Optimizer interface
│   │   ├── differential_evolution.go  # DE algorithm implementation
│   │   └── random_search.go   # Random search baseline
│   │
│   └── visualization/          # Plotting and reporting
│       ├── plotter.go         # PR/ROC curve generation
│       └── report.go          # HTML report generation
│
├── models/                     # Example model implementations
│   ├── example_model.go       # Template for creating new models
│   └── spam_detector.go       # Spam detection model examples
│
├── examples/                   # Usage examples
│   ├── simple/                # Basic usage example
│   │   └── main.go
│   ├── advanced/              # Advanced features (cross-validation)
│   │   └── main.go
│   ├── production/            # Production-ready example
│   │   └── main.go
│   ├── complex_realistic/     # Complex real-world scenario
│   │   └── main.go
│   ├── realistic/             # Realistic use case
│   │   └── main.go
│   ├── improved_test/         # Improved testing example
│   │   └── main.go
│   ├── enforce_weights_test/  # Weight enforcement example
│   │   └── main.go
│   ├── overfitting_analysis/  # Overfitting analysis
│   │   └── main.go
│   ├── calibration_demo.go    # Calibration methods demo
│   └── precision_analysis.go  # Precision optimization demo
│
├── demo/                       # Demonstration programs
│   ├── demo.go                # Main demo program
│   ├── demo_simple.go         # Simple demonstration
│   ├── demo_detailed.go       # Detailed demo with explanations
│   ├── demo_challenging.go    # Challenging scenarios demo
│   ├── run_demo.sh           # Script to run all demos
│   └── README.md             # Demo documentation
│
├── config/                     # Configuration examples
│   ├── default_config.json    # Standard balanced configuration
│   ├── quick_training.json    # Fast prototyping configuration
│   ├── production_config.json # Robust production configuration
│   ├── calibration_examples.json  # Calibration-specific configs
│   └── README.md             # Configuration documentation
│
├── test/                       # Test suite
│   ├── integration/           # End-to-end integration tests
│   │   └── spam_test.go      # Spam detection integration test
│   ├── cache_test.go         # Cache functionality tests
│   ├── early_stopping_test.go # Early stopping tests
│   ├── model_extended_test.go # Extended model interface tests
│   ├── result_writer_test.go  # Result writer tests
│   ├── validation_test.go     # Validation tests
│   └── README.md             # Test documentation
│
├── scripts/                    # Utility scripts
│   └── test_framework.sh      # Comprehensive test runner
│
├── .claude/                    # Claude AI settings
│   └── settings.local.json    # Local Claude settings
│
├── docs/                       # Additional documentation
│
├── .gitignore                 # Git ignore rules
├── go.mod                     # Go module definition
├── go.sum                     # Go module checksums
├── README.md                  # Main project documentation
├── CLAUDE.md                  # Claude AI guidance
├── theory.md                  # Theoretical foundations
├── PROJECT_STRUCTURE.md       # This file
├── TODO.md                    # Future enhancements
└── PRODUCTION_ISSUES.md       # Known issues and fixes

```

## Directory Purposes

### Core Packages (`pkg/`)

- **data/**: Handles all data-related operations including loading CSV files, splitting datasets, and generating synthetic data
- **framework/**: Contains the core training logic, model interfaces, configuration management, and supporting utilities
- **metrics/**: Implements evaluation metrics with focus on PR-AUC for imbalanced datasets
- **optimizer/**: Provides gradient-free optimization algorithms, primarily Differential Evolution
- **visualization/**: Generates plots and HTML reports for results visualization

### Application Code

- **cmd/train/**: Command-line interface for training ensemble models
- **models/**: Example implementations of the Model interface
- **examples/**: Various usage examples from simple to production-ready
- **demo/**: Demonstration programs showing framework capabilities

### Configuration and Testing

- **config/**: Pre-configured JSON files for different use cases
- **test/**: Comprehensive test suite including unit and integration tests
- **scripts/**: Utility scripts for testing and automation

### Documentation

- **README.md**: Main project documentation and usage guide
- **CLAUDE.md**: Instructions for Claude AI when working with the codebase
- **theory.md**: Detailed explanations of theoretical concepts
- **PROJECT_STRUCTURE.md**: This structural overview
- **TODO.md**: Roadmap and completed features
- **PRODUCTION_ISSUES.md**: Known issues and their resolutions

## Key Files

1. **pkg/framework/trainer.go**: Core training orchestrator
2. **pkg/optimizer/differential_evolution.go**: Main optimization algorithm
3. **pkg/metrics/metrics.go**: Metric calculations including PR-AUC
4. **cmd/train/main.go**: CLI entry point
5. **pkg/framework/calibration.go**: Probability calibration implementations

## Output Directories (Git-ignored)

- **output/**: Default output directory for training results
- **demo_output/**: Output from demo programs
- ***/output/**: Various example-specific output directories

These directories are created during runtime and contain generated files like:
- ensemble_weights.json
- training_results.json
- pr_curve.png
- roc_curve.png
- report.html