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
├── framework/      # Core trainer, config, model interfaces
├── optimizer/      # Differential Evolution implementation
├── metrics/        # PR-AUC, ROC-AUC calculations
├── data/          # Dataset loading, stratified splitting
├── ensemble/      # Ensemble model implementation
└── visualization/ # Plot generation with gonum/plot
```

### Key Interfaces
- `framework.Model`: Core interface for any classifier (`Predict()`, `GetName()`)
- `optimizer.Optimizer`: Interface for optimization algorithms
- Models must return probabilities (0-1) for binary classification

### Design Patterns
- **Gradient-Free Optimization**: Uses Differential Evolution instead of gradient descent
- **Naive Bayes Aggregation**: Combines predictions using `∏(prediction_i ^ weight_i)`
- **PR-AUC Focus**: Optimizes for Precision-Recall AUC without fixed thresholds
- **Modular Design**: Clean separation between data, models, optimization, and visualization

## Current State & Known Issues

### Production Readiness
- ✅ Fixed: Model persistence, probability calibration, input validation, memory efficiency
- ❌ Pending: Early stopping bug (best weights not restored)
- ⚠️ Missing: Feature preprocessing, sparse data support, monitoring hooks, distributed training

### Active Development Focus (from todos.txt)
- Critical production fixes
- API server implementation (gRPC/REST)
- Monitoring/observability (Prometheus, OpenTelemetry)
- Containerization and Kubernetes deployment
- Performance optimizations (GPU support, caching)

### Important Bug
Early stopping doesn't restore best weights. Fix needed in `trainer.go`:
```go
if t.earlyStopping != nil && t.earlyStopping.bestWeights != nil {
    result.BestWeights = t.earlyStopping.bestWeights
}
```

## Data Requirements
- Input format: CSV with classifier predictions as features
- Labels must be binary (0 or 1)
- Predictions should be probabilities between 0 and 1
- Example format:
  ```csv
  classifier1_pred,classifier2_pred,classifier3_pred,label
  0.8,0.7,0.9,1
  0.2,0.3,0.1,0
  ```

## Configuration
The framework uses a comprehensive configuration system with:
- `DataConfig`: Validation split, K-folds, stratified sampling
- `TrainingConfig`: Max epochs, batch size, optimization metric
- `OptimizerConfig`: DE algorithm parameters, weight bounds
- `EarlyStopping`: Patience, min delta, monitoring metric
- `Visualization`: Output formats, report generation

Default configuration available via `framework.DefaultConfig()`.