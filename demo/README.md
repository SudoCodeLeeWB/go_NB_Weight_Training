# Demo Programs

This directory contains demonstration programs for the Weighted Naive Bayes Training Framework.

## Available Demos

### 1. `demo.go` - Main Demo
Complete demonstration of the framework with spam detection models.

### 2. `demo_simple.go` - Simple Demo
Basic example showing minimal usage of the framework.

### 3. `demo_detailed.go` - Detailed Demo
Comprehensive example with detailed logging and visualization.

### 4. `demo_challenging.go` - Challenging Demo
Tests the framework with a more difficult dataset and model configuration.

## Running the Demos

From the project root:
```bash
# Run the main demo
cd demo
./run_demo.sh

# Or run individual demos
go run demo.go
go run demo_simple.go
go run demo_detailed.go
go run demo_challenging.go
```

## Output

Demo results are saved to `demo_output/` directory in the project root, including:
- Training results
- Model weights
- Visualization plots
- Performance metrics