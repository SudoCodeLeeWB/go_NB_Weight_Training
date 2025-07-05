#!/bin/bash

# Test calibration comparison feature

echo "============================================"
echo "Testing Calibration Comparison Feature"
echo "============================================"
echo

# Build the modular trainer
echo "Building trainer..."
go build -o ../train_modular ../cmd/train_modular/main.go
if [ $? -ne 0 ]; then
    echo "Build failed!"
    exit 1
fi

# Create a test dataset with clear separation for calibration testing
echo "Creating test dataset..."
cat > ../test_calibration_data.csv << EOF
feature1,feature2,feature3,label
0.9,0.85,0.88,1
0.88,0.9,0.87,1
0.85,0.82,0.9,1
0.87,0.88,0.85,1
0.9,0.87,0.89,1
0.2,0.15,0.18,0
0.15,0.2,0.17,0
0.18,0.17,0.2,0
0.17,0.19,0.15,0
0.2,0.18,0.19,0
0.92,0.88,0.9,1
0.89,0.91,0.88,1
0.12,0.15,0.13,0
0.14,0.13,0.16,0
0.85,0.87,0.86,1
0.16,0.18,0.15,0
EOF

# Create a quick test config
echo "Creating test configuration..."
cat > ../test_calibration_config.json << EOF
{
  "data_config": {
    "validation_split": 0.3,
    "k_folds": 1,
    "stratified": true,
    "random_seed": 42
  },
  "training_config": {
    "max_epochs": 50,
    "optimization_metric": "pr_auc",
    "threshold_metric": "f1",
    "verbose": true
  },
  "optimizer_config": {
    "type": "differential_evolution",
    "min_weight": 0.1,
    "max_weight": 2.0,
    "population_size": 20
  },
  "visualization": {
    "enabled": true
  }
}
EOF

# Run training with calibration comparison
echo
echo "Running training with calibration comparison..."
echo "============================================"
../train_modular -model ../models/spam_ensemble -data ../test_calibration_data.csv -config ../test_calibration_config.json

# Check if output was created
if [ -d "../output" ]; then
    echo
    echo "============================================"
    echo "Training complete! Check the output directory for:"
    echo "  - Calibration score distribution plots"
    echo "  - Calibration method comparison charts"
    echo "  - HTML report with calibration analysis"
    echo "============================================"
    
    # Find the latest result directory
    LATEST_DIR=$(ls -t ../output | head -1)
    echo
    echo "Results saved in: ../output/$LATEST_DIR"
    echo
    
    # Check if calibration plots were created
    if [ -f "../output/$LATEST_DIR/calibration_score_distributions.png" ]; then
        echo "✅ Calibration comparison plots generated successfully!"
    else
        echo "⚠️  Calibration plots not found - model may not implement CalibratedAggregatedModel"
    fi
else
    echo "❌ Output directory not created"
fi

# Cleanup
echo
echo "Cleaning up test files..."
rm -f ../train_modular ../test_calibration_data.csv ../test_calibration_config.json

echo
echo "Test complete!"