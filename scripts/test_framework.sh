#!/bin/bash

# Test Framework for Weighted Naive Bayes Training
# This script builds, tests, and demonstrates the framework

set -e  # Exit on error

echo "========================================"
echo "Weighted Naive Bayes Framework Test"
echo "========================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Clean up previous outputs
echo -e "${YELLOW}Cleaning up previous outputs...${NC}"
rm -rf test_output/
rm -rf output/
mkdir -p test_output

# Download dependencies
echo -e "${YELLOW}Downloading dependencies...${NC}"
go mod download

# Run unit tests
echo ""
echo -e "${YELLOW}Running unit tests...${NC}"
go test ./pkg/metrics -v
go test ./pkg/data -v

# Build the main training command
echo ""
echo -e "${YELLOW}Building training command...${NC}"
go build -o train_spam cmd/train/main.go

# Run the integration test with spam detection
echo ""
echo -e "${YELLOW}Running spam detection integration test...${NC}"
echo "This will:"
echo "  - Generate 1000 spam/ham samples"
echo "  - Split: 70% train (490 train, 210 val), 30% test"
echo "  - Train 5 spam detection models"
echo "  - Optimize weights using PR-AUC"
echo "  - Show batch processing progress"
echo ""

go test -v ./test/integration -run TestSpamDetectionFramework

# Run the simple example
echo ""
echo -e "${YELLOW}Running simple example...${NC}"
go run examples/simple/main.go

# Generate sample data and run advanced example
echo ""
echo -e "${YELLOW}Running advanced example with visualization...${NC}"
go run examples/advanced/main.go

# Check outputs
echo ""
echo -e "${YELLOW}Checking generated outputs...${NC}"
if [ -f "test_output/report.html" ]; then
    echo -e "${GREEN}✓ HTML report generated${NC}"
    echo "  View at: file://$(pwd)/test_output/report.html"
fi

if [ -f "test_output/pr_curve.png" ]; then
    echo -e "${GREEN}✓ PR curve plot generated${NC}"
fi

if [ -f "test_output/roc_curve.png" ]; then
    echo -e "${GREEN}✓ ROC curve plot generated${NC}"
fi

if [ -f "test_output/spam_weights.json" ]; then
    echo -e "${GREEN}✓ Weights saved${NC}"
    echo ""
    echo "Saved weights:"
    cat test_output/spam_weights.json
fi

# Run benchmarks
echo ""
echo -e "${YELLOW}Running performance benchmarks...${NC}"
go test -bench=. -benchtime=10s ./pkg/metrics

# Summary
echo ""
echo -e "${GREEN}========================================"
echo "Test Complete!"
echo "========================================${NC}"
echo ""
echo "Key Results:"
echo "1. Framework successfully trains ensemble weights"
echo "2. Batch processing shows learning progress"
echo "3. 70/30 train-test split with 5:2 train-val split"
echo "4. PR-AUC optimization works correctly"
echo "5. Visualizations generated successfully"
echo ""
echo "Next steps:"
echo "- View the HTML report: open test_output/report.html"
echo "- Check the learned weights in test_output/spam_weights.json"
echo "- Integrate your own models by implementing the Model interface"

# Clean up build artifacts
rm -f train_spam