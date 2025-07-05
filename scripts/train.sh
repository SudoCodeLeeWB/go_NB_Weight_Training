#!/bin/bash

# Weighted Naive Bayes Training Framework - Modular Training Script
# Usage: ./train.sh <model_dir> <dataset> [config]

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check arguments
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Missing required arguments${NC}"
    echo "Usage: $0 <model_dir> <dataset> [config]"
    echo ""
    echo "Examples:"
    echo "  $0 models/spam_ensemble datasets/spam_data.csv"
    echo "  $0 models/spam_ensemble datasets/spam_data.csv config/production_config.json"
    exit 1
fi

MODEL_DIR=$1
DATASET=$2
CONFIG=${3:-""}

# Validate model directory
if [ ! -d "$MODEL_DIR" ]; then
    echo -e "${RED}Error: Model directory not found: $MODEL_DIR${NC}"
    exit 1
fi

# Validate dataset
if [ ! -f "$DATASET" ]; then
    echo -e "${RED}Error: Dataset file not found: $DATASET${NC}"
    exit 1
fi

# Validate config if provided
if [ -n "$CONFIG" ] && [ ! -f "$CONFIG" ]; then
    echo -e "${RED}Error: Config file not found: $CONFIG${NC}"
    exit 1
fi

# Print header
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Weighted Naive Bayes Training Framework${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Show configuration
echo -e "${YELLOW}Configuration:${NC}"
echo "  Model:   $MODEL_DIR"
echo "  Dataset: $DATASET"
if [ -n "$CONFIG" ]; then
    echo "  Config:  $CONFIG"
else
    echo "  Config:  (using defaults)"
fi
echo ""

# Build the modular CLI if needed
echo -e "${YELLOW}Building modular CLI...${NC}"
go build -o ../train_modular ../cmd/train_modular/main.go
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Failed to build CLI${NC}"
    exit 1
fi

# Run training
echo -e "${YELLOW}Starting training...${NC}"
echo ""

if [ -n "$CONFIG" ]; then
    ../train_modular -model "$MODEL_DIR" -data "$DATASET" -config "$CONFIG"
else
    ../train_modular -model "$MODEL_DIR" -data "$DATASET"
fi

# Check if training succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}✅ Training completed successfully!${NC}"
    echo -e "${GREEN}Check the output directory for results.${NC}"
else
    echo ""
    echo -e "${RED}❌ Training failed!${NC}"
    exit 1
fi