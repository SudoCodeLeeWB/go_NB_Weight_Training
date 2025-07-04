#!/bin/bash

echo "Building and running Weighted Naive Bayes demo..."
echo ""

# Clean previous outputs
rm -rf ../demo_output/

# Run the demo from demo directory
cd "$(dirname "$0")"
go run demo.go

echo ""
echo "Demo complete! Check demo_output/ for results."