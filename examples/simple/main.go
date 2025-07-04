package main

import (
	"fmt"
	"log"

	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/models"
)

func main() {
	fmt.Println("=== Simple Weighted Naive Bayes Training Example ===")

	// Step 1: Create sample data
	// In real usage, you would load your actual classifier predictions
	dataset := createSampleDataset()

	// Step 2: Create your base models
	// Replace these with your actual classifiers
	baseModels := []framework.Model{
		models.NewSimpleClassifier("Classifier1", 0.5, 0.1, 42),
		models.NewSimpleClassifier("Classifier2", 0.6, 0.15, 43),
		models.NewSimpleClassifier("Classifier3", 0.4, 0.1, 44),
		models.NewLogisticClassifier("Classifier4", 3),
	}

	// Step 3: Configure training
	config := framework.DefaultConfig()
	config.TrainingConfig.MaxEpochs = 50
	config.DataConfig.KFolds = 5
	config.TrainingConfig.OptimizationMetric = "pr_auc"

	// Step 4: Create trainer and train
	trainer := framework.NewTrainer(config)
	
	fmt.Println("\nStarting training...")
	result, err := trainer.Train(dataset, baseModels)
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	// Step 5: Display results
	fmt.Println("\n=== Training Complete ===")
	fmt.Printf("Training time: %v\n", result.TrainingTime)
	fmt.Printf("Final PR-AUC: %.4f\n", result.FinalMetrics["pr_auc"])
	fmt.Printf("Final ROC-AUC: %.4f\n", result.FinalMetrics["roc_auc"])

	fmt.Println("\nOptimized Weights:")
	for i, weight := range result.BestWeights {
		fmt.Printf("  %s: %.4f\n", baseModels[i].GetName(), weight)
	}

	// Step 6: Use the trained ensemble
	ensemble := &framework.EnsembleModel{
		Models:  baseModels,
		Weights: result.BestWeights,
	}

	// Make predictions on new data
	newSamples := [][]float64{
		{0.8, 0.7, 0.9},
		{0.2, 0.3, 0.1},
		{0.5, 0.5, 0.5},
	}

	predictions, err := ensemble.Predict(newSamples)
	if err != nil {
		log.Fatalf("Prediction failed: %v", err)
	}

	fmt.Println("\nPredictions on new samples:")
	for i, pred := range predictions {
		fmt.Printf("  Sample %d: %.4f\n", i+1, pred)
	}
}

// createSampleDataset creates a sample dataset for demonstration
// In real usage, you would load actual predictions from your classifiers
func createSampleDataset() *data.Dataset {
	// Create samples with 3 features (predictions from 3 base classifiers)
	samples := []data.Sample{
		// Positive examples (label = 1)
		{Features: []float64{0.8, 0.7, 0.9}, Label: 1},
		{Features: []float64{0.9, 0.8, 0.7}, Label: 1},
		{Features: []float64{0.7, 0.9, 0.8}, Label: 1},
		{Features: []float64{0.85, 0.75, 0.8}, Label: 1},
		{Features: []float64{0.75, 0.85, 0.9}, Label: 1},
		{Features: []float64{0.8, 0.8, 0.85}, Label: 1},
		{Features: []float64{0.9, 0.7, 0.8}, Label: 1},
		{Features: []float64{0.7, 0.8, 0.75}, Label: 1},
		{Features: []float64{0.85, 0.9, 0.7}, Label: 1},
		{Features: []float64{0.8, 0.85, 0.8}, Label: 1},
		
		// Negative examples (label = 0)
		{Features: []float64{0.2, 0.3, 0.1}, Label: 0},
		{Features: []float64{0.1, 0.2, 0.3}, Label: 0},
		{Features: []float64{0.3, 0.1, 0.2}, Label: 0},
		{Features: []float64{0.25, 0.15, 0.2}, Label: 0},
		{Features: []float64{0.15, 0.25, 0.1}, Label: 0},
		{Features: []float64{0.2, 0.2, 0.15}, Label: 0},
		{Features: []float64{0.1, 0.3, 0.2}, Label: 0},
		{Features: []float64{0.3, 0.2, 0.25}, Label: 0},
		{Features: []float64{0.15, 0.1, 0.3}, Label: 0},
		{Features: []float64{0.2, 0.15, 0.2}, Label: 0},
		
		// Some ambiguous examples
		{Features: []float64{0.5, 0.6, 0.4}, Label: 1},
		{Features: []float64{0.4, 0.5, 0.6}, Label: 0},
		{Features: []float64{0.6, 0.4, 0.5}, Label: 1},
		{Features: []float64{0.45, 0.55, 0.5}, Label: 0},
		{Features: []float64{0.55, 0.45, 0.5}, Label: 1},
	}
	
	return data.NewDataset(samples)
}