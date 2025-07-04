package main

import (
	"fmt"
	"log"
	"math/rand"

	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/models"
)

func main() {
	fmt.Println("=== Realistic Weighted Naive Bayes Training Example ===")
	fmt.Println("This example uses more realistic, overlapping data")

	// Create realistic dataset with overlapping classes
	dataset := createRealisticDataset()

	// Create base models with varying performance
	baseModels := []framework.Model{
		models.NewSimpleClassifier("WeakClassifier", 0.3, 0.2, 42),    // Weak model
		models.NewSimpleClassifier("ModerateClassifier", 0.5, 0.15, 43), // Moderate
		models.NewSimpleClassifier("GoodClassifier", 0.7, 0.1, 44),      // Good
		models.NewLogisticClassifier("LogisticModel", 3),
		models.NewRandomClassifier("RandomBaseline", 0.5, 45), // Random baseline
	}

	// Configure training
	config := framework.DefaultConfig()
	config.TrainingConfig.MaxEpochs = 100
	config.DataConfig.KFolds = 5
	config.TrainingConfig.OptimizationMetric = "pr_auc"
	config.DataConfig.ValidationSplit = 0.2
	
	// Enable early stopping
	config.EarlyStopping = &framework.EarlyStoppingConfig{
		Patience: 10,
		MinDelta: 0.001,
		Monitor:  "pr_auc",
		Mode:     "max",
	}

	// Train
	trainer := framework.NewTrainer(config)
	
	fmt.Println("\nStarting training on realistic data...")
	result, err := trainer.Train(dataset, baseModels)
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	// Display results
	fmt.Println("\n=== Training Complete ===")
	fmt.Printf("Training time: %v\n", result.TrainingTime)
	fmt.Printf("Total epochs: %d\n", result.TotalEpochs)
	fmt.Println("\nFinal Metrics:")
	fmt.Printf("  PR-AUC: %.4f\n", result.FinalMetrics["pr_auc"])
	fmt.Printf("  ROC-AUC: %.4f\n", result.FinalMetrics["roc_auc"])
	fmt.Printf("  Precision: %.4f\n", result.FinalMetrics["precision"])
	fmt.Printf("  Recall: %.4f\n", result.FinalMetrics["recall"])
	fmt.Printf("  F1-Score: %.4f\n", result.FinalMetrics["f1_score"])

	fmt.Println("\nOptimized Weights (importance):")
	totalWeight := 0.0
	for _, w := range result.BestWeights {
		totalWeight += w
	}
	
	for i, weight := range result.BestWeights {
		importance := (weight / totalWeight) * 100
		fmt.Printf("  %s: %.4f (%.1f%%)\n", baseModels[i].GetName(), weight, importance)
	}

	// Test on new realistic samples
	testRealisticPredictions(result, baseModels)
}

// createRealisticDataset creates a dataset with overlapping classes
// This simulates real-world scenarios where perfect separation is impossible
func createRealisticDataset() *data.Dataset {
	rand.Seed(42)
	samples := []data.Sample{}
	
	// Generate 200 samples with realistic overlap
	for i := 0; i < 200; i++ {
		var features []float64
		var label float64
		
		if i < 100 {
			// Positive class - tend to have higher values but with noise
			label = 1
			features = []float64{
				clamp(0.6 + rand.NormFloat64()*0.2), // Mean 0.6, std 0.2
				clamp(0.7 + rand.NormFloat64()*0.25),
				clamp(0.65 + rand.NormFloat64()*0.2),
			}
		} else {
			// Negative class - tend to have lower values but with overlap
			label = 0
			features = []float64{
				clamp(0.4 + rand.NormFloat64()*0.2), // Mean 0.4, std 0.2
				clamp(0.3 + rand.NormFloat64()*0.25),
				clamp(0.35 + rand.NormFloat64()*0.2),
			}
		}
		
		// Add some mislabeled samples (noise in labels) - 5% noise
		if rand.Float64() < 0.05 {
			label = 1 - label
		}
		
		samples = append(samples, data.Sample{
			Features: features,
			Label:    label,
		})
	}
	
	// Shuffle samples
	rand.Shuffle(len(samples), func(i, j int) {
		samples[i], samples[j] = samples[j], samples[i]
	})
	
	return data.NewDataset(samples)
}

// clamp ensures values are in [0, 1] range
func clamp(val float64) float64 {
	if val < 0 {
		return 0
	}
	if val > 1 {
		return 1
	}
	return val
}

// testRealisticPredictions shows predictions on realistic test cases
func testRealisticPredictions(result *framework.TrainingResult, models []framework.Model) {
	ensemble := &framework.EnsembleModel{
		Models:  models,
		Weights: result.BestWeights,
	}
	
	fmt.Println("\n=== Predictions on Realistic Test Cases ===")
	
	testCases := []struct {
		name     string
		features []float64
		expected string
	}{
		{"Clear Positive", []float64{0.8, 0.9, 0.85}, "positive"},
		{"Clear Negative", []float64{0.2, 0.1, 0.15}, "negative"},
		{"Borderline Case 1", []float64{0.5, 0.45, 0.55}, "uncertain"},
		{"Borderline Case 2", []float64{0.48, 0.52, 0.5}, "uncertain"},
		{"Slightly Positive", []float64{0.6, 0.55, 0.58}, "likely positive"},
		{"Slightly Negative", []float64{0.4, 0.45, 0.42}, "likely negative"},
		{"Noisy Positive", []float64{0.7, 0.3, 0.8}, "mixed signals"},
		{"Noisy Negative", []float64{0.3, 0.7, 0.2}, "mixed signals"},
	}
	
	for _, tc := range testCases {
		pred, _ := ensemble.Predict([][]float64{tc.features})
		fmt.Printf("%-20s: features=%v, prediction=%.4f (%s)\n", 
			tc.name, tc.features, pred[0], tc.expected)
	}
	
	// Show decision boundary
	fmt.Println("\nDecision boundary is around 0.5")
	fmt.Println("Values > 0.5 are classified as positive")
	fmt.Println("Values < 0.5 are classified as negative")
}