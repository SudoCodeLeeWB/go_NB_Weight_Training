package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/visualization"
)

// MySpamEnsemble is a user's custom implementation of AggregatedModel
// This example shows complete freedom in how you structure your internal models
type MySpamEnsemble struct {
	// These are YOUR models - the framework doesn't know or care about their structure
	randomForest    *MyRandomForest
	neuralNet       *MyNeuralNet
	naiveBayes      *MyNaiveBayes
	svmClassifier   *MySVM
	xgboost         *MyXGBoost
	
	// The only thing the framework cares about: weights
	weights []float64
}

// Example internal model structures (completely up to the user)
type MyRandomForest struct{ threshold float64 }
type MyNeuralNet struct{ layers []float64 }
type MyNaiveBayes struct{ priors map[string]float64 }
type MySVM struct{ kernel string }
type MyXGBoost struct{ trees int }

// NewMySpamEnsemble creates your custom ensemble
func NewMySpamEnsemble() *MySpamEnsemble {
	return &MySpamEnsemble{
		// Initialize your models however you want
		randomForest:  &MyRandomForest{threshold: 0.5},
		neuralNet:     &MyNeuralNet{layers: []float64{10, 5, 1}},
		naiveBayes:    &MyNaiveBayes{priors: make(map[string]float64)},
		svmClassifier: &MySVM{kernel: "rbf"},
		xgboost:       &MyXGBoost{trees: 100},
		
		// Initialize weights (5 models = 5 weights)
		weights: []float64{1.0, 1.0, 1.0, 1.0, 1.0},
	}
}

// Predict - The main method the framework calls
// You implement YOUR OWN aggregation logic here
func (e *MySpamEnsemble) Predict(samples [][]float64) ([]float64, error) {
	n := len(samples)
	results := make([]float64, n)
	
	// Get predictions from each of your models
	// (In real implementation, these would call actual ML models)
	rfPreds := e.randomForest.predict(samples)
	nnPreds := e.neuralNet.predict(samples)
	nbPreds := e.naiveBayes.predict(samples)
	svmPreds := e.svmClassifier.predict(samples)
	xgbPreds := e.xgboost.predict(samples)
	
	// Aggregate using YOUR chosen method
	// Example: Weighted Naive Bayes multiplication
	for i := 0; i < n; i++ {
		// Start with 1.0 for multiplication
		results[i] = 1.0
		
		// Apply each model's prediction with its weight
		results[i] *= math.Pow(rfPreds[i], e.weights[0])
		results[i] *= math.Pow(nnPreds[i], e.weights[1])
		results[i] *= math.Pow(nbPreds[i], e.weights[2])
		results[i] *= math.Pow(svmPreds[i], e.weights[3])
		results[i] *= math.Pow(xgbPreds[i], e.weights[4])
		
		// Ensure result is in [0,1]
		if results[i] > 1.0 {
			results[i] = 1.0
		} else if results[i] < 0.0 {
			results[i] = 0.0
		}
	}
	
	return results, nil
}

// GetWeights - Framework needs this to know current weights
func (e *MySpamEnsemble) GetWeights() []float64 {
	weightsCopy := make([]float64, len(e.weights))
	copy(weightsCopy, e.weights)
	return weightsCopy
}

// SetWeights - Framework calls this to test different weight combinations
func (e *MySpamEnsemble) SetWeights(weights []float64) error {
	if len(weights) != 5 {
		return fmt.Errorf("expected 5 weights, got %d", len(weights))
	}
	copy(e.weights, weights)
	return nil
}

// GetNumModels - Framework needs to know how many weights to optimize
func (e *MySpamEnsemble) GetNumModels() int {
	return 5
}

// GetModelNames - Optional, but helpful for reporting
func (e *MySpamEnsemble) GetModelNames() []string {
	return []string{
		"RandomForest",
		"NeuralNet",
		"NaiveBayes",
		"SVM",
		"XGBoost",
	}
}

// Example predict methods for internal models (mock implementations)
func (rf *MyRandomForest) predict(samples [][]float64) []float64 {
	// In real implementation: call your actual Random Forest model
	preds := make([]float64, len(samples))
	for i := range samples {
		// Mock: use feature average
		avg := 0.0
		for _, f := range samples[i] {
			avg += f
		}
		avg /= float64(len(samples[i]))
		preds[i] = sigmoid(avg - rf.threshold)
	}
	return preds
}

func (nn *MyNeuralNet) predict(samples [][]float64) []float64 {
	// In real implementation: call your actual Neural Network
	preds := make([]float64, len(samples))
	for i := range samples {
		// Mock: simple linear combination
		sum := 0.0
		for j, f := range samples[i] {
			if j < len(nn.layers) {
				sum += f * nn.layers[j]
			}
		}
		preds[i] = sigmoid(sum)
	}
	return preds
}

func (nb *MyNaiveBayes) predict(samples [][]float64) []float64 {
	// In real implementation: call your actual Naive Bayes model
	preds := make([]float64, len(samples))
	for i := range samples {
		// Mock: random with bias
		preds[i] = 0.3 + rand.Float64()*0.4
		if samples[i][0] > 0.5 {
			preds[i] += 0.3
		}
		preds[i] = math.Min(1.0, preds[i])
	}
	return preds
}

func (svm *MySVM) predict(samples [][]float64) []float64 {
	// In real implementation: call your actual SVM
	preds := make([]float64, len(samples))
	for i := range samples {
		// Mock: based on distance from hyperplane
		dist := 0.0
		for _, f := range samples[i] {
			dist += (f - 0.5) * (f - 0.5)
		}
		preds[i] = 1.0 / (1.0 + math.Sqrt(dist))
	}
	return preds
}

func (xgb *MyXGBoost) predict(samples [][]float64) []float64 {
	// In real implementation: call your actual XGBoost model
	preds := make([]float64, len(samples))
	for i := range samples {
		// Mock: ensemble of weak learners
		score := 0.5
		for j := 0; j < xgb.trees && j < len(samples[i]); j++ {
			if samples[i][j] > 0.5 {
				score += 0.01
			} else {
				score -= 0.01
			}
		}
		preds[i] = sigmoid(score)
	}
	return preds
}

// Helper function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// Main function showing how to use the framework
func main() {
	fmt.Println("=== Custom Aggregated Model Example ===")
	fmt.Println("This shows how YOU control the model structure")
	fmt.Println("The framework only optimizes weights\n")

	// Step 1: Create YOUR ensemble however you want
	ensemble := NewMySpamEnsemble()
	fmt.Println("Created custom ensemble with 5 models:")
	for i, name := range ensemble.GetModelNames() {
		fmt.Printf("  %d. %s (weight=%.2f)\n", i+1, name, ensemble.weights[i])
	}

	// Step 2: Load YOUR data
	dataset := createSampleDataset(1000)
	fmt.Printf("\nLoaded dataset: %d samples, %d features\n", 
		dataset.NumSamples, dataset.NumFeatures)

	// Step 3: Configure the framework (only cares about optimization)
	config := framework.DefaultConfig()
	config.TrainingConfig.MaxEpochs = 100
	config.TrainingConfig.OptimizationMetric = "pr_auc"
	config.DataConfig.ValidationSplit = 0.2
	config.Visualization.Enabled = true
	config.Visualization.GenerateReport = true

	// Step 4: Train - Framework ONLY optimizes weights
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("Training: Framework will find optimal weights for YOUR models")
	fmt.Println(strings.Repeat("=", 60))
	
	result, err := framework.TrainAggregatedModel(dataset, ensemble, config)
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	// Step 5: See the optimized weights
	fmt.Println("\n" + strings.Repeat("=", 60))
	fmt.Println("Training Complete!")
	fmt.Println(strings.Repeat("=", 60))
	
	fmt.Printf("\nOptimization took: %v\n", result.TrainingTime)
	fmt.Printf("Best PR-AUC achieved: %.4f\n", result.FinalMetrics["pr_auc"])
	
	fmt.Println("\nOptimized Weights (Framework found these are best):")
	fmt.Println("Model         | Before | After  | Change")
	fmt.Println("--------------|--------|--------|--------")
	originalWeights := []float64{1.0, 1.0, 1.0, 1.0, 1.0}
	for i, name := range ensemble.GetModelNames() {
		change := result.BestWeights[i] - originalWeights[i]
		fmt.Printf("%-13s | %.4f | %.4f | %+.4f\n", 
			name, originalWeights[i], result.BestWeights[i], change)
	}

	// Step 6: Your ensemble now has optimized weights!
	fmt.Println("\nYour ensemble is now optimized and ready to use!")
	
	// Make some predictions
	testSamples := [][]float64{
		{0.9, 0.8, 0.7, 0.9, 0.8},  // Likely spam
		{0.1, 0.2, 0.1, 0.1, 0.2},  // Likely ham
		{0.5, 0.5, 0.5, 0.5, 0.5},  // Uncertain
	}
	
	predictions, _ := ensemble.Predict(testSamples)
	fmt.Println("\nTest Predictions:")
	for i, pred := range predictions {
		label := "Ham"
		if pred > 0.5 {
			label = "Spam"
		}
		fmt.Printf("Sample %d: %.4f => %s\n", i+1, pred, label)
	}

	// Generate report if enabled
	if result.OutputDir != "" {
		reporter := visualization.NewReportGenerator(result.OutputDir)
		if err := reporter.GenerateReport(result, config); err == nil {
			fmt.Printf("\n✅ Report generated at: %s/report.html\n", result.OutputDir)
		}
	}

	fmt.Println("\n🎉 The framework optimized YOUR model's weights without knowing its internals!")
}

// Helper to create sample data
func createSampleDataset(n int) *data.Dataset {
	samples := make([]data.Sample, n)
	
	for i := 0; i < n; i++ {
		// Create random features
		features := make([]float64, 5)
		label := 0
		
		if i < n/2 {
			// Spam samples (label = 1)
			label = 1
			for j := range features {
				features[j] = 0.6 + rand.Float64()*0.4 // Higher values
			}
		} else {
			// Ham samples (label = 0)
			label = 0
			for j := range features {
				features[j] = rand.Float64() * 0.4 // Lower values
			}
		}
		
		samples[i] = data.Sample{
			Features: features,
			Label:    label,
		}
	}
	
	// Shuffle
	rand.Shuffle(len(samples), func(i, j int) {
		samples[i], samples[j] = samples[j], samples[i]
	})
	
	return data.NewDataset(samples)
}

// Add this import at the top
var strings = struct {
	Repeat func(string, int) string
}{
	Repeat: func(s string, n int) string {
		result := ""
		for i := 0; i < n; i++ {
			result += s
		}
		return result
	},
}