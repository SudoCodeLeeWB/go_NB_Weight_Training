package main

import (
	"fmt"
	"log"
	"sort"

	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/metrics"
	"github.com/iwonbin/go-nb-weight-training/models"
)

func main() {
	fmt.Println("=== Precision Analysis at Different Thresholds ===\n")

	// Create realistic dataset
	dataset := createRealisticDataset()

	// Create models
	baseModels := []framework.Model{
		models.NewSimpleClassifier("Model1", 0.5, 0.15, 42),
		models.NewSimpleClassifier("Model2", 0.6, 0.1, 43),
		models.NewLogisticClassifier("Model3", 3),
	}

	// Train
	config := framework.DefaultConfig()
	config.TrainingConfig.MaxEpochs = 50
	config.DataConfig.KFolds = 1
	config.TrainingConfig.Verbose = false
	
	trainer := framework.NewTrainer(config)
	result, err := trainer.Train(dataset, baseModels)
	if err != nil {
		log.Fatal(err)
	}

	// Create ensemble
	ensemble := &framework.EnsembleModel{
		Models:  baseModels,
		Weights: result.BestWeights,
	}

	// Get predictions on test data
	testData := createTestDataset()
	features := testData.GetFeatures()
	labels := testData.GetLabels()
	
	predictions, _ := ensemble.Predict(features)

	// Analyze precision at different thresholds
	fmt.Printf("PR-AUC: %.4f\n\n", result.FinalMetrics["pr_auc"])
	
	// Show prediction distribution
	analyzePredictionDistribution(predictions)
	
	// Calculate metrics at different thresholds
	fmt.Println("\nMetrics at Different Thresholds:")
	fmt.Println("Threshold | Precision | Recall | F1-Score | TP | FP | TN | FN")
	fmt.Println("----------|-----------|--------|----------|----|----|----|----|")
	
	thresholds := []float64{0.5, 0.1, 0.05, 0.01, 0.005, 0.001, 0.0001}
	bestF1 := 0.0
	bestThreshold := 0.0
	
	for _, threshold := range thresholds {
		cm := metrics.CalculateConfusionMatrix(predictions, labels, threshold)
		precision := cm.Precision()
		recall := cm.Recall()
		f1 := cm.F1Score()
		
		if f1 > bestF1 {
			bestF1 = f1
			bestThreshold = threshold
		}
		
		fmt.Printf("%9.4f | %9.4f | %6.4f | %8.4f | %2d | %2d | %2d | %2d\n",
			threshold, precision, recall, f1,
			cm.TruePositives, cm.FalsePositives, cm.TrueNegatives, cm.FalseNegatives)
	}
	
	fmt.Printf("\nBest F1-Score: %.4f at threshold %.4f\n", bestF1, bestThreshold)
	
	// Show the PR curve behavior
	fmt.Println("\n=== Understanding PR-AUC ===")
	fmt.Println("PR-AUC measures the area under the Precision-Recall curve across ALL thresholds.")
	fmt.Println("A high PR-AUC means the model ranks positive examples higher than negative ones.")
	fmt.Println("The actual threshold for deployment should be chosen based on your needs:")
	fmt.Println("- High precision (few false positives): Use higher threshold")
	fmt.Println("- High recall (catch most positives): Use lower threshold")
}

func createRealisticDataset() *data.Dataset {
	samples := []data.Sample{}
	
	// Positive examples with some overlap
	for i := 0; i < 50; i++ {
		samples = append(samples, data.Sample{
			Features: []float64{0.6 + float64(i%10)*0.02, 0.7 - float64(i%5)*0.02, 0.65},
			Label:    1,
		})
	}
	
	// Negative examples with some overlap
	for i := 0; i < 50; i++ {
		samples = append(samples, data.Sample{
			Features: []float64{0.4 - float64(i%10)*0.02, 0.3 + float64(i%5)*0.02, 0.35},
			Label:    0,
		})
	}
	
	return data.NewDataset(samples)
}

func createTestDataset() *data.Dataset {
	samples := []data.Sample{}
	
	// Test positive examples
	for i := 0; i < 25; i++ {
		samples = append(samples, data.Sample{
			Features: []float64{0.55 + float64(i%5)*0.03, 0.6 + float64(i%3)*0.02, 0.58},
			Label:    1,
		})
	}
	
	// Test negative examples
	for i := 0; i < 25; i++ {
		samples = append(samples, data.Sample{
			Features: []float64{0.35 + float64(i%5)*0.02, 0.4 - float64(i%3)*0.02, 0.38},
			Label:    0,
		})
	}
	
	return data.NewDataset(samples)
}

func analyzePredictionDistribution(predictions []float64) {
	// Sort predictions
	sorted := make([]float64, len(predictions))
	copy(sorted, predictions)
	sort.Float64s(sorted)
	
	// Calculate statistics
	min := sorted[0]
	max := sorted[len(sorted)-1]
	median := sorted[len(sorted)/2]
	
	// Count predictions above common thresholds
	above50 := 0
	above10 := 0
	above01 := 0
	
	for _, p := range predictions {
		if p > 0.5 {
			above50++
		}
		if p > 0.1 {
			above10++
		}
		if p > 0.01 {
			above01++
		}
	}
	
	fmt.Println("Prediction Distribution:")
	fmt.Printf("Min: %.6f, Median: %.6f, Max: %.6f\n", min, median, max)
	fmt.Printf("Predictions > 0.5: %d/%d (%.1f%%)\n", above50, len(predictions), float64(above50)*100/float64(len(predictions)))
	fmt.Printf("Predictions > 0.1: %d/%d (%.1f%%)\n", above10, len(predictions), float64(above10)*100/float64(len(predictions)))
	fmt.Printf("Predictions > 0.01: %d/%d (%.1f%%)\n", above01, len(predictions), float64(above01)*100/float64(len(predictions)))
}