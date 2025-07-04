package main

import (
	"fmt"
	"log"
	"time"
	
	"github.com/iwonbin/go-nb-weight-training/pkg/data/generators"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/metrics"
	"github.com/iwonbin/go-nb-weight-training/models"
)

func main() {
	fmt.Println("=== Complex Realistic Dataset Test ===")
	fmt.Println("This simulates real-world conditions with:")
	fmt.Println("- 5000 samples with class imbalance (35% positive)")
	fmt.Println("- 20 features with correlations and nonlinear patterns")
	fmt.Println("- Subtle noise that's hard to detect")
	fmt.Println("- Hidden subgroups and temporal drift")
	fmt.Println("- Automatic calibration and threshold optimization")
	fmt.Println("=========================================\n")
	
	// Generate complex dataset
	fmt.Println("Generating complex dataset...")
	dataset := generators.GenerateComplexDataset(generators.ComplexDatasetConfig{
		NumSamples:         5000,
		NumFeatures:        20,
		NoiseLevel:         0.15,      // 15% noise - subtle but present
		ClassImbalance:     0.35,      // 35% positive class
		FeatureCorrelation: 0.4,       // moderate correlation
		Nonlinearity:       0.3,       // some nonlinear patterns
		TemporalDrift:      true,      // temporal patterns
		HiddenGroups:       5,         // 5 hidden subgroups
		RandomSeed:         42,
	})
	
	// Analyze dataset characteristics
	characteristics := generators.AnalyzeDataset(dataset)
	fmt.Printf("\nDataset Characteristics:\n")
	fmt.Printf("- Samples: %d\n", characteristics.NumSamples)
	fmt.Printf("- Features: %d\n", characteristics.NumFeatures)
	fmt.Printf("- Class Balance: %.2f%% positive\n", characteristics.ClassBalance*100)
	fmt.Printf("- Estimated Separability: %.4f\n", characteristics.Separability)
	
	// Create diverse set of models
	fmt.Println("\nCreating diverse model ensemble...")
	baseModels := []framework.Model{
		// Simple classifiers with different characteristics
		models.NewSimpleClassifier("Conservative", 0.45, 0.1, 42),
		models.NewSimpleClassifier("Moderate", 0.5, 0.15, 43),
		models.NewSimpleClassifier("Aggressive", 0.55, 0.2, 44),
		
		// Logistic classifiers with different sensitivities
		models.NewLogisticClassifier("Logistic1", 2),
		models.NewLogisticClassifier("Logistic2", 3),
		models.NewLogisticClassifier("Logistic3", 4),
		
		// Additional diverse models
		models.NewSimpleClassifier("Balanced", 0.5, 0.12, 45),
		models.NewLogisticClassifier("Sensitive", 5),
	}
	
	// Configure training with calibration
	config := framework.DefaultConfig()
	config.TrainingConfig.MaxEpochs = 100
	config.TrainingConfig.OptimizationMetric = "pr_auc"
	config.TrainingConfig.EnableCalibration = true
	config.TrainingConfig.ThresholdMetric = "precision"
	config.TrainingConfig.Verbose = true
	config.TrainingConfig.LogInterval = 20
	
	// Use single split for calibration to work
	config.DataConfig.KFolds = 1
	config.DataConfig.ValidationSplit = 0.3
	
	// Adjust optimizer for complex problem
	config.OptimizerConfig.PopulationSize = 75
	
	// Enable early stopping
	config.EarlyStopping = &framework.EarlyStoppingConfig{
		Patience: 15,
		MinDelta: 0.001,
		Mode:     "max",
		Monitor:  "val_pr_auc",
	}
	
	// Train the ensemble
	fmt.Println("\nTraining ensemble (this may take a few minutes)...")
	startTime := time.Now()
	
	trainer := framework.NewTrainer(config)
	result, err := trainer.Train(dataset, baseModels)
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}
	
	trainingTime := time.Since(startTime)
	
	// Display results
	fmt.Println("\n=== TRAINING RESULTS ===")
	fmt.Printf("Training Time: %v\n", trainingTime)
	fmt.Printf("Total Epochs: %d\n", result.TotalEpochs)
	fmt.Printf("Converged: %v\n", result.Converged)
	
	fmt.Println("\nBest Model Weights:")
	for i, weight := range result.BestWeights {
		fmt.Printf("  %s: %.4f\n", baseModels[i].GetName(), weight)
	}
	
	fmt.Println("\nValidation Metrics (at 0.5 threshold):")
	for metric, value := range result.ValMetrics {
		fmt.Printf("  %s: %.4f\n", metric, value)
	}
	
	if result.IsCalibrated {
		fmt.Printf("\n=== CALIBRATION RESULTS ===\n")
		fmt.Printf("Optimal Threshold: %.4f (optimized for %s)\n", 
			result.OptimalThreshold, result.ThresholdMetric)
		
		fmt.Println("\nMetrics at Optimal Threshold:")
		for metric, value := range result.MetricsAtThreshold {
			if metric != "threshold" {
				fmt.Printf("  %s: %.4f\n", metric, value)
			}
		}
		
		// Compare with standard threshold
		fmt.Println("\nImprovement over 0.5 threshold:")
		if f1_05, ok := result.ValMetrics["f1_score"]; ok {
			f1_opt := result.MetricsAtThreshold["f1_score"]
			improvement := (f1_opt - f1_05) / f1_05 * 100
			fmt.Printf("  F1-Score improvement: %.1f%%\n", improvement)
		}
	}
	
	// Test on separate holdout set
	fmt.Println("\n=== HOLDOUT TEST ===")
	holdoutDataset := generators.GenerateComplexDataset(generators.ComplexDatasetConfig{
		NumSamples:         1000,
		NumFeatures:        20,
		NoiseLevel:         0.15,
		ClassImbalance:     0.35,
		FeatureCorrelation: 0.4,
		Nonlinearity:       0.3,
		TemporalDrift:      true,
		HiddenGroups:       5,
		RandomSeed:         99, // Different seed for true holdout
	})
	
	// Create ensemble with best weights
	ensemble := &framework.EnsembleModel{
		Models:  baseModels,
		Weights: result.BestWeights,
	}
	
	// Test predictions
	testFeatures := holdoutDataset.GetFeatures()
	testLabels := holdoutDataset.GetLabels()
	
	var predictions []float64
	var predictionType string
	
	// Use calibrated ensemble if available
	if result.IsCalibrated && result.CalibratedEnsemble != nil {
		predictions, _ = result.CalibratedEnsemble.PredictCalibrated(testFeatures)
		predictionType = "calibrated"
	} else {
		predictions, _ = ensemble.Predict(testFeatures)
		predictionType = "uncalibrated"
	}
	
	fmt.Printf("Using %s predictions for holdout test\n", predictionType)
	
	// Evaluate at different thresholds
	fmt.Println("\nHoldout Performance at Different Thresholds:")
	thresholds := []float64{0.5, result.OptimalThreshold, 0.1, 0.01}
	
	for _, threshold := range thresholds {
		cm := metrics.CalculateConfusionMatrix(predictions, testLabels, threshold)
		fmt.Printf("\nThreshold %.4f:\n", threshold)
		fmt.Printf("  Precision: %.4f\n", cm.Precision())
		fmt.Printf("  Recall: %.4f\n", cm.Recall())
		fmt.Printf("  F1-Score: %.4f\n", cm.F1Score())
		fmt.Printf("  Accuracy: %.4f\n", cm.Accuracy())
	}
	
	// Analyze prediction distribution
	fmt.Println("\n=== PREDICTION ANALYSIS ===")
	analyzePredictionDistribution(predictions, testLabels)
	
	fmt.Println("\n=== SUMMARY ===")
	fmt.Println("This complex dataset demonstrates:")
	fmt.Println("1. Realistic class imbalance and noise patterns")
	fmt.Println("2. The importance of calibration for threshold selection")
	fmt.Println("3. How ensemble weighting handles diverse models")
	fmt.Println("4. Performance on truly unseen holdout data")
	
	if result.TotalEpochs > 0 {
		fmt.Printf("\nThe optimizer ran %d epochs, showing it needed to work to find good weights.\n", result.TotalEpochs)
		fmt.Println("This is more realistic than the instant convergence seen with toy datasets.")
	}
}

func analyzePredictionDistribution(predictions, labels []float64) {
	// Separate predictions by class
	var posPreds, negPreds []float64
	
	for i, pred := range predictions {
		if labels[i] > 0.5 {
			posPreds = append(posPreds, pred)
		} else {
			negPreds = append(negPreds, pred)
		}
	}
	
	// Calculate statistics
	posMin, posMax, posMean := calculateStats(posPreds)
	negMin, negMax, negMean := calculateStats(negPreds)
	
	fmt.Printf("Positive Class Predictions:\n")
	fmt.Printf("  Range: [%.6f, %.6f], Mean: %.6f\n", posMin, posMax, posMean)
	
	fmt.Printf("Negative Class Predictions:\n")
	fmt.Printf("  Range: [%.6f, %.6f], Mean: %.6f\n", negMin, negMax, negMean)
	
	// Calculate overlap
	overlap := calculateOverlap(posPreds, negPreds)
	fmt.Printf("\nPrediction Overlap: %.2f%%\n", overlap*100)
	fmt.Printf("(Lower overlap = better separability)\n")
}

func calculateStats(values []float64) (min, max, mean float64) {
	if len(values) == 0 {
		return 0, 0, 0
	}
	
	min, max = values[0], values[0]
	sum := 0.0
	
	for _, v := range values {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
		sum += v
	}
	
	mean = sum / float64(len(values))
	return
}

func calculateOverlap(pos, neg []float64) float64 {
	if len(pos) == 0 || len(neg) == 0 {
		return 0
	}
	
	// Find the overlapping range
	posMin, posMax, _ := calculateStats(pos)
	negMin, negMax, _ := calculateStats(neg)
	
	overlapStart := max(posMin, negMin)
	overlapEnd := min(posMax, negMax)
	
	if overlapStart >= overlapEnd {
		return 0 // No overlap
	}
	
	// Calculate overlap as fraction of total range
	totalRange := max(posMax, negMax) - min(posMin, negMin)
	overlapRange := overlapEnd - overlapStart
	
	return overlapRange / totalRange
}

func max(a, b float64) float64 {
	if a > b {
		return a
	}
	return b
}

func min(a, b float64) float64 {
	if a < b {
		return a
	}
	return b
}