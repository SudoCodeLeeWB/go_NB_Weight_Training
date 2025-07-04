package main

import (
	"fmt"
	"log"
	"math"
	
	"github.com/iwonbin/go-nb-weight-training/pkg/data/generators"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/metrics"
	"github.com/iwonbin/go-nb-weight-training/models"
)

func main() {
	fmt.Println("=== Overfitting and Data Leakage Analysis ===")
	fmt.Println("This test will:")
	fmt.Println("1. Train on one dataset")
	fmt.Println("2. Validate on a second dataset (for weight optimization)")
	fmt.Println("3. Calibrate on validation set")
	fmt.Println("4. Test on completely separate holdout sets")
	fmt.Println("5. Compare performance across multiple holdout sets")
	fmt.Println("===========================================\n")
	
	// Generate datasets with same distribution but different seeds
	config := generators.ComplexDatasetConfig{
		NumSamples:         3000,
		NumFeatures:        20,
		NoiseLevel:         0.15,
		ClassImbalance:     0.35,
		FeatureCorrelation: 0.4,
		Nonlinearity:       0.3,
		TemporalDrift:      true,
		HiddenGroups:       5,
		RandomSeed:         42,
	}
	
	fmt.Println("Generating training dataset...")
	trainDataset := generators.GenerateComplexDataset(config)
	
	// Create models
	baseModels := []framework.Model{
		models.NewSimpleClassifier("Conservative", 0.45, 0.1, 42),
		models.NewSimpleClassifier("Moderate", 0.5, 0.15, 43),
		models.NewSimpleClassifier("Aggressive", 0.55, 0.2, 44),
		models.NewLogisticClassifier("Logistic1", 2),
		models.NewLogisticClassifier("Logistic2", 3),
		models.NewLogisticClassifier("Logistic3", 4),
		models.NewSimpleClassifier("Balanced", 0.5, 0.12, 45),
		models.NewLogisticClassifier("Sensitive", 5),
	}
	
	// Configure training
	trainConfig := framework.DefaultConfig()
	trainConfig.TrainingConfig.MaxEpochs = 100
	trainConfig.TrainingConfig.EnableCalibration = true
	trainConfig.TrainingConfig.ThresholdMetric = "precision"
	trainConfig.TrainingConfig.Verbose = false
	trainConfig.DataConfig.KFolds = 1
	trainConfig.DataConfig.ValidationSplit = 0.3
	trainConfig.OptimizerConfig.PopulationSize = 50
	
	// Train
	fmt.Println("\nTraining ensemble...")
	trainer := framework.NewTrainer(trainConfig)
	result, err := trainer.Train(trainDataset, baseModels)
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}
	
	fmt.Printf("\nTraining complete:")
	fmt.Printf("\n- Epochs: %d", result.TotalEpochs)
	fmt.Printf("\n- Validation PR-AUC: %.4f", result.ValMetrics["pr_auc"])
	fmt.Printf("\n- Optimal threshold: %.4f (maximizing %s)", result.OptimalThreshold, result.ThresholdMetric)
	
	if result.IsCalibrated {
		fmt.Printf("\n\nCalibration metrics on validation set:")
		fmt.Printf("\n- Precision: %.4f", result.MetricsAtThreshold["precision"])
		fmt.Printf("\n- Recall: %.4f", result.MetricsAtThreshold["recall"])
		fmt.Printf("\n- F1: %.4f", result.MetricsAtThreshold["f1_score"])
	}
	
	// Test on multiple independent holdout sets
	fmt.Println("\n\n=== Testing on Multiple Independent Holdout Sets ===")
	
	holdoutSeeds := []int64{99, 123, 456, 789, 1000}
	var precisions, recalls, f1s []float64
	
	for i, seed := range holdoutSeeds {
		config.RandomSeed = seed
		config.NumSamples = 1000 // Smaller holdout sets
		holdout := generators.GenerateComplexDataset(config)
		
		features := holdout.GetFeatures()
		labels := holdout.GetLabels()
		
		// Get predictions
		var predictions []float64
		if result.IsCalibrated && result.CalibratedEnsemble != nil {
			predictions, _ = result.CalibratedEnsemble.PredictCalibrated(features)
		} else {
			ensemble := &framework.EnsembleModel{
				Models:  baseModels,
				Weights: result.BestWeights,
			}
			predictions, _ = ensemble.Predict(features)
		}
		
		// Evaluate at optimal threshold
		cm := metrics.CalculateConfusionMatrix(predictions, labels, result.OptimalThreshold)
		precision := cm.Precision()
		recall := cm.Recall()
		f1 := cm.F1Score()
		
		precisions = append(precisions, precision)
		recalls = append(recalls, recall)
		f1s = append(f1s, f1)
		
		fmt.Printf("\nHoldout Set %d (seed=%d):", i+1, seed)
		fmt.Printf("\n  Precision: %.4f", precision)
		fmt.Printf("\n  Recall: %.4f", recall)
		fmt.Printf("\n  F1-Score: %.4f", f1)
		
		// Check prediction distribution
		min, max, mean := getStats(predictions)
		fmt.Printf("\n  Prediction range: [%.4f, %.4f], mean: %.4f", min, max, mean)
	}
	
	// Calculate statistics across holdout sets
	fmt.Println("\n\n=== Performance Statistics Across Holdout Sets ===")
	
	precMean, precStd := meanStd(precisions)
	recMean, recStd := meanStd(recalls)
	f1Mean, f1Std := meanStd(f1s)
	
	fmt.Printf("\nPrecision: %.4f ± %.4f", precMean, precStd)
	fmt.Printf("\nRecall: %.4f ± %.4f", recMean, recStd)
	fmt.Printf("\nF1-Score: %.4f ± %.4f", f1Mean, f1Std)
	
	// Check for overfitting
	fmt.Println("\n\n=== Overfitting Analysis ===")
	
	valPrecision := result.MetricsAtThreshold["precision"]
	valRecall := result.MetricsAtThreshold["recall"]
	valF1 := result.MetricsAtThreshold["f1_score"]
	
	precDiff := math.Abs(valPrecision - precMean)
	recDiff := math.Abs(valRecall - recMean)
	f1Diff := math.Abs(valF1 - f1Mean)
	
	fmt.Printf("\nValidation vs Holdout Performance Differences:")
	fmt.Printf("\n- Precision difference: %.4f (%.1f%%)", precDiff, precDiff/valPrecision*100)
	fmt.Printf("\n- Recall difference: %.4f (%.1f%%)", recDiff, recDiff/valRecall*100)
	fmt.Printf("\n- F1 difference: %.4f (%.1f%%)", f1Diff, f1Diff/valF1*100)
	
	// Determine if overfitting
	if precDiff/valPrecision > 0.1 || f1Diff/valF1 > 0.1 {
		fmt.Println("\n⚠️  WARNING: Possible overfitting detected!")
		fmt.Println("Validation performance is significantly better than holdout.")
	} else {
		fmt.Println("\n✓ No significant overfitting detected.")
		fmt.Println("Performance is consistent between validation and holdout sets.")
	}
	
	// Check for data leakage
	fmt.Println("\n=== Data Leakage Check ===")
	fmt.Println("Calibration process:")
	fmt.Println("1. Weights optimized on: Training set (70%)")
	fmt.Println("2. Calibration fitted on: Validation set (30%)")
	fmt.Println("3. Threshold selected on: Same validation set (30%)")
	fmt.Println("4. Final test on: Independent holdout sets")
	
	if precStd > 0.1 {
		fmt.Println("\n⚠️  High variance in holdout performance may indicate:")
		fmt.Println("- Threshold overfit to validation set")
		fmt.Println("- Need for separate calibration/threshold sets")
	} else {
		fmt.Println("\n✓ Low variance suggests threshold generalizes well.")
	}
	
	// Recommendations
	fmt.Println("\n=== Recommendations ===")
	if precDiff/valPrecision > 0.05 {
		fmt.Println("- Consider using 3-way split: train/calibration/validation")
		fmt.Println("- Increase regularization or reduce model complexity")
	}
	if result.OptimalThreshold > 0.9 || result.OptimalThreshold < 0.1 {
		fmt.Println("- Extreme threshold suggests calibration issues")
		fmt.Println("- Consider alternative calibration methods")
	}
	if precStd > 0.1 {
		fmt.Println("- High variance suggests unstable predictions")
		fmt.Println("- Consider ensemble averaging or more training data")
	}
}

func getStats(values []float64) (min, max, mean float64) {
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

func meanStd(values []float64) (mean, std float64) {
	if len(values) == 0 {
		return 0, 0
	}
	
	// Calculate mean
	sum := 0.0
	for _, v := range values {
		sum += v
	}
	mean = sum / float64(len(values))
	
	// Calculate standard deviation
	sumSq := 0.0
	for _, v := range values {
		diff := v - mean
		sumSq += diff * diff
	}
	std = math.Sqrt(sumSq / float64(len(values)))
	
	return
}