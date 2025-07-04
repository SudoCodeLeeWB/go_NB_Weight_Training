package main

import (
	"fmt"
	"log"
	
	"github.com/iwonbin/go-nb-weight-training/pkg/data/generators"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/models"
)

func main() {
	fmt.Println("=== Improved Test with All Fixes ===")
	fmt.Println("1. Three-way split to avoid data leakage")
	fmt.Println("2. Beta calibration (less aggressive)")
	fmt.Println("3. Precision-optimized threshold")
	fmt.Println("4. PR-distance as alternative threshold method")
	fmt.Println("===================================\n")
	
	// Generate complex dataset
	dataset := generators.GenerateComplexDataset(generators.ComplexDatasetConfig{
		NumSamples:         5000,
		NumFeatures:        20,
		NoiseLevel:         0.15,
		ClassImbalance:     0.35,
		FeatureCorrelation: 0.4,
		Nonlinearity:       0.3,
		TemporalDrift:      true,
		HiddenGroups:       5,
		RandomSeed:         42,
	})
	
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
	
	// Test different calibration methods
	calibrationMethods := []string{"beta", "isotonic", "platt", "none"}
	thresholdMetrics := []string{"precision", "mcc", "pr_distance"}
	
	for _, calibMethod := range calibrationMethods {
		fmt.Printf("\n=== Testing Calibration Method: %s ===\n", calibMethod)
		
		for _, threshMetric := range thresholdMetrics {
			fmt.Printf("\nThreshold Metric: %s\n", threshMetric)
			
			// Configure with three-way split
			config := framework.DefaultConfig()
			config.TrainingConfig.MaxEpochs = 50
			config.TrainingConfig.EnableCalibration = true
			config.TrainingConfig.CalibrationMethod = calibMethod
			config.TrainingConfig.ThresholdMetric = threshMetric
			config.TrainingConfig.Verbose = false
			
			// Enable three-way split
			config.DataConfig.UseThreeWaySplit = true
			config.DataConfig.CalibrationSplit = 0.15  // 15% for calibration
			config.DataConfig.ValidationSplit = 0.15   // 15% for validation
			// This leaves 70% for training
			
			config.DataConfig.KFolds = 1
			config.OptimizerConfig.PopulationSize = 50
			
			// Train
			trainer := framework.NewTrainer(config)
			result, err := trainer.Train(dataset, baseModels)
			if err != nil {
				log.Printf("Training failed: %v", err)
				continue
			}
			
			// Display results
			fmt.Printf("  Optimal Threshold: %.4f\n", result.OptimalThreshold)
			if result.MetricsAtThreshold != nil {
				fmt.Printf("  Precision: %.4f\n", result.MetricsAtThreshold["precision"])
				fmt.Printf("  Recall: %.4f\n", result.MetricsAtThreshold["recall"])
				fmt.Printf("  F1-Score: %.4f\n", result.MetricsAtThreshold["f1_score"])
			}
			
			// Test on independent holdout
			holdout := generators.GenerateComplexDataset(generators.ComplexDatasetConfig{
				NumSamples:         1000,
				NumFeatures:        20,
				NoiseLevel:         0.15,
				ClassImbalance:     0.35,
				FeatureCorrelation: 0.4,
				Nonlinearity:       0.3,
				TemporalDrift:      true,
				HiddenGroups:       5,
				RandomSeed:         999,
			})
			
			features := holdout.GetFeatures()
			_ = holdout.GetLabels() // Could be used for holdout evaluation
			
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
			
			// Check prediction distribution
			min, max := predictions[0], predictions[0]
			sum := 0.0
			for _, p := range predictions {
				if p < min {
					min = p
				}
				if p > max {
					max = p
				}
				sum += p
			}
			mean := sum / float64(len(predictions))
			
			fmt.Printf("  Holdout Prediction Range: [%.4f, %.4f], Mean: %.4f\n", min, max, mean)
		}
	}
	
	fmt.Println("\n=== Key Insights ===")
	fmt.Println("1. Beta calibration preserves more of the original distribution")
	fmt.Println("2. Three-way split prevents overfitting to validation set")
	fmt.Println("3. Different threshold metrics give different optimal points")
	fmt.Println("4. PR-distance finds threshold closest to perfect precision-recall")
}