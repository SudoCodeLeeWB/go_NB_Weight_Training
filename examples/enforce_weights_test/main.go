package main

import (
	"fmt"
	"log"
	
	"github.com/iwonbin/go-nb-weight-training/pkg/data/generators"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/models"
)

func main() {
	fmt.Println("=== Testing Model Exclusion vs. Forced Inclusion ===\n")
	
	// Generate dataset
	dataset := generators.GenerateComplexDataset(generators.ComplexDatasetConfig{
		NumSamples:         2000,
		NumFeatures:        15,
		NoiseLevel:         0.2, // More noise
		ClassImbalance:     0.3,
		FeatureCorrelation: 0.5,
		Nonlinearity:       0.4,
		TemporalDrift:      false,
		HiddenGroups:       3,
		RandomSeed:         42,
	})
	
	// Create diverse models (some are intentionally bad)
	baseModels := []framework.Model{
		models.NewSimpleClassifier("Good1", 0.65, 0.1, 42),
		models.NewSimpleClassifier("Good2", 0.6, 0.15, 43),
		models.NewSimpleClassifier("Bad1", 0.2, 0.4, 44),      // Very low accuracy
		models.NewSimpleClassifier("Bad2", 0.5, 0.8, 45),      // Extreme noise (almost random)
		models.NewLogisticClassifier("Good3", 3),
		models.NewSimpleClassifier("Opposite", 0.3, 0.1, 46),  // Predicts opposite class
		models.NewLogisticClassifier("Good4", 2),
		models.NewSimpleClassifier("Noise", 0.5, 1.0, 47),     // Pure noise
	}
	
	// Test 1: Allow model exclusion (default)
	fmt.Println("TEST 1: Allowing Model Exclusion (enforce_non_zero = false)")
	fmt.Println("-----------------------------------------------------")
	
	config1 := framework.DefaultConfig()
	config1.TrainingConfig.MaxEpochs = 50
	config1.TrainingConfig.Verbose = false
	config1.DataConfig.KFolds = 1
	config1.DataConfig.ValidationSplit = 0.3
	config1.OptimizerConfig.EnforceNonZero = false  // Allow zero weights
	
	trainer1 := framework.NewTrainer(config1)
	result1, err := trainer1.Train(dataset, baseModels)
	if err != nil {
		log.Fatal(err)
	}
	
	fmt.Println("\nModel Weights (allowing exclusion):")
	activeCount := 0
	for i, weight := range result1.BestWeights {
		status := "EXCLUDED"
		if weight > 0.001 {
			status = "ACTIVE"
			activeCount++
		}
		fmt.Printf("  %s: %.4f [%s]\n", baseModels[i].GetName(), weight, status)
	}
	fmt.Printf("\nActive models: %d/%d\n", activeCount, len(baseModels))
	fmt.Printf("PR-AUC: %.4f\n", result1.FinalMetrics["pr_auc"])
	
	// Test 2: Force all models to be included
	fmt.Println("\n\nTEST 2: Forcing All Models (enforce_non_zero = true)")
	fmt.Println("----------------------------------------------------")
	
	config2 := framework.DefaultConfig()
	config2.TrainingConfig.MaxEpochs = 50
	config2.TrainingConfig.Verbose = false
	config2.DataConfig.KFolds = 1
	config2.DataConfig.ValidationSplit = 0.3
	config2.OptimizerConfig.EnforceNonZero = true   // Force non-zero weights
	config2.OptimizerConfig.MinWeight = 0.01       // Minimum weight
	
	trainer2 := framework.NewTrainer(config2)
	result2, err := trainer2.Train(dataset, baseModels)
	if err != nil {
		log.Fatal(err)
	}
	
	fmt.Println("\nModel Weights (forced inclusion):")
	minWeight := 10.0
	for i, weight := range result2.BestWeights {
		fmt.Printf("  %s: %.4f\n", baseModels[i].GetName(), weight)
		if weight < minWeight {
			minWeight = weight
		}
	}
	fmt.Printf("\nMinimum weight: %.4f (all models active)\n", minWeight)
	fmt.Printf("PR-AUC: %.4f\n", result2.FinalMetrics["pr_auc"])
	
	// Compare results
	fmt.Println("\n\n=== COMPARISON ===")
	fmt.Printf("PR-AUC with exclusion:    %.4f\n", result1.FinalMetrics["pr_auc"])
	fmt.Printf("PR-AUC forced inclusion:  %.4f\n", result2.FinalMetrics["pr_auc"])
	
	diff := result1.FinalMetrics["pr_auc"] - result2.FinalMetrics["pr_auc"]
	if diff > 0 {
		fmt.Printf("\nExclusion improved PR-AUC by %.4f (%.1f%%)\n", diff, diff/result2.FinalMetrics["pr_auc"]*100)
		fmt.Println("The optimizer correctly identified and excluded bad models!")
	} else {
		fmt.Printf("\nForced inclusion improved PR-AUC by %.4f\n", -diff)
		fmt.Println("All models contributed positively in this case.")
	}
	
	fmt.Println("\n=== KEY INSIGHTS ===")
	fmt.Println("1. With enforce_non_zero=false: Bad models get weight ≈ 0")
	fmt.Println("2. With enforce_non_zero=true: All models get weight ≥ 0.01")
	fmt.Println("3. Model exclusion usually improves performance")
	fmt.Println("4. Use enforce_non_zero=true only when you need all models active")
}