package main

import (
	"fmt"
	"log"

	"github.com/iwonbin/go-nb-weight-training/models"
	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
)

func main() {
	fmt.Println("=== Simple Training Progress Demo ===\n")

	// Create a small dataset
	fmt.Println("Creating dataset with 200 samples...")
	dataset := createSimpleDataset(200)
	
	// Single train/val split (no cross-validation for clarity)
	config := &framework.Config{
		DataConfig: framework.DataConfig{
			ValidationSplit: 0.3,  // 30% validation
			KFolds:          2,    // Minimum required
			Stratified:      true,
			RandomSeed:      42,
		},
		TrainingConfig: framework.TrainingConfig{
			MaxEpochs:          50,
			BatchSize:          10,  // Small batch size to see more updates
			OptimizationMetric: "pr_auc",
			Verbose:            true,
			LogInterval:        1,   // Log every epoch
		},
		OptimizerConfig: framework.OptimizerConfig{
			Type:           "differential_evolution",
			PopulationSize: 10,  // Small population for faster iterations
			MutationFactor: 0.8,
			CrossoverProb:  0.9,
			MinWeight:      0.0,
			MaxWeight:      2.0,
		},
		Visualization: framework.VisualizationConfig{
			Enabled: false,  // Disable for simple demo
		},
	}

	// Create models
	spamModels := []framework.Model{
		models.NewBayesianSpamDetector(),
		models.NewNeuralNetSpamDetector(),
		models.NewSVMSpamDetector(),
	}

	// Train
	fmt.Println("\nTraining with batch progress logging:")
	fmt.Println("(Watch how the score improves and loss decreases)\n")
	
	trainer := framework.NewTrainer(config)
	result, err := trainer.Train(dataset, spamModels)
	if err != nil {
		log.Fatal(err)
	}

	// Results
	fmt.Println("\n=== Final Results ===")
	fmt.Printf("Total training time: %v\n", result.TrainingTime)
	fmt.Printf("Final PR-AUC: %.4f\n", result.FinalMetrics["pr_auc"])
	
	fmt.Println("\nLearned weights:")
	for i, w := range result.BestWeights {
		fmt.Printf("  %s: %.4f\n", spamModels[i].GetName(), w)
	}
}

func createSimpleDataset(n int) *data.Dataset {
	samples := make([]data.Sample, n)
	
	for i := 0; i < n; i++ {
		var features []float64
		var label float64
		
		if i%2 == 0 { // Ham
			features = []float64{
				float64(i%3),     // Low spam words
				float64(5+i%3),   // High ham words
				float64(i%2),     // Low exclamations
				0.1 + float64(i%10)/100, // Low caps
				0.5,              // Medium length
			}
			label = 0.0
		} else { // Spam
			features = []float64{
				float64(5+i%3),   // High spam words
				float64(i%3),     // Low ham words
				float64(3+i%2),   // High exclamations
				0.6 + float64(i%10)/100, // High caps
				0.3,              // Short length
			}
			label = 1.0
		}
		
		samples[i] = data.Sample{
			Features: features,
			Label:    label,
		}
	}
	
	return data.NewDataset(samples)
}