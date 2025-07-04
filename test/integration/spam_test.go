package integration

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"testing"

	"github.com/iwonbin/go-nb-weight-training/models"
	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/stretchr/testify/require"
)

// GenerateSpamDataset creates a realistic spam/ham dataset
func GenerateSpamDataset(numSamples int) *data.Dataset {
	rand.Seed(42)
	samples := make([]data.Sample, numSamples)
	
	// Email templates
	spamTemplates := []string{
		"FREE MONEY!!! Click here to claim your $%d prize NOW!",
		"Congratulations! You've WON %d million dollars! Act fast!",
		"Buy cheap viagra pills! %d%% discount TODAY ONLY!!!",
		"URGENT: Your account will be closed! Click here immediately!",
		"Hot singles in your area! %d matches waiting for you!",
		"Lose %d pounds in 1 week! Guaranteed results or money back!",
		"Make $%d working from home! No experience needed!",
		"LIMITED TIME OFFER: Get %d%% off all products! BUY NOW!",
		"You are the WINNER of our weekly lottery! Claim $%d here!",
		"RISK FREE investment opportunity! %d%% returns guaranteed!",
	}
	
	hamTemplates := []string{
		"Hi team, please review the attached report for tomorrow's meeting.",
		"Can we schedule a call to discuss the project update?",
		"Thanks for your help with the presentation yesterday.",
		"Here's the document you requested. Let me know if you need anything else.",
		"Meeting reminder: Team sync at %d PM today in conference room.",
		"Quick question about the budget report - when do you need it by?",
		"I'll be out of office tomorrow. Please contact my colleague for urgent matters.",
		"The quarterly review is scheduled for next week. Please prepare your updates.",
		"Coffee catch-up tomorrow at %d? Let me know what works for you.",
		"Following up on our discussion from last week about the new project.",
	}
	
	// Generate samples
	for i := 0; i < numSamples; i++ {
		var emailText string
		var label float64
		
		// 60% ham, 40% spam for slight imbalance
		if rand.Float64() < 0.6 {
			// Generate ham
			template := hamTemplates[rand.Intn(len(hamTemplates))]
			emailText = fmt.Sprintf(template, rand.Intn(5)+1)
			label = 0.0
			
			// Add some noise to make it harder
			if rand.Float64() < 0.1 {
				emailText += " Special offer inside!"
			}
		} else {
			// Generate spam
			template := spamTemplates[rand.Intn(len(spamTemplates))]
			emailText = fmt.Sprintf(template, rand.Intn(1000)+100)
			label = 1.0
			
			// Add some legitimate words to make it harder
			if rand.Float64() < 0.1 {
				emailText += " Please let me know if you have questions."
			}
		}
		
		// Extract features
		features := models.ExtractEmailFeatures(emailText)
		
		samples[i] = data.Sample{
			Features: features,
			Label:    label,
			ID:       fmt.Sprintf("email_%d", i),
		}
	}
	
	return data.NewDataset(samples)
}

func TestSpamDetectionFramework(t *testing.T) {
	// Generate dataset
	dataset := GenerateSpamDataset(1000)
	
	// Create spam detection models
	spamModels := []framework.Model{
		models.NewBayesianSpamDetector(),
		models.NewNeuralNetSpamDetector(),
		models.NewSVMSpamDetector(),
		models.NewRandomForestSpamDetector(),
		models.NewLogisticRegressionSpamDetector(),
	}
	
	// Configure training with enhanced settings
	config := &framework.Config{
		DataConfig: framework.DataConfig{
			ValidationSplit: 0.2857, // To get 5:2 split from 70% train data
			KFolds:          1,      // Single split for this test
			Stratified:      true,
			RandomSeed:      42,
		},
		TrainingConfig: framework.TrainingConfig{
			MaxEpochs:          50,
			BatchSize:          32,
			OptimizationMetric: "pr_auc",
			Verbose:            true,
			LogInterval:        5,
		},
		OptimizerConfig: framework.OptimizerConfig{
			Type:           "differential_evolution",
			PopulationSize: 30,
			MutationFactor: 0.8,
			CrossoverProb:  0.9,
			MinWeight:      0.0,
			MaxWeight:      2.0,
		},
		EarlyStopping: &framework.EarlyStoppingConfig{
			Patience: 10,
			MinDelta: 0.001,
			Monitor:  "val_pr_auc",
			Mode:     "max",
		},
		Visualization: framework.VisualizationConfig{
			Enabled:        true,
			OutputDir:      "./test_output",
			Formats:        []string{"png"},
			GenerateReport: true,
			DPI:            150,
		},
	}
	
	// Create output directory
	os.MkdirAll("./test_output", 0755)
	
	// First split: 70% train, 30% test
	splitter := data.NewStratifiedSplitter(0.3, config.DataConfig.RandomSeed)
	mainSplit, err := splitter.Split(dataset)
	require.NoError(t, err)
	
	fmt.Printf("\n=== Dataset Split ===\n")
	fmt.Printf("Total samples: %d\n", dataset.NumSamples)
	fmt.Printf("Train set: %d samples (%.1f%%)\n", mainSplit.Train.NumSamples, 
		float64(mainSplit.Train.NumSamples)/float64(dataset.NumSamples)*100)
	fmt.Printf("Test set: %d samples (%.1f%%)\n", mainSplit.Test.NumSamples,
		float64(mainSplit.Test.NumSamples)/float64(dataset.NumSamples)*100)
	fmt.Printf("Train set class balance: %.1f%% spam\n", mainSplit.Train.ClassBalance()*100)
	fmt.Printf("Test set class balance: %.1f%% spam\n", mainSplit.Test.ClassBalance()*100)
	
	// Train the ensemble
	fmt.Printf("\n=== Starting Training ===\n")
	trainer := framework.NewTrainer(config)
	result, err := trainer.Train(mainSplit.Train, spamModels)
	require.NoError(t, err)
	
	// Display results
	fmt.Printf("\n=== Training Results ===\n")
	fmt.Printf("Training time: %v\n", result.TrainingTime)
	fmt.Printf("Total epochs: %d\n", result.TotalEpochs)
	fmt.Printf("Converged: %v\n", result.Converged)
	
	fmt.Printf("\nValidation Metrics:\n")
	for metric, value := range result.ValMetrics {
		fmt.Printf("  %s: %.4f\n", metric, value)
	}
	
	fmt.Printf("\nOptimized Weights:\n")
	totalWeight := 0.0
	for _, w := range result.BestWeights {
		totalWeight += w
	}
	for i, weight := range result.BestWeights {
		fmt.Printf("  %-20s: %.4f (%.1f%%)\n", 
			spamModels[i].GetName(), 
			weight, 
			weight/totalWeight*100)
	}
	
	// Test on holdout test set
	fmt.Printf("\n=== Testing on Holdout Set ===\n")
	ensemble := &framework.EnsembleModel{
		Models:  spamModels,
		Weights: result.BestWeights,
	}
	
	testFeatures := mainSplit.Test.GetFeatures()
	testLabels := mainSplit.Test.GetLabels()
	
	predictions, err := ensemble.Predict(testFeatures)
	require.NoError(t, err)
	
	// Calculate test metrics
	testMetrics := trainer.evaluateMetrics(ensemble, mainSplit.Test)
	fmt.Printf("Test Set Metrics:\n")
	for metric, value := range testMetrics {
		fmt.Printf("  %s: %.4f\n", metric, value)
	}
	
	// Show some example predictions
	fmt.Printf("\n=== Example Predictions ===\n")
	fmt.Printf("Sample | True Label | Prediction | Correct?\n")
	fmt.Printf("-------|------------|------------|----------\n")
	for i := 0; i < 10 && i < len(predictions); i++ {
		trueLabel := testLabels[i]
		predLabel := 0.0
		if predictions[i] > 0.5 {
			predLabel = 1.0
		}
		correct := "✓"
		if trueLabel != predLabel {
			correct = "✗"
		}
		labelStr := "ham"
		if trueLabel == 1.0 {
			labelStr = "spam"
		}
		fmt.Printf("  %3d  | %10s |   %.4f   |    %s\n", 
			i, labelStr, predictions[i], correct)
	}
	
	// Save results
	weightsPath := filepath.Join("./test_output", "spam_weights.json")
	err = framework.SaveWeights(result.BestWeights, weightsPath)
	if err == nil {
		fmt.Printf("\nWeights saved to: %s\n", weightsPath)
	}
	
	// Assertions for test validation
	require.Greater(t, result.ValMetrics["pr_auc"], 0.7, "PR-AUC should be reasonably good")
	require.Greater(t, testMetrics["pr_auc"], 0.6, "Test PR-AUC should be decent")
	require.NotEqual(t, result.BestWeights[0], result.BestWeights[1], "Weights should differ")
}