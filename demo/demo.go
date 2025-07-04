package main

import (
	"fmt"
	"log"
	"os"

	"github.com/iwonbin/go-nb-weight-training/models"
	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/visualization"
)

func main() {
	fmt.Println("=== Weighted Naive Bayes Spam Detection Demo ===\n")

	// Step 1: Generate spam/ham dataset
	fmt.Println("1. Generating spam/ham dataset...")
	dataset := generateSpamDataset(500) // Smaller dataset for demo
	fmt.Printf("   Generated %d samples (%.1f%% spam)\n\n", 
		dataset.NumSamples, dataset.ClassBalance()*100)

	// Step 2: Split data - 70% train, 30% test
	fmt.Println("2. Splitting dataset...")
	splitter := data.NewStratifiedSplitter(0.3, 42)
	mainSplit, err := splitter.Split(dataset)
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("   Train: %d samples\n", mainSplit.Train.NumSamples)
	fmt.Printf("   Test:  %d samples\n\n", mainSplit.Test.NumSamples)

	// Step 3: Create spam detection models
	fmt.Println("3. Creating spam detection models...")
	spamModels := []framework.Model{
		models.NewBayesianSpamDetector(),
		models.NewNeuralNetSpamDetector(),
		models.NewSVMSpamDetector(),
		models.NewRandomForestSpamDetector(),
		models.NewLogisticRegressionSpamDetector(),
	}
	for _, m := range spamModels {
		fmt.Printf("   - %s\n", m.GetName())
	}

	// Step 4: Configure training
	fmt.Println("\n4. Configuring training...")
	config := &framework.Config{
		DataConfig: framework.DataConfig{
			ValidationSplit: 0.2857, // 5:2 split from train data
			KFolds:          5,
			Stratified:      true,
			RandomSeed:      42,
		},
		TrainingConfig: framework.TrainingConfig{
			MaxEpochs:          30, // Fewer epochs for demo
			BatchSize:          20,
			OptimizationMetric: "pr_auc",
			Verbose:            true,
			LogInterval:        5,
		},
		OptimizerConfig: framework.OptimizerConfig{
			Type:           "differential_evolution",
			PopulationSize: 20, // Smaller population for demo
			MutationFactor: 0.8,
			CrossoverProb:  0.9,
			MinWeight:      0.0,
			MaxWeight:      2.0,
		},
		EarlyStopping: &framework.EarlyStoppingConfig{
			Patience: 5,
			MinDelta: 0.001,
			Monitor:  "val_pr_auc",
			Mode:     "max",
		},
		Visualization: framework.VisualizationConfig{
			Enabled:        true,
			OutputDir:      "./demo_output",
			Formats:        []string{"png"},
			GenerateReport: true,
			DPI:            150,
		},
	}

	// Create output directory
	os.MkdirAll("./demo_output", 0755)

	// Step 5: Train the ensemble
	fmt.Println("\n5. Training ensemble (this shows batch progress)...")
	fmt.Println("   Watch the batch loss decrease as training progresses:\n")
	
	trainer := framework.NewTrainer(config)
	result, err := trainer.Train(mainSplit.Train, spamModels)
	if err != nil {
		log.Fatal(err)
	}

	// Step 6: Show results
	fmt.Println("\n6. Training Results:")
	fmt.Printf("   Training time: %v\n", result.TrainingTime)
	fmt.Printf("   Converged at epoch: %d\n", result.TotalEpochs)
	fmt.Printf("   Final validation PR-AUC: %.4f\n", result.ValMetrics["pr_auc"])

	fmt.Println("\n7. Optimized Model Weights:")
	totalWeight := 0.0
	for _, w := range result.BestWeights {
		totalWeight += w
	}
	fmt.Println("   Model                | Weight | Importance")
	fmt.Println("   ---------------------|--------|------------")
	for i, weight := range result.BestWeights {
		fmt.Printf("   %-20s | %.4f | %5.1f%%\n", 
			spamModels[i].GetName(), 
			weight, 
			weight/totalWeight*100)
	}

	// Step 7: Test on holdout set
	fmt.Println("\n8. Testing on holdout test set...")
	ensemble := &framework.EnsembleModel{
		Models:  spamModels,
		Weights: result.BestWeights,
	}

	testFeatures := mainSplit.Test.GetFeatures()
	testLabels := mainSplit.Test.GetLabels()
	predictions, _ := ensemble.Predict(testFeatures)

	// Calculate test metrics
	correct := 0
	for i := range predictions {
		predLabel := 0.0
		if predictions[i] > 0.5 {
			predLabel = 1.0
		}
		if predLabel == testLabels[i] {
			correct++
		}
	}
	accuracy := float64(correct) / float64(len(predictions))
	fmt.Printf("   Test accuracy: %.2f%%\n", accuracy*100)

	// Show example predictions
	fmt.Println("\n9. Example Predictions:")
	fmt.Println("   True  | Pred  | Confidence")
	fmt.Println("   ------|-------|------------")
	for i := 0; i < 10 && i < len(predictions); i++ {
		trueLabel := "ham "
		if testLabels[i] == 1.0 {
			trueLabel = "spam"
		}
		predLabel := "ham "
		if predictions[i] > 0.5 {
			predLabel = "spam"
		}
		fmt.Printf("   %s | %s | %.4f\n", trueLabel, predLabel, predictions[i])
	}

	// Generate visualizations
	fmt.Println("\n10. Generating visualizations...")
	reporter := visualization.NewReportGenerator("./demo_output")
	if err := reporter.GenerateReport(result, config); err != nil {
		log.Printf("Failed to generate report: %v", err)
	} else {
		fmt.Println("    ✓ Report saved to: demo_output/report.html")
		fmt.Println("    ✓ PR curve saved to: demo_output/pr_curve.png")
		fmt.Println("    ✓ ROC curve saved to: demo_output/roc_curve.png")
	}

	fmt.Println("\n=== Demo Complete ===")
	fmt.Println("Open demo_output/report.html in your browser to see the full results!")
}

// Simplified dataset generation for demo
func generateSpamDataset(n int) *data.Dataset {
	samples := make([]data.Sample, n)
	
	for i := 0; i < n; i++ {
		var features []float64
		var label float64
		
		if i < n*6/10 { // 60% ham
			// Ham emails have low spam indicators
			features = []float64{
				float64(randInt(0, 3)),   // Few spam words
				float64(randInt(5, 10)),  // Many ham words
				float64(randInt(0, 2)),   // Few exclamations
				randFloat(0, 0.3),        // Low caps ratio
				randFloat(0.3, 0.7),      // Medium length
			}
			label = 0.0
		} else { // 40% spam
			// Spam emails have high spam indicators
			features = []float64{
				float64(randInt(5, 10)),  // Many spam words
				float64(randInt(0, 3)),   // Few ham words
				float64(randInt(3, 8)),   // Many exclamations
				randFloat(0.5, 0.9),      // High caps ratio
				randFloat(0.1, 0.5),      // Short length
			}
			label = 1.0
		}
		
		samples[i] = data.Sample{
			Features: features,
			Label:    label,
			ID:       fmt.Sprintf("email_%d", i),
		}
	}
	
	return data.NewDataset(samples)
}

var seed int64 = 42

func randInt(min, max int) int {
	seed = (seed*1103515245 + 12345) & 0x7fffffff
	return min + int(seed)%(max-min+1)
}

func randFloat(min, max float64) float64 {
	seed = (seed*1103515245 + 12345) & 0x7fffffff
	return min + (max-min)*float64(seed)/float64(0x7fffffff)
}