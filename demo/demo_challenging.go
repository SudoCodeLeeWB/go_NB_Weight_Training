package main

import (
	"fmt"
	"log"
	"math"
	"math/rand"

	"github.com/iwonbin/go-nb-weight-training/models"
	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/metrics"
	"github.com/iwonbin/go-nb-weight-training/pkg/optimizer"
)

func main() {
	fmt.Println("=== Challenging Dataset Training Demo ===\n")

	// Create challenging dataset with overlapping classes
	dataset := createChallengingDataset(500)
	fmt.Printf("Dataset: %d samples (%.1f%% spam)\n", 
		dataset.NumSamples, dataset.ClassBalance()*100)

	// Split data
	splitter := data.NewStratifiedSplitter(0.3, 42)
	split, _ := splitter.Split(dataset)
	
	// Further split train into train/val
	trainSplitter := data.NewStratifiedSplitter(0.2857, 42) // 5:2 ratio
	trainValSplit, _ := trainSplitter.Split(split.Train)
	
	fmt.Printf("Train: %d, Val: %d, Test: %d samples\n\n", 
		trainValSplit.Train.NumSamples, 
		trainValSplit.Test.NumSamples,
		split.Test.NumSamples)

	// Create models with different strengths
	models := []framework.Model{
		models.NewBayesianSpamDetector(),
		models.NewNeuralNetSpamDetector(),
		models.NewSVMSpamDetector(),
		models.NewRandomForestSpamDetector(),
		models.NewLogisticRegressionSpamDetector(),
	}

	// Show individual model performance first
	fmt.Println("Individual Model Performance on Validation Set:")
	fmt.Println("Model                | PR-AUC | Accuracy")
	fmt.Println("---------------------|--------|----------")
	
	valFeatures := trainValSplit.Test.GetFeatures()
	valLabels := trainValSplit.Test.GetLabels()
	
	for _, model := range models {
		preds, _ := model.Predict(valFeatures)
		prauc := &metrics.PRAUC{}
		score, _ := prauc.Calculate(preds, valLabels)
		
		// Calculate accuracy
		correct := 0
		for i := range preds {
			pred := 0.0
			if preds[i] > 0.5 {
				pred = 1.0
			}
			if pred == valLabels[i] {
				correct++
			}
		}
		acc := float64(correct) / float64(len(preds))
		
		fmt.Printf("%-20s | %.4f | %.2f%%\n", model.GetName(), score, acc*100)
	}

	// Train ensemble
	fmt.Println("\nTraining Ensemble with Weight Optimization:")
	fmt.Println("Iter | Train PR-AUC | Val PR-AUC | Best Val")
	fmt.Println("-----|--------------|------------|----------")

	// Track validation performance
	var bestValScore float64
	trainFeatures := trainValSplit.Train.GetFeatures()
	trainLabels := trainValSplit.Train.GetLabels()
	
	ensemble := &framework.EnsembleModel{
		Models: models,
	}
	
	// Objective function that uses validation set
	objectiveFunc := func(weights []float64) float64 {
		ensemble.Weights = weights
		predictions, _ := ensemble.Predict(valFeatures)
		
		prauc := &metrics.PRAUC{}
		score, _ := prauc.Calculate(predictions, valLabels)
		return score
	}
	
	// Optimizer with detailed callback
	optConfig := &optimizer.Config{
		MaxIterations:  50,
		PopulationSize: 20,
		MutationFactor: 0.8,
		CrossoverProb:  0.9,
		MinWeight:      0.0,
		MaxWeight:      3.0,
		RandomSeed:     42,
		Callback: func(iter int, valScore float64, weights []float64) {
			// Also calculate train score
			ensemble.Weights = weights
			trainPreds, _ := ensemble.Predict(trainFeatures)
			prauc := &metrics.PRAUC{}
			trainScore, _ := prauc.Calculate(trainPreds, trainLabels)
			
			if valScore > bestValScore {
				bestValScore = valScore
			}
			
			if iter % 5 == 0 {
				fmt.Printf("%4d |     %.4f   |   %.4f   |  %.4f\n", 
					iter, trainScore, valScore, bestValScore)
			}
		},
	}

	// Optimize
	opt := optimizer.NewDifferentialEvolution()
	result, err := opt.Optimize(objectiveFunc, len(models), optConfig)
	if err != nil {
		log.Fatal(err)
	}

	// Show final results
	fmt.Printf("\n=== Final Results ===\n")
	fmt.Printf("Best Validation PR-AUC: %.4f\n", result.BestScore)
	fmt.Printf("Iterations: %d\n\n", result.Iterations)

	fmt.Println("Optimized Weights:")
	totalWeight := 0.0
	for _, w := range result.BestWeights {
		totalWeight += w
	}
	for i, w := range result.BestWeights {
		fmt.Printf("  %-20s: %.4f (%.1f%%)\n", 
			models[i].GetName(), w, w/totalWeight*100)
	}

	// Test on holdout set
	fmt.Printf("\n=== Test Set Performance ===\n")
	ensemble.Weights = result.BestWeights
	testFeatures := split.Test.GetFeatures()
	testLabels := split.Test.GetLabels()
	
	testPreds, _ := ensemble.Predict(testFeatures)
	
	// Calculate metrics
	prauc := &metrics.PRAUC{}
	testPRAUC, _ := prauc.Calculate(testPreds, testLabels)
	
	rocauc := &metrics.ROCAUC{}
	testROCAUC, _ := rocauc.Calculate(testPreds, testLabels)
	
	// Calculate accuracy
	correct := 0
	for i := range testPreds {
		pred := 0.0
		if testPreds[i] > 0.5 {
			pred = 1.0
		}
		if pred == testLabels[i] {
			correct++
		}
	}
	
	fmt.Printf("Test PR-AUC:  %.4f\n", testPRAUC)
	fmt.Printf("Test ROC-AUC: %.4f\n", testROCAUC)
	fmt.Printf("Test Accuracy: %.2f%%\n", float64(correct)/float64(len(testPreds))*100)
	
	// Show some example predictions
	fmt.Println("\nExample Test Predictions:")
	fmt.Println("True | Pred Score | Pred Label | Correct")
	fmt.Println("-----|------------|------------|--------")
	for i := 0; i < 15 && i < len(testPreds); i++ {
		trueLabel := "ham "
		if testLabels[i] == 1.0 {
			trueLabel = "spam"
		}
		predLabel := "ham "
		if testPreds[i] > 0.5 {
			predLabel = "spam"
		}
		correct := "✗"
		if (testLabels[i] == 1.0 && testPreds[i] > 0.5) || 
		   (testLabels[i] == 0.0 && testPreds[i] <= 0.5) {
			correct = "✓"
		}
		fmt.Printf("%s |   %.4f   |    %s    |   %s\n", 
			trueLabel, testPreds[i], predLabel, correct)
	}
}

// Create a challenging dataset with overlapping features
func createChallengingDataset(n int) *data.Dataset {
	rand.Seed(42)
	samples := make([]data.Sample, n)
	
	for i := 0; i < n; i++ {
		var features []float64
		var label float64
		
		// 40% spam, 60% ham
		if rand.Float64() < 0.6 {
			// Ham - but with some spam-like characteristics
			label = 0.0
			
			// Add variability and overlap
			noise := rand.NormFloat64() * 0.3
			
			features = []float64{
				math.Max(0, 2.0 + rand.NormFloat64()*1.5),     // spam words (some ham has spam words)
				math.Max(0, 6.0 + rand.NormFloat64()*2.0),     // ham words
				math.Max(0, 1.0 + rand.NormFloat64()*1.0),     // exclamations
				math.Max(0, math.Min(1, 0.2 + noise)),         // caps ratio
				math.Max(0, math.Min(1, 0.6 + rand.Float64()*0.3)), // length
			}
			
			// 10% of ham looks very spammy
			if rand.Float64() < 0.1 {
				features[0] += 3.0 // More spam words
				features[2] += 2.0 // More exclamations
			}
		} else {
			// Spam - but with some ham-like characteristics
			label = 1.0
			
			noise := rand.NormFloat64() * 0.3
			
			features = []float64{
				math.Max(0, 6.0 + rand.NormFloat64()*2.0),     // spam words
				math.Max(0, 1.5 + rand.NormFloat64()*1.5),     // ham words (some spam has ham words)
				math.Max(0, 3.5 + rand.NormFloat64()*1.5),     // exclamations
				math.Max(0, math.Min(1, 0.7 + noise)),         // caps ratio
				math.Max(0, math.Min(1, 0.3 + rand.Float64()*0.2)), // length
			}
			
			// 15% of spam looks legitimate
			if rand.Float64() < 0.15 {
				features[0] -= 2.0 // Fewer spam words
				features[1] += 3.0 // More ham words
				features[3] = 0.3  // Lower caps
			}
		}
		
		samples[i] = data.Sample{
			Features: features,
			Label:    label,
			ID:       fmt.Sprintf("email_%d", i),
		}
	}
	
	return data.NewDataset(samples)
}