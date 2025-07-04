package main

import (
	"fmt"
	"log"

	"github.com/iwonbin/go-nb-weight-training/models"
	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/metrics"
	"github.com/iwonbin/go-nb-weight-training/pkg/optimizer"
)

func main() {
	fmt.Println("=== Detailed Training Progress Demo ===\n")

	// Create dataset
	dataset := createRealisticDataset(300)
	fmt.Printf("Dataset: %d samples (%.1f%% spam)\n\n", 
		dataset.NumSamples, dataset.ClassBalance()*100)

	// Split data
	splitter := data.NewStratifiedSplitter(0.3, 42)
	split, _ := splitter.Split(dataset)
	
	fmt.Printf("Train set: %d samples\n", split.Train.NumSamples)
	fmt.Printf("Test set: %d samples\n\n", split.Test.NumSamples)

	// Create models
	models := []framework.Model{
		models.NewBayesianSpamDetector(),
		models.NewNeuralNetSpamDetector(),
		models.NewSVMSpamDetector(),
		models.NewRandomForestSpamDetector(),
	}

	// Direct optimization to see progress
	fmt.Println("Training Progress:")
	fmt.Println("Iter | Best Score | Current Score | Improvement")
	fmt.Println("-----|------------|---------------|-------------")

	// Create optimizer with callback
	var lastScore float64
	optConfig := &optimizer.Config{
		MaxIterations:  30,
		PopulationSize: 15,
		MutationFactor: 0.8,
		CrossoverProb:  0.9,
		MinWeight:      0.0,
		MaxWeight:      2.0,
		RandomSeed:     42,
		Callback: func(iter int, bestScore float64, weights []float64) {
			improvement := bestScore - lastScore
			if iter == 0 {
				improvement = 0
			}
			fmt.Printf("%4d | %10.4f | %13.4f | %+11.4f\n", 
				iter, bestScore, bestScore, improvement)
			lastScore = bestScore
		},
	}

	// Create ensemble
	ensemble := &framework.EnsembleModel{
		Models: models,
	}

	// Create objective function
	trainFeatures := split.Train.GetFeatures()
	trainLabels := split.Train.GetLabels()
	
	objectiveFunc := func(weights []float64) float64 {
		ensemble.Weights = weights
		predictions, _ := ensemble.Predict(trainFeatures)
		
		// Calculate PR-AUC
		prauc := &metrics.PRAUC{}
		score, _ := prauc.Calculate(predictions, trainLabels)
		return score
	}

	// Optimize
	opt := optimizer.NewDifferentialEvolution()
	result, err := opt.Optimize(objectiveFunc, len(models), optConfig)
	if err != nil {
		log.Fatal(err)
	}

	// Show results
	fmt.Printf("\n=== Optimization Complete ===\n")
	fmt.Printf("Final PR-AUC: %.4f\n", result.BestScore)
	fmt.Printf("Converged: %v\n", result.Converged)
	fmt.Printf("Total iterations: %d\n\n", result.Iterations)

	fmt.Println("Optimized Weights:")
	for i, w := range result.BestWeights {
		fmt.Printf("  %s: %.4f\n", models[i].GetName(), w)
	}

	// Test on holdout set
	ensemble.Weights = result.BestWeights
	testFeatures := split.Test.GetFeatures()
	testLabels := split.Test.GetLabels()
	
	predictions, _ := ensemble.Predict(testFeatures)
	
	// Calculate test accuracy
	correct := 0
	for i := range predictions {
		pred := 0.0
		if predictions[i] > 0.5 {
			pred = 1.0
		}
		if pred == testLabels[i] {
			correct++
		}
	}
	
	fmt.Printf("\nTest Set Performance:\n")
	fmt.Printf("  Accuracy: %.2f%%\n", float64(correct)/float64(len(predictions))*100)
	
	// Calculate test PR-AUC
	prauc := &metrics.PRAUC{}
	testPRAUC, _ := prauc.Calculate(predictions, testLabels)
	fmt.Printf("  PR-AUC: %.4f\n", testPRAUC)
}

func createRealisticDataset(n int) *data.Dataset {
	samples := make([]data.Sample, n)
	
	// Create more realistic spam/ham patterns
	for i := 0; i < n; i++ {
		var features []float64
		var label float64
		
		r := float64(i) / float64(n)
		
		if i < n*65/100 { // 65% ham
			// Ham patterns with some noise
			spamWords := 1.0 + r*2 + randNoise()*2
			hamWords := 7.0 + r*3 - randNoise()*2
			exclamations := 0.5 + randNoise()
			capsRatio := 0.1 + randNoise()*0.2
			length := 0.6 + randNoise()*0.3
			
			features = []float64{spamWords, hamWords, exclamations, capsRatio, length}
			label = 0.0
		} else { // 35% spam
			// Spam patterns with some noise
			spamWords := 8.0 - r*2 + randNoise()*2
			hamWords := 2.0 - r + randNoise()*2
			exclamations := 4.0 + randNoise()*2
			capsRatio := 0.7 + randNoise()*0.2
			length := 0.3 + randNoise()*0.2
			
			features = []float64{spamWords, hamWords, exclamations, capsRatio, length}
			label = 1.0
		}
		
		samples[i] = data.Sample{
			Features: features,
			Label:    label,
		}
	}
	
	return data.NewDataset(samples)
}

var noiseSeed int64 = 123

func randNoise() float64 {
	noiseSeed = (noiseSeed*1103515245 + 12345) & 0x7fffffff
	return (float64(noiseSeed)/float64(0x7fffffff) - 0.5) * 0.5
}