package main

import (
	"fmt"
	"log"
	"os"

	"github.com/iwonbin/go-nb-weight-training/models"
	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
)

// ProductionExample shows how to use the framework in production
func main() {
	fmt.Println("=== Production Usage Example ===\n")

	// 1. Load your data
	dataset, err := data.LoadData("training_data.csv")
	if err != nil {
		// For demo, create sample data
		dataset = createSampleData()
	}

	// 2. Split data properly
	splitter := data.NewStratifiedSplitter(0.2, 42)
	split, _ := splitter.Split(dataset)
	
	// Further split train into train/val
	trainSplitter := data.NewStratifiedSplitter(0.25, 42)
	trainValSplit, _ := trainSplitter.Split(split.Train)

	fmt.Printf("Data splits: Train=%d, Val=%d, Test=%d\n\n",
		trainValSplit.Train.NumSamples,
		trainValSplit.Test.NumSamples,
		split.Test.NumSamples)

	// 3. Create your models (wrap with validation)
	baseModels := []framework.Model{
		&framework.SafeModel{Model: models.NewBayesianSpamDetector()},
		&framework.SafeModel{Model: models.NewNeuralNetSpamDetector()},
		&framework.SafeModel{Model: models.NewSVMSpamDetector()},
	}

	// 4. Train ensemble
	config := framework.DefaultConfig()
	config.TrainingConfig.MaxEpochs = 30
	config.DataConfig.KFolds = 2 // Faster for demo

	trainer := framework.NewTrainer(config)
	result, err := trainer.Train(trainValSplit.Train, baseModels)
	if err != nil {
		log.Fatal(err)
	}

	// 5. Create calibrated ensemble
	ensemble := &framework.CalibratedEnsemble{
		EnsembleModel: &framework.EnsembleModel{
			Models:  baseModels,
			Weights: result.BestWeights,
		},
		UseLogSpace: true, // Important for numerical stability
	}

	// 6. Fit calibration on validation set
	fmt.Println("Fitting calibration...")
	valFeatures := trainValSplit.Test.GetFeatures()
	valLabels := trainValSplit.Test.GetLabels()
	
	err = ensemble.FitCalibration(valFeatures, valLabels)
	if err != nil {
		log.Fatal(err)
	}

	// 7. Find optimal threshold
	calibratedPreds, _ := ensemble.PredictCalibrated(valFeatures)
	threshold, score := framework.FindOptimalThreshold(calibratedPreds, valLabels, "f1")
	
	fmt.Printf("\nOptimal threshold: %.3f (F1 score: %.3f)\n", threshold, score)

	// 8. Test on holdout set
	testFeatures := split.Test.GetFeatures()
	testLabels := split.Test.GetLabels()
	
	testPreds, err := ensemble.PredictCalibrated(testFeatures)
	if err != nil {
		log.Fatal(err)
	}

	// 9. Show calibrated predictions
	fmt.Println("\nCalibrated Predictions (now in proper 0-1 range):")
	fmt.Println("True | Raw Score | Calibrated | Binary@Threshold")
	fmt.Println("-----|-----------|------------|------------------")
	
	// Also get raw scores for comparison
	ensemble.UseLogSpace = false
	rawPreds, _ := ensemble.EnsembleModel.Predict(testFeatures[:10])
	ensemble.UseLogSpace = true
	
	for i := 0; i < 10 && i < len(testPreds); i++ {
		label := "ham "
		if testLabels[i] > 0.5 {
			label = "spam"
		}
		
		binary := "ham "
		if testPreds[i] >= threshold {
			binary = "spam"
		}
		
		fmt.Printf("%s |  %.6f |   %.3f    |      %s\n",
			label, rawPreds[i], testPreds[i], binary)
	}

	// 10. Calculate final metrics
	correct := 0
	tp, fp, fn := 0, 0, 0
	
	for i := range testPreds {
		pred := 0.0
		if testPreds[i] >= threshold {
			pred = 1.0
		}
		
		if pred == testLabels[i] {
			correct++
		}
		
		if pred == 1 && testLabels[i] == 1 {
			tp++
		} else if pred == 1 && testLabels[i] == 0 {
			fp++
		} else if pred == 0 && testLabels[i] == 1 {
			fn++
		}
	}
	
	accuracy := float64(correct) / float64(len(testPreds))
	precision := float64(tp) / float64(tp+fp+1)
	recall := float64(tp) / float64(tp+fn+1)
	f1 := 2 * precision * recall / (precision + recall + 1e-10)
	
	fmt.Printf("\nTest Set Performance (threshold=%.3f):\n", threshold)
	fmt.Printf("  Accuracy:  %.2f%%\n", accuracy*100)
	fmt.Printf("  Precision: %.2f%%\n", precision*100)
	fmt.Printf("  Recall:    %.2f%%\n", recall*100)
	fmt.Printf("  F1 Score:  %.2f%%\n", f1*100)

	// 11. Save for production use
	err = framework.SaveEnsemble(ensemble.EnsembleModel, config, "model.json")
	if err == nil {
		fmt.Println("\nModel saved to: model.json")
	}

	// 12. Production serving example
	fmt.Println("\n=== Production Serving Example ===")
	
	// Create batch processor for efficiency
	batchProcessor := framework.NewBatchProcessor(ensemble, 100)
	
	// Simulate incoming requests
	newSamples := [][]float64{
		{8, 2, 5, 0.8, 0.3},  // Likely spam
		{2, 8, 1, 0.1, 0.7},  // Likely ham
		{5, 5, 3, 0.5, 0.5},  // Ambiguous
	}
	
	predictions, err := batchProcessor.PredictBatch(newSamples)
	if err != nil {
		log.Fatal(err)
	}
	
	fmt.Println("\nProduction predictions:")
	for i, pred := range predictions {
		decision := "HAM"
		confidence := "LOW"
		
		if pred >= threshold {
			decision = "SPAM"
		}
		
		// Confidence based on distance from threshold
		distance := abs(pred - threshold)
		if distance > 0.3 {
			confidence = "HIGH"
		} else if distance > 0.1 {
			confidence = "MEDIUM"
		}
		
		fmt.Printf("Sample %d: %s (score=%.3f, confidence=%s)\n",
			i+1, decision, pred, confidence)
	}
}

func createSampleData() *data.Dataset {
	// Create sample data for demo
	samples := make([]data.Sample, 200)
	for i := 0; i < 200; i++ {
		var features []float64
		var label float64
		
		if i%2 == 0 {
			// Ham
			features = []float64{
				float64(1 + i%3),    // Low spam words
				float64(7 + i%3),    // High ham words
				float64(i % 2),      // Low exclamations
				0.1 + float64(i%10)/100, // Low caps
				0.6,                 // Medium length
			}
			label = 0.0
		} else {
			// Spam
			features = []float64{
				float64(7 + i%3),    // High spam words
				float64(1 + i%3),    // Low ham words
				float64(3 + i%3),    // High exclamations
				0.7 + float64(i%10)/100, // High caps
				0.3,                 // Short length
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

func abs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}