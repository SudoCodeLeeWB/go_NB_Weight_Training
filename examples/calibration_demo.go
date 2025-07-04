package main

import (
	"fmt"
	"log"
	"sort"

	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/metrics"
	"github.com/iwonbin/go-nb-weight-training/models"
)

func main() {
	fmt.Println("=== Calibration Demo: Before vs After ===\n")

	// Create dataset
	dataset := createDataset()

	// Create models
	baseModels := []framework.Model{
		models.NewSimpleClassifier("Model1", 0.5, 0.15, 42),
		models.NewSimpleClassifier("Model2", 0.6, 0.1, 43),
		models.NewLogisticClassifier("Model3", 3),
	}

	// Train
	config := framework.DefaultConfig()
	config.TrainingConfig.MaxEpochs = 50
	config.DataConfig.KFolds = 1
	config.TrainingConfig.Verbose = false
	config.DataConfig.ValidationSplit = 0.3
	
	trainer := framework.NewTrainer(config)
	result, err := trainer.Train(dataset, baseModels)
	if err != nil {
		log.Fatal(err)
	}

	// Create ensemble
	ensemble := &framework.EnsembleModel{
		Models:  baseModels,
		Weights: result.BestWeights,
	}

	// Get predictions on test data
	testData := createTestDataset()
	features := testData.GetFeatures()
	labels := testData.GetLabels()
	
	// Get uncalibrated predictions
	uncalibratedPreds, _ := ensemble.Predict(features)

	// Create calibrated ensemble
	calibratedEnsemble := &framework.CalibratedEnsemble{
		EnsembleModel: ensemble,
		UseLogSpace: false,  // Use regular space for now
	}
	
	// Fit calibration on validation data
	valFeatures := dataset.GetFeatures()[:30]  // Use first 30 for calibration
	valLabels := dataset.GetLabels()[:30]
	
	err = calibratedEnsemble.FitCalibration(valFeatures, valLabels)
	if err != nil {
		fmt.Printf("Error fitting calibration: %v\n", err)
		// Use simple min-max scaling as fallback
		calibratedEnsemble = createSimplyCalibratedEnsemble(ensemble, uncalibratedPreds)
	}
	
	// Get calibrated predictions
	calibratedPreds, _ := calibratedEnsemble.PredictCalibrated(features)

	// Compare distributions
	fmt.Println("1. PREDICTION DISTRIBUTIONS")
	fmt.Println("---------------------------")
	fmt.Println("\nUncalibrated predictions:")
	analyzePredictions(uncalibratedPreds)
	
	fmt.Println("\nCalibrated predictions:")
	analyzePredictions(calibratedPreds)

	// Compare metrics at threshold 0.5
	fmt.Println("\n2. METRICS AT THRESHOLD 0.5")
	fmt.Println("---------------------------")
	
	uncalCM := metrics.CalculateConfusionMatrix(uncalibratedPreds, labels, 0.5)
	calCM := metrics.CalculateConfusionMatrix(calibratedPreds, labels, 0.5)
	
	fmt.Printf("\nUncalibrated at 0.5 threshold:\n")
	fmt.Printf("Precision: %.4f, Recall: %.4f, F1: %.4f\n", 
		uncalCM.Precision(), uncalCM.Recall(), uncalCM.F1Score())
	fmt.Printf("TP=%d, FP=%d, TN=%d, FN=%d\n",
		uncalCM.TruePositives, uncalCM.FalsePositives, 
		uncalCM.TrueNegatives, uncalCM.FalseNegatives)
	
	fmt.Printf("\nCalibrated at 0.5 threshold:\n")
	fmt.Printf("Precision: %.4f, Recall: %.4f, F1: %.4f\n", 
		calCM.Precision(), calCM.Recall(), calCM.F1Score())
	fmt.Printf("TP=%d, FP=%d, TN=%d, FN=%d\n",
		calCM.TruePositives, calCM.FalsePositives, 
		calCM.TrueNegatives, calCM.FalseNegatives)

	// Show PR-AUC (should be similar)
	fmt.Println("\n3. PR-AUC COMPARISON")
	fmt.Println("--------------------")
	uncalPRAUC := calculatePRAUC(uncalibratedPreds, labels)
	calPRAUC := calculatePRAUC(calibratedPreds, labels)
	fmt.Printf("Uncalibrated PR-AUC: %.4f\n", uncalPRAUC)
	fmt.Printf("Calibrated PR-AUC: %.4f\n", calPRAUC)
	
	// Show example predictions
	fmt.Println("\n4. EXAMPLE PREDICTIONS")
	fmt.Println("----------------------")
	fmt.Println("Sample | True | Uncalibrated | Calibrated | Difference")
	fmt.Println("-------|------|--------------|------------|------------")
	for i := 0; i < 10 && i < len(labels); i++ {
		diff := calibratedPreds[i] - uncalibratedPreds[i]
		fmt.Printf("%6d | %4.0f | %12.6f | %10.4f | %+10.4f\n", 
			i, labels[i], uncalibratedPreds[i], calibratedPreds[i], diff)
	}
	
	fmt.Println("\n=== KEY INSIGHT ===")
	fmt.Println("After calibration, predictions are spread across [0,1] range")
	fmt.Println("Now threshold 0.5 makes sense and precision is meaningful!")
}

func createDataset() *data.Dataset {
	samples := []data.Sample{}
	
	// Clear positive examples
	for i := 0; i < 40; i++ {
		samples = append(samples, data.Sample{
			Features: []float64{0.7 + float64(i%5)*0.02, 0.75 - float64(i%3)*0.01, 0.72},
			Label:    1,
		})
	}
	
	// Clear negative examples
	for i := 0; i < 40; i++ {
		samples = append(samples, data.Sample{
			Features: []float64{0.3 - float64(i%5)*0.02, 0.25 + float64(i%3)*0.01, 0.28},
			Label:    0,
		})
	}
	
	// Some borderline cases
	for i := 0; i < 20; i++ {
		label := float64(i % 2)
		samples = append(samples, data.Sample{
			Features: []float64{0.45 + float64(i%3)*0.05, 0.5, 0.48},
			Label:    label,
		})
	}
	
	return data.NewDataset(samples)
}

func createTestDataset() *data.Dataset {
	samples := []data.Sample{}
	
	// Test positive examples
	for i := 0; i < 15; i++ {
		samples = append(samples, data.Sample{
			Features: []float64{0.65 + float64(i%3)*0.03, 0.7, 0.68},
			Label:    1,
		})
	}
	
	// Test negative examples
	for i := 0; i < 15; i++ {
		samples = append(samples, data.Sample{
			Features: []float64{0.35 - float64(i%3)*0.03, 0.3, 0.32},
			Label:    0,
		})
	}
	
	return data.NewDataset(samples)
}

func analyzePredictions(predictions []float64) {
	sorted := make([]float64, len(predictions))
	copy(sorted, predictions)
	sort.Float64s(sorted)
	
	min := sorted[0]
	max := sorted[len(sorted)-1]
	q1 := sorted[len(sorted)/4]
	median := sorted[len(sorted)/2]
	q3 := sorted[3*len(sorted)/4]
	
	above50 := 0
	for _, p := range predictions {
		if p > 0.5 {
			above50++
		}
	}
	
	fmt.Printf("Range: [%.6f, %.6f]\n", min, max)
	fmt.Printf("Quartiles: Q1=%.4f, Median=%.4f, Q3=%.4f\n", q1, median, q3)
	fmt.Printf("Predictions > 0.5: %d/%d (%.1f%%)\n", 
		above50, len(predictions), float64(above50)*100/float64(len(predictions)))
}

func calculatePRAUC(predictions, labels []float64) float64 {
	prauc := &metrics.PRAUC{}
	score, _ := prauc.Calculate(predictions, labels)
	return score
}

// Simple min-max calibration as fallback
func createSimplyCalibratedEnsemble(ensemble *framework.EnsembleModel, predictions []float64) *framework.CalibratedEnsemble {
	// Find min and max
	min, max := predictions[0], predictions[0]
	for _, p := range predictions {
		if p < min {
			min = p
		}
		if p > max {
			max = p
		}
	}
	
	// Create a simple scaling calibrated ensemble with min-max normalization
	return &framework.CalibratedEnsemble{
		EnsembleModel: ensemble,
		UseLogSpace: false,
		ScoreRange: framework.ScoreRange{
			MinScore: min,
			MaxScore: max,
		},
		CalibrationFunc: nil, // Will use default min-max normalization
	}
}