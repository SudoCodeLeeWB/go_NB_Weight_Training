package framework

import (
	"fmt"
	"math"
	
	"github.com/iwonbin/go-nb-weight-training/pkg/data"
)

// ValidateInput validates model inputs
func ValidateInput(samples [][]float64) error {
	if len(samples) == 0 {
		return fmt.Errorf("empty input samples")
	}
	
	numFeatures := len(samples[0])
	if numFeatures == 0 {
		return fmt.Errorf("empty feature vector")
	}
	
	for i, sample := range samples {
		if len(sample) != numFeatures {
			return fmt.Errorf("inconsistent feature count at sample %d: expected %d, got %d",
				i, numFeatures, len(sample))
		}
		
		for j, val := range sample {
			if math.IsNaN(val) || math.IsInf(val, 0) {
				return fmt.Errorf("invalid value at sample %d, feature %d: %v", i, j, val)
			}
		}
	}
	
	return nil
}

// ValidatePredictions validates model output
func ValidatePredictions(predictions []float64, numSamples int) error {
	if len(predictions) != numSamples {
		return fmt.Errorf("prediction count mismatch: expected %d, got %d",
			numSamples, len(predictions))
	}
	
	for i, pred := range predictions {
		if math.IsNaN(pred) || math.IsInf(pred, 0) {
			return fmt.Errorf("invalid prediction at index %d: %v", i, pred)
		}
		if pred < 0 || pred > 1 {
			return fmt.Errorf("prediction out of range [0,1] at index %d: %v", i, pred)
		}
	}
	
	return nil
}

// SafeModel wraps a model with validation
type SafeModel struct {
	Model
}

func (sm *SafeModel) Predict(samples [][]float64) ([]float64, error) {
	// Validate input
	if err := ValidateInput(samples); err != nil {
		return nil, fmt.Errorf("input validation failed: %w", err)
	}
	
	// Call underlying model
	predictions, err := sm.Model.Predict(samples)
	if err != nil {
		return nil, err
	}
	
	// Validate output
	if err := ValidatePredictions(predictions, len(samples)); err != nil {
		return nil, fmt.Errorf("output validation failed: %w", err)
	}
	
	return predictions, nil
}

// ValidateDataset performs comprehensive validation on a dataset
func ValidateDataset(dataset *data.Dataset) error {
	if dataset == nil {
		return ErrNoData
	}
	
	if dataset.NumSamples == 0 {
		return fmt.Errorf("%w: dataset is empty", ErrNoData)
	}
	
	if dataset.NumFeatures == 0 {
		return fmt.Errorf("%w: no features in dataset", ErrInvalidData)
	}
	
	// Check for valid labels
	for i, sample := range dataset.Samples {
		if sample.Label != 0 && sample.Label != 1 {
			return fmt.Errorf("%w: sample %d has invalid label %.2f (must be 0 or 1)", 
				ErrInvalidData, i, sample.Label)
		}
		
		// Check for NaN or Inf in features
		for j, feat := range sample.Features {
			if math.IsNaN(feat) || math.IsInf(feat, 0) {
				return fmt.Errorf("%w: sample %d feature %d has invalid value %v", 
					ErrInvalidData, i, j, feat)
			}
		}
	}
	
	// Check class balance
	if len(dataset.ClassCounts) == 0 {
		return fmt.Errorf("%w: no class counts", ErrInvalidData)
	}
	
	// Ensure we have both classes
	if _, ok := dataset.ClassCounts[0.0]; !ok {
		return fmt.Errorf("%w: no negative samples (class 0) in dataset", ErrInvalidData)
	}
	
	if _, ok := dataset.ClassCounts[1.0]; !ok {
		return fmt.Errorf("%w: no positive samples (class 1) in dataset", ErrInvalidData)
	}
	
	// Warn about extreme class imbalance
	balance := dataset.ClassBalance()
	if balance < 0.01 || balance > 0.99 {
		// This is a warning, not an error
		fmt.Printf("WARNING: Extreme class imbalance detected (%.1f%% positive samples)\n", balance*100)
	}
	
	return nil
}

// ValidateModels validates a slice of models
func ValidateModels(models []Model) error {
	if len(models) == 0 {
		return ErrNoModels
	}
	
	// Check for nil models
	for i, model := range models {
		if model == nil {
			return fmt.Errorf("%w: model %d is nil", ErrNoModels, i)
		}
	}
	
	// Check for duplicate names
	names := make(map[string]int)
	for i, model := range models {
		name := model.GetName()
		if name == "" {
			return fmt.Errorf("%w: model %d has empty name", ErrInvalidData, i)
		}
		
		if prevIdx, exists := names[name]; exists {
			return fmt.Errorf("%w: duplicate model name '%s' at indices %d and %d", 
				ErrInvalidData, name, prevIdx, i)
		}
		names[name] = i
	}
	
	return nil
}

// ValidateWeights validates ensemble weights
func ValidateWeights(weights []float64, numModels int) error {
	if len(weights) != numModels {
		return fmt.Errorf("%w: expected %d weights, got %d", 
			ErrWeightModelMismatch, numModels, len(weights))
	}
	
	// Check for valid weight values
	for i, w := range weights {
		if math.IsNaN(w) || math.IsInf(w, 0) {
			return fmt.Errorf("weight %d is %v", i, w)
		}
		
		if w < 0 {
			return fmt.Errorf("weight %d is negative: %.4f", i, w)
		}
	}
	
	// Check if all weights are zero
	allZero := true
	for _, w := range weights {
		if w > 0 {
			allZero = false
			break
		}
	}
	
	if allZero {
		return fmt.Errorf("all weights are zero")
	}
	
	return nil
}

// SafeDivision performs division with zero check
func SafeDivision(numerator, denominator float64) float64 {
	if denominator == 0 {
		return 0
	}
	return numerator / denominator
}

// ClampValue ensures a value is within bounds
func ClampValue(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}