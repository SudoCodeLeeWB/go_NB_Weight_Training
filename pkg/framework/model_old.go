package framework

import "math"

// DEPRECATED: This file contains the old Model interface architecture.
// New code should use AggregatedModel interface instead.
// This is kept for backward compatibility with old demos and tests.

// Model interface that users must implement for their classifiers
type Model interface {
	// Predict takes a batch of samples and returns probability scores (0-1)
	// Input: samples [][]float64 - each sample is a feature vector
	// Output: []float64 - probability scores for the positive class
	Predict(samples [][]float64) ([]float64, error)
	
	// GetName returns a descriptive name for the model
	GetName() string
}

// EnsembleModel represents a weighted ensemble of base models
type EnsembleModel struct {
	Models  []Model   // Base models
	Weights []float64 // Weights for each model
}

// Predict implements the Model interface for ensemble
func (em *EnsembleModel) Predict(samples [][]float64) ([]float64, error) {
	if len(em.Models) == 0 {
		return nil, ErrNoModels
	}
	
	if len(em.Models) != len(em.Weights) {
		return nil, ErrWeightModelMismatch
	}
	
	n := len(samples)
	result := make([]float64, n)
	
	// Initialize with 1.0 for multiplication (naive bayes style)
	for i := range result {
		result[i] = 1.0
	}
	
	// Get predictions from each model and apply weights
	for idx, model := range em.Models {
		predictions, err := model.Predict(samples)
		if err != nil {
			return nil, err
		}
		
		weight := em.Weights[idx]
		for i := range result {
			// Weighted multiplication (naive bayes aggregation)
			// Apply weight as exponent to maintain probability properties
			result[i] *= pow(predictions[i], weight)
		}
	}
	
	// Normalize if needed
	return normalizeScores(result), nil
}

// GetName returns the ensemble name
func (em *EnsembleModel) GetName() string {
	return "WeightedNaiveBayesEnsemble"
}

// GetNumModels returns the number of models in the ensemble
func (em *EnsembleModel) GetNumModels() int {
	return len(em.Models)
}

// GetModelNames returns the names of all models in the ensemble
func (em *EnsembleModel) GetModelNames() []string {
	names := make([]string, len(em.Models))
	for i, model := range em.Models {
		names[i] = model.GetName()
	}
	return names
}

// Helper function for power calculation
func pow(base, exp float64) float64 {
	if exp == 0 {
		return 1
	}
	if exp == 1 {
		return base
	}
	
	// Handle edge cases
	if base == 0 {
		return 0
	}
	
	// Use exp(ln(base) * exp) for fractional exponents
	return math.Exp(math.Log(base) * exp)
}

// normalizeScores ensures scores are valid probabilities
func normalizeScores(scores []float64) []float64 {
	result := make([]float64, len(scores))
	
	for i, score := range scores {
		// Ensure score is in [0, 1] range
		if score < 0 {
			result[i] = 0
		} else if score > 1 {
			result[i] = 1
		} else {
			result[i] = score
		}
	}
	
	return result
}