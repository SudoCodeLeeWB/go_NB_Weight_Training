package models

import (
	"math/rand"
)

// SimpleClassifier is an example base classifier that users can replace
// with their own implementation
type SimpleClassifier struct {
	Name      string
	Threshold float64
	Noise     float64
	seed      int64
}

// NewSimpleClassifier creates a new simple classifier
func NewSimpleClassifier(name string, threshold, noise float64, seed int64) *SimpleClassifier {
	return &SimpleClassifier{
		Name:      name,
		Threshold: threshold,
		Noise:     noise,
		seed:      seed,
	}
}

// Predict implements the Model interface
// This is a dummy implementation - replace with your actual classifier
func (sc *SimpleClassifier) Predict(samples [][]float64) ([]float64, error) {
	rng := rand.New(rand.NewSource(sc.seed))
	predictions := make([]float64, len(samples))
	
	for i, sample := range samples {
		// Simple logic: if average of features > threshold, predict positive
		sum := 0.0
		for _, feature := range sample {
			sum += feature
		}
		avg := sum / float64(len(sample))
		
		// Base prediction
		if avg > sc.Threshold {
			predictions[i] = 0.7 + rng.Float64()*0.3 // High confidence positive
		} else {
			predictions[i] = rng.Float64() * 0.3 // Low confidence positive
		}
		
		// Add noise
		predictions[i] += (rng.Float64() - 0.5) * sc.Noise
		
		// Ensure in [0, 1] range
		if predictions[i] > 1 {
			predictions[i] = 1
		} else if predictions[i] < 0 {
			predictions[i] = 0
		}
	}
	
	return predictions, nil
}

// GetName returns the classifier name
func (sc *SimpleClassifier) GetName() string {
	return sc.Name
}

// LogisticClassifier is another example classifier
type LogisticClassifier struct {
	Name    string
	Weights []float64
	Bias    float64
}

// NewLogisticClassifier creates a new logistic classifier
func NewLogisticClassifier(name string, numFeatures int) *LogisticClassifier {
	// Initialize with random weights
	weights := make([]float64, numFeatures)
	for i := range weights {
		weights[i] = rand.Float64()*2 - 1 // Random between -1 and 1
	}
	
	return &LogisticClassifier{
		Name:    name,
		Weights: weights,
		Bias:    rand.Float64()*2 - 1,
	}
}

// Predict implements the Model interface
func (lc *LogisticClassifier) Predict(samples [][]float64) ([]float64, error) {
	predictions := make([]float64, len(samples))
	
	for i, sample := range samples {
		// Linear combination
		z := lc.Bias
		for j, feature := range sample {
			if j < len(lc.Weights) {
				z += feature * lc.Weights[j]
			}
		}
		
		// Sigmoid activation
		predictions[i] = 1.0 / (1.0 + exp(-z))
	}
	
	return predictions, nil
}

// GetName returns the classifier name
func (lc *LogisticClassifier) GetName() string {
	return lc.Name
}

// Helper function for exponential
func exp(x float64) float64 {
	// Simple approximation for demonstration
	// In production, use math.Exp
	if x > 10 {
		return 22026.465794806718
	}
	if x < -10 {
		return 0.00004539992976248485
	}
	
	// Taylor series approximation
	result := 1.0
	term := 1.0
	for i := 1; i <= 20; i++ {
		term *= x / float64(i)
		result += term
	}
	return result
}