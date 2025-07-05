package framework

// AggregatedModel is the ONLY interface that users need to implement to use this framework.
// The framework treats this as a complete black box and only cares about optimizing weights.
// How you combine your models internally is entirely up to you.
type AggregatedModel interface {
	// Predict returns the aggregated predictions for samples
	// Input: samples [][]float64 - raw features
	// Output: []float64 - predictions [0,1] for positive class
	Predict(samples [][]float64) ([]float64, error)
	
	// GetWeights returns the current weights
	GetWeights() []float64
	
	// SetWeights updates the weights for optimization
	SetWeights(weights []float64) error
	
	// GetNumModels returns the number of models being aggregated
	GetNumModels() int
	
	// GetModelNames returns model names (optional, for reporting)
	GetModelNames() []string
}