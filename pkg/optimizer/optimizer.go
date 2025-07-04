package optimizer

import (
	"math/rand"
)

// Optimizer interface for weight optimization algorithms
type Optimizer interface {
	// Optimize finds optimal weights for the given objective function
	// objectiveFunc should return a score to maximize
	Optimize(objectiveFunc ObjectiveFunc, numWeights int, config *Config) (*Result, error)
	
	// GetName returns the optimizer name
	GetName() string
}

// ObjectiveFunc is the function to optimize (returns score to maximize)
type ObjectiveFunc func(weights []float64) float64

// Config holds optimizer configuration
type Config struct {
	// Common parameters
	MaxIterations int
	Tolerance     float64
	RandomSeed    int64
	
	// Bounds for weights
	MinWeight float64
	MaxWeight float64
	
	// Algorithm-specific parameters
	PopulationSize int     // For evolutionary algorithms
	MutationFactor float64 // For differential evolution
	CrossoverProb  float64 // For differential evolution
	
	// Enforce non-zero weights
	EnforceNonZero bool
	
	// Callback for progress updates
	Callback ProgressCallback
}

// ProgressCallback is called after each iteration
type ProgressCallback func(iteration int, bestScore float64, bestWeights []float64)

// Result holds optimization results
type Result struct {
	BestWeights   []float64
	BestScore     float64
	Iterations    int
	Converged     bool
	ScoreHistory  []float64
	WeightHistory [][]float64
}

// DefaultConfig returns default optimizer configuration
func DefaultConfig() *Config {
	return &Config{
		MaxIterations:  100,
		Tolerance:      1e-6,
		RandomSeed:     42,
		MinWeight:      0.0,
		MaxWeight:      2.0,
		PopulationSize: 50,
		MutationFactor: 0.8,
		CrossoverProb:  0.9,
	}
}

// Individual represents a candidate solution
type Individual struct {
	Weights []float64
	Score   float64
}

// Population represents a collection of individuals
type Population []Individual

// InitializePopulation creates random initial population
func InitializePopulation(size, numWeights int, minWeight, maxWeight float64, rng *rand.Rand) Population {
	pop := make(Population, size)
	
	// Ensure minWeight is not exactly 0 for multiplicative ensembles
	if minWeight == 0 {
		minWeight = 0.01
	}
	
	for i := range pop {
		weights := make([]float64, numWeights)
		for j := range weights {
			weights[j] = minWeight + rng.Float64()*(maxWeight-minWeight)
		}
		pop[i] = Individual{Weights: weights}
	}
	
	return pop
}

// EvaluatePopulation computes scores for all individuals
func EvaluatePopulation(pop Population, objectiveFunc ObjectiveFunc) {
	for i := range pop {
		pop[i].Score = objectiveFunc(pop[i].Weights)
	}
}

// FindBest returns the best individual in the population
func (pop Population) FindBest() Individual {
	best := pop[0]
	for _, ind := range pop[1:] {
		if ind.Score > best.Score {
			best = ind
		}
	}
	return best
}

// ClipWeight ensures weight is within bounds
func ClipWeight(weight, minWeight, maxWeight float64) float64 {
	if weight < minWeight {
		return minWeight
	}
	if weight > maxWeight {
		return maxWeight
	}
	return weight
}

// ClipWeightWithEnforcement ensures weight is within bounds and optionally enforces non-zero
func ClipWeightWithEnforcement(weight, minWeight, maxWeight float64, enforceNonZero bool) float64 {
	if enforceNonZero && minWeight <= 0 {
		minWeight = 0.01 // Enforce minimum non-zero weight
	}
	
	if weight < minWeight {
		return minWeight
	}
	if weight > maxWeight {
		return maxWeight
	}
	return weight
}

// HasConverged checks if optimization has converged
func HasConverged(history []float64, tolerance float64, window int) bool {
	if len(history) < window {
		return false
	}
	
	// Check if improvement over last 'window' iterations is below tolerance
	recent := history[len(history)-window:]
	minScore := recent[0]
	maxScore := recent[0]
	
	for _, score := range recent[1:] {
		if score < minScore {
			minScore = score
		}
		if score > maxScore {
			maxScore = score
		}
	}
	
	return (maxScore - minScore) < tolerance
}