package optimizer

import (
	"fmt"
	"math/rand"
)

// RandomSearch implements random search optimization
type RandomSearch struct{}

// NewRandomSearch creates a new random search optimizer
func NewRandomSearch() *RandomSearch {
	return &RandomSearch{}
}

// GetName returns the optimizer name
func (rs *RandomSearch) GetName() string {
	return "RandomSearch"
}

// Optimize implements the Optimizer interface
func (rs *RandomSearch) Optimize(objectiveFunc ObjectiveFunc, numWeights int, config *Config) (*Result, error) {
	if config == nil {
		config = DefaultConfig()
	}
	
	if numWeights <= 0 {
		return nil, fmt.Errorf("numWeights must be positive")
	}
	
	rng := rand.New(rand.NewSource(config.RandomSeed))
	
	// Initialize result
	result := &Result{
		ScoreHistory:  make([]float64, 0, config.MaxIterations),
		WeightHistory: make([][]float64, 0, config.MaxIterations),
		BestWeights:   make([]float64, numWeights),
		BestScore:     -1e9, // Initialize with very low score
		Converged:     false,
	}
	
	// Random search
	for iter := 0; iter < config.MaxIterations; iter++ {
		// Generate random weights
		weights := make([]float64, numWeights)
		for j := range weights {
			weights[j] = config.MinWeight + rng.Float64()*(config.MaxWeight-config.MinWeight)
		}
		
		// Evaluate
		score := objectiveFunc(weights)
		
		// Update best if improved
		if score > result.BestScore {
			result.BestScore = score
			copy(result.BestWeights, weights)
		}
		
		// Record history
		weightsCopy := make([]float64, len(weights))
		copy(weightsCopy, weights)
		result.ScoreHistory = append(result.ScoreHistory, score)
		result.WeightHistory = append(result.WeightHistory, weightsCopy)
		
		// Callback
		if config.Callback != nil {
			config.Callback(iter, result.BestScore, result.BestWeights)
		}
		
		// Check convergence (for random search, we check if best hasn't improved)
		if len(result.ScoreHistory) >= 20 {
			recentBest := result.BestScore
			oldBest := result.BestScore
			for i := len(result.ScoreHistory) - 20; i < len(result.ScoreHistory)-10; i++ {
				if result.ScoreHistory[i] > oldBest {
					oldBest = result.ScoreHistory[i]
				}
			}
			
			if recentBest-oldBest < config.Tolerance {
				result.Converged = true
				result.Iterations = iter + 1
				break
			}
		}
	}
	
	if !result.Converged {
		result.Iterations = config.MaxIterations
	}
	
	return result, nil
}