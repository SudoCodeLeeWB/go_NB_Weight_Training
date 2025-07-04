package optimizer

import (
	"fmt"
	"math/rand"
)

// DifferentialEvolution implements the DE optimization algorithm
type DifferentialEvolution struct{}

// NewDifferentialEvolution creates a new DE optimizer
func NewDifferentialEvolution() *DifferentialEvolution {
	return &DifferentialEvolution{}
}

// GetName returns the optimizer name
func (de *DifferentialEvolution) GetName() string {
	return "DifferentialEvolution"
}

// Optimize implements the Optimizer interface
func (de *DifferentialEvolution) Optimize(objectiveFunc ObjectiveFunc, numWeights int, config *Config) (*Result, error) {
	if config == nil {
		config = DefaultConfig()
	}
	
	if numWeights <= 0 {
		return nil, fmt.Errorf("numWeights must be positive")
	}
	
	if config.PopulationSize < 4 {
		return nil, fmt.Errorf("population size must be at least 4 for DE")
	}
	
	rng := rand.New(rand.NewSource(config.RandomSeed))
	
	// Initialize population
	population := InitializePopulation(config.PopulationSize, numWeights, 
		config.MinWeight, config.MaxWeight, rng)
	
	// Evaluate initial population
	EvaluatePopulation(population, objectiveFunc)
	
	// Track results
	result := &Result{
		ScoreHistory:  make([]float64, 0, config.MaxIterations),
		WeightHistory: make([][]float64, 0, config.MaxIterations),
		Converged:     false,
	}
	
	// Main optimization loop
	for iter := 0; iter < config.MaxIterations; iter++ {
		// Create new population
		newPopulation := make(Population, config.PopulationSize)
		
		for i := range population {
			// Select three random individuals (different from current)
			r1, r2, r3 := de.selectRandomIndices(i, config.PopulationSize, rng)
			
			// Create trial vector
			trial := de.createTrialVector(
				population[i].Weights,
				population[r1].Weights,
				population[r2].Weights,
				population[r3].Weights,
				config.MutationFactor,
				config.CrossoverProb,
				config.MinWeight,
				config.MaxWeight,
				rng,
			)
			
			// Evaluate trial
			trialScore := objectiveFunc(trial)
			
			// Selection
			if trialScore > population[i].Score {
				newPopulation[i] = Individual{Weights: trial, Score: trialScore}
			} else {
				newPopulation[i] = population[i]
			}
		}
		
		population = newPopulation
		
		// Find best individual
		best := population.FindBest()
		
		// Record history
		bestWeightsCopy := make([]float64, len(best.Weights))
		copy(bestWeightsCopy, best.Weights)
		result.ScoreHistory = append(result.ScoreHistory, best.Score)
		result.WeightHistory = append(result.WeightHistory, bestWeightsCopy)
		
		// Callback
		if config.Callback != nil {
			config.Callback(iter, best.Score, best.Weights)
		}
		
		// Check convergence
		if len(result.ScoreHistory) >= 10 {
			if HasConverged(result.ScoreHistory, config.Tolerance, 10) {
				result.Converged = true
				result.Iterations = iter + 1
				break
			}
		}
	}
	
	// Set final results
	best := population.FindBest()
	result.BestWeights = best.Weights
	result.BestScore = best.Score
	if !result.Converged {
		result.Iterations = config.MaxIterations
	}
	
	return result, nil
}

// selectRandomIndices selects three unique random indices different from current
func (de *DifferentialEvolution) selectRandomIndices(current, popSize int, rng *rand.Rand) (int, int, int) {
	indices := make([]int, 0, popSize-1)
	for i := 0; i < popSize; i++ {
		if i != current {
			indices = append(indices, i)
		}
	}
	
	// Shuffle and pick first three
	rng.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})
	
	return indices[0], indices[1], indices[2]
}

// createTrialVector creates a trial vector using DE/rand/1/bin strategy
func (de *DifferentialEvolution) createTrialVector(
	target, base, diff1, diff2 []float64,
	mutationFactor, crossoverProb, minWeight, maxWeight float64,
	rng *rand.Rand) []float64 {
	
	n := len(target)
	trial := make([]float64, n)
	
	// Ensure at least one parameter is changed
	jrand := rng.Intn(n)
	
	for j := 0; j < n; j++ {
		if rng.Float64() < crossoverProb || j == jrand {
			// Mutation: base + F * (diff1 - diff2)
			trial[j] = base[j] + mutationFactor*(diff1[j]-diff2[j])
			trial[j] = ClipWeight(trial[j], minWeight, maxWeight)
		} else {
			// Keep original
			trial[j] = target[j]
		}
	}
	
	return trial
}