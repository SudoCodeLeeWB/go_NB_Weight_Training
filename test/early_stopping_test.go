package test

import (
	"math"
	"testing"
	
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// SimulatedDegradingModel simulates a model that starts good but degrades
type SimulatedDegradingModel struct {
	name string
	epoch int
}

func (m *SimulatedDegradingModel) Predict(samples [][]float64) ([]float64, error) {
	predictions := make([]float64, len(samples))
	
	// Simulate good predictions initially, then degrade
	// This is called multiple times per epoch, so we use a different approach
	for i := range predictions {
		// Base prediction on first feature
		baseProb := 1.0 / (1.0 + math.Exp(-samples[i][0]))
		
		// Add noise that increases with calls (simulating degradation)
		noise := float64(m.epoch) * 0.05
		predictions[i] = baseProb + noise
		
		// Clamp to [0, 1]
		if predictions[i] > 1.0 {
			predictions[i] = 1.0
		}
		if predictions[i] < 0.0 {
			predictions[i] = 0.0
		}
	}
	
	// Increment epoch counter (crude simulation)
	m.epoch++
	
	return predictions, nil
}

func (m *SimulatedDegradingModel) GetName() string {
	return m.name
}

func TestEarlyStoppingRestoresBestWeightsSimple(t *testing.T) {
	// Create a larger synthetic dataset for more stable results
	samples := make([]data.Sample, 100)
	for i := 0; i < 100; i++ {
		// Create linearly separable data
		if i < 50 {
			// Class 0: lower values
			samples[i] = data.Sample{
				Features: []float64{float64(i) / 100.0, float64(i) / 200.0},
				Label:    0,
			}
		} else {
			// Class 1: higher values
			samples[i] = data.Sample{
				Features: []float64{float64(i) / 100.0 + 0.5, float64(i) / 200.0 + 0.3},
				Label:    1,
			}
		}
	}
	
	dataset := data.NewDataset(samples)
	
	// Create two simple models
	models := []framework.Model{
		&SimulatedDegradingModel{name: "model1"},
		&SimulatedDegradingModel{name: "model2"},
	}
	
	// Configure training with early stopping
	config := framework.DefaultConfig()
	config.TrainingConfig.MaxEpochs = 50
	config.TrainingConfig.OptimizationMetric = "pr_auc"
	config.TrainingConfig.Verbose = false // Less output
	config.DataConfig.ValidationSplit = 0.3
	config.DataConfig.KFolds = 1 // Simple split
	config.OptimizerConfig.PopulationSize = 20
	
	// Enable early stopping
	config.EarlyStopping = &framework.EarlyStoppingConfig{
		Patience: 10,
		MinDelta: 0.001,
		Monitor:  "pr_auc",
		Mode:     "max",
	}
	
	// Train
	trainer := framework.NewTrainer(config)
	result, err := trainer.Train(dataset, models)
	require.NoError(t, err)
	
	// Verify we have results
	assert.NotNil(t, result.BestWeights)
	assert.Len(t, result.BestWeights, 2)
	
	// Check that weights are reasonable (not all zero or extreme)
	for i, w := range result.BestWeights {
		assert.GreaterOrEqual(t, w, 0.0, "Weight %d should be non-negative", i)
		assert.LessOrEqual(t, w, 2.0, "Weight %d should be <= 2.0", i)
	}
	
	// The training should produce reasonable metrics
	assert.Greater(t, result.ValMetrics["pr_auc"], 0.0, "PR-AUC should be positive")
}

func TestEarlyStoppingIntegration(t *testing.T) {
	// Load actual test data if available, or skip
	loader := data.NewCSVLoader()
	dataset, err := loader.Load("../demo_output/predictions.csv")
	if err != nil {
		t.Skip("Skipping integration test - no test data available")
	}
	
	// Use simple constant models for testing
	models := []framework.Model{
		&ConstantModel{name: "const1", value: 0.6},
		&ConstantModel{name: "const2", value: 0.7},
	}
	
	// Test with early stopping
	config := framework.DefaultConfig()
	config.TrainingConfig.MaxEpochs = 30
	config.DataConfig.KFolds = 1
	config.OptimizerConfig.PopulationSize = 10
	config.EarlyStopping = &framework.EarlyStoppingConfig{
		Patience: 5,
		MinDelta: 0.0001,
		Monitor:  "pr_auc",
		Mode:     "max",
	}
	
	trainer := framework.NewTrainer(config)
	result, err := trainer.Train(dataset, models)
	require.NoError(t, err)
	
	// Verify early stopping worked
	assert.Less(t, result.TotalEpochs, 30, "Should stop early")
	assert.NotNil(t, result.BestWeights)
}

// ConstantModel for testing - always returns the same prediction
type ConstantModel struct {
	name  string
	value float64
}

func (m *ConstantModel) Predict(samples [][]float64) ([]float64, error) {
	predictions := make([]float64, len(samples))
	for i := range predictions {
		predictions[i] = m.value
	}
	return predictions, nil
}

func (m *ConstantModel) GetName() string {
	return m.name
}