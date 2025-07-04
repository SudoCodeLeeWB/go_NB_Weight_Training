package test

import (
	"testing"
	
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestModel implements basic Model interface
type TestModel struct {
	name string
}

func (tm *TestModel) Predict(samples [][]float64) ([]float64, error) {
	predictions := make([]float64, len(samples))
	for i, sample := range samples {
		// Simple prediction: average of features
		sum := 0.0
		for _, f := range sample {
			sum += f
		}
		predictions[i] = sum / float64(len(sample))
	}
	return predictions, nil
}

func (tm *TestModel) GetName() string {
	return tm.name
}

// TestExtendedModel implements ExtendedModel interface
type TestExtendedModel struct {
	name string
}

func (tem *TestExtendedModel) Predict(samples [][]float64) ([]float64, error) {
	predictions := make([]float64, len(samples))
	for i, sample := range samples {
		pred, _ := tem.PredictSingle(sample)
		predictions[i] = pred
	}
	return predictions, nil
}

func (tem *TestExtendedModel) PredictSingle(sample []float64) (float64, error) {
	// Simple prediction: average of features
	sum := 0.0
	for _, f := range sample {
		sum += f
	}
	return sum / float64(len(sample)), nil
}

func (tem *TestExtendedModel) GetName() string {
	return tem.name
}

func TestSinglePredictAdapter(t *testing.T) {
	// Create a basic model
	model := &TestModel{name: "test"}
	
	// Convert to extended model
	extended := framework.ToExtended(model)
	
	// Test single prediction
	sample := []float64{0.2, 0.4, 0.6}
	prediction, err := extended.PredictSingle(sample)
	require.NoError(t, err)
	assert.InDelta(t, 0.4, prediction, 0.001)
	
	// Test batch prediction still works
	samples := [][]float64{
		{0.2, 0.4, 0.6},
		{0.1, 0.3, 0.5},
		{0.8, 0.9, 1.0},
	}
	predictions, err := extended.Predict(samples)
	require.NoError(t, err)
	assert.Len(t, predictions, 3)
	assert.InDelta(t, 0.4, predictions[0], 0.001)
	assert.InDelta(t, 0.3, predictions[1], 0.001)
	assert.InDelta(t, 0.9, predictions[2], 0.001)
}

func TestBatchPredictAdapter(t *testing.T) {
	// Create an extended model
	model := &TestExtendedModel{name: "extended"}
	
	// Wrap with batch adapter
	adapter := &framework.BatchPredictAdapter{ExtendedModel: model}
	
	// Test batch prediction
	samples := [][]float64{
		{0.2, 0.4, 0.6},
		{0.1, 0.3, 0.5},
		{0.8, 0.9, 1.0},
	}
	predictions, err := adapter.Predict(samples)
	require.NoError(t, err)
	assert.Len(t, predictions, 3)
	assert.InDelta(t, 0.4, predictions[0], 0.001)
	assert.InDelta(t, 0.3, predictions[1], 0.001)
	assert.InDelta(t, 0.9, predictions[2], 0.001)
}

func TestExtendedEnsemble(t *testing.T) {
	// Create multiple models
	models := []framework.Model{
		&TestModel{name: "model1"},
		&TestModel{name: "model2"},
	}
	
	weights := []float64{0.7, 1.3}
	
	// Create extended ensemble
	ensemble := framework.NewExtendedEnsemble(models, weights)
	
	// Test single prediction
	sample := []float64{0.6, 0.8}
	prediction, err := ensemble.PredictSingle(sample)
	require.NoError(t, err)
	assert.Greater(t, prediction, 0.0)
	assert.Less(t, prediction, 1.0)
	
	// Test batch prediction
	samples := [][]float64{
		{0.6, 0.8},
		{0.3, 0.4},
	}
	predictions, err := ensemble.Predict(samples)
	require.NoError(t, err)
	assert.Len(t, predictions, 2)
}

func TestToExtendedIdempotent(t *testing.T) {
	// Create an already extended model
	model := &TestExtendedModel{name: "already_extended"}
	
	// Convert to extended (should return same instance)
	extended := framework.ToExtended(model)
	
	// Should be the same instance
	assert.Equal(t, model, extended)
}

func BenchmarkSingleVsBatchPrediction(b *testing.B) {
	model := &TestModel{name: "bench"}
	extended := framework.ToExtended(model)
	
	samples := make([][]float64, 100)
	for i := range samples {
		samples[i] = []float64{0.5, 0.6, 0.7, 0.8}
	}
	
	b.Run("BatchPredict", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = extended.Predict(samples)
		}
	})
	
	b.Run("SinglePredict", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			for _, sample := range samples {
				_, _ = extended.PredictSingle(sample)
			}
		}
	})
}