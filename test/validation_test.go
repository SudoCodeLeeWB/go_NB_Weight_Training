package test

import (
	"math"
	"testing"
	
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/stretchr/testify/assert"
)

func TestValidateDataset(t *testing.T) {
	// Test nil dataset
	err := framework.ValidateDataset(nil)
	assert.Error(t, err)
	assert.ErrorIs(t, err, framework.ErrNoData)
	
	// Test empty dataset
	emptyDataset := data.NewDataset([]data.Sample{})
	err = framework.ValidateDataset(emptyDataset)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "dataset is empty")
	
	// Test invalid labels
	invalidLabelDataset := data.NewDataset([]data.Sample{
		{Features: []float64{1.0, 2.0}, Label: 0},
		{Features: []float64{3.0, 4.0}, Label: 2}, // Invalid label
	})
	err = framework.ValidateDataset(invalidLabelDataset)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid label")
	
	// Test NaN in features
	nanDataset := data.NewDataset([]data.Sample{
		{Features: []float64{1.0, math.NaN()}, Label: 0},
	})
	err = framework.ValidateDataset(nanDataset)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "invalid value")
	
	// Test missing class
	missingClassDataset := data.NewDataset([]data.Sample{
		{Features: []float64{1.0, 2.0}, Label: 0},
		{Features: []float64{3.0, 4.0}, Label: 0},
	})
	err = framework.ValidateDataset(missingClassDataset)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "no positive samples")
	
	// Test valid dataset
	validDataset := data.NewDataset([]data.Sample{
		{Features: []float64{1.0, 2.0}, Label: 0},
		{Features: []float64{3.0, 4.0}, Label: 1},
	})
	err = framework.ValidateDataset(validDataset)
	assert.NoError(t, err)
}

func TestValidateModels(t *testing.T) {
	// Test empty models
	err := framework.ValidateModels([]framework.Model{})
	assert.Error(t, err)
	assert.ErrorIs(t, err, framework.ErrNoModels)
	
	// Test nil model
	err = framework.ValidateModels([]framework.Model{nil})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "model 0 is nil")
	
	// Test duplicate names
	model1 := &TestModel{name: "duplicate"}
	model2 := &TestModel{name: "duplicate"}
	err = framework.ValidateModels([]framework.Model{model1, model2})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "duplicate model name")
	
	// Test empty name
	modelEmpty := &TestModel{name: ""}
	err = framework.ValidateModels([]framework.Model{modelEmpty})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty name")
	
	// Test valid models
	validModels := []framework.Model{
		&TestModel{name: "model1"},
		&TestModel{name: "model2"},
	}
	err = framework.ValidateModels(validModels)
	assert.NoError(t, err)
}

func TestValidateWeights(t *testing.T) {
	// Test mismatched count
	err := framework.ValidateWeights([]float64{0.5, 0.5}, 3)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "expected 3 weights, got 2")
	
	// Test NaN weight
	err = framework.ValidateWeights([]float64{0.5, math.NaN()}, 2)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "weight 1 is NaN")
	
	// Test negative weight
	err = framework.ValidateWeights([]float64{0.5, -0.1}, 2)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "weight 1 is negative")
	
	// Test all zeros
	err = framework.ValidateWeights([]float64{0.0, 0.0}, 2)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "all weights are zero")
	
	// Test valid weights
	err = framework.ValidateWeights([]float64{0.5, 1.5}, 2)
	assert.NoError(t, err)
}

func TestSafeDivision(t *testing.T) {
	// Test normal division
	result := framework.SafeDivision(10, 2)
	assert.Equal(t, 5.0, result)
	
	// Test division by zero
	result = framework.SafeDivision(10, 0)
	assert.Equal(t, 0.0, result)
	
	// Test zero numerator
	result = framework.SafeDivision(0, 5)
	assert.Equal(t, 0.0, result)
}

func TestClampValue(t *testing.T) {
	// Test value within bounds
	result := framework.ClampValue(0.5, 0.0, 1.0)
	assert.Equal(t, 0.5, result)
	
	// Test value below min
	result = framework.ClampValue(-0.5, 0.0, 1.0)
	assert.Equal(t, 0.0, result)
	
	// Test value above max
	result = framework.ClampValue(1.5, 0.0, 1.0)
	assert.Equal(t, 1.0, result)
}

func TestSafeModel(t *testing.T) {
	// Create a model that returns invalid predictions
	badModel := &BadModel{}
	safeModel := &framework.SafeModel{Model: badModel}
	
	// Test with empty input
	_, err := safeModel.Predict([][]float64{})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "empty input samples")
	
	// Test with NaN input
	_, err = safeModel.Predict([][]float64{{1.0, math.NaN()}})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "input validation failed")
	
	// Test with valid input but invalid output
	badModel.returnInvalid = true
	_, err = safeModel.Predict([][]float64{{1.0, 2.0}})
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "output validation failed")
	
	// Test with valid input and output
	badModel.returnInvalid = false
	predictions, err := safeModel.Predict([][]float64{{1.0, 2.0}})
	assert.NoError(t, err)
	assert.Len(t, predictions, 1)
	assert.Equal(t, 0.5, predictions[0])
}

// BadModel for testing validation
type BadModel struct {
	returnInvalid bool
}

func (bm *BadModel) Predict(samples [][]float64) ([]float64, error) {
	predictions := make([]float64, len(samples))
	for i := range predictions {
		if bm.returnInvalid {
			predictions[i] = 1.5 // Out of range
		} else {
			predictions[i] = 0.5
		}
	}
	return predictions, nil
}

func (bm *BadModel) GetName() string {
	return "BadModel"
}