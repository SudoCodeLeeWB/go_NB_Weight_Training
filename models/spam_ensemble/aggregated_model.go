package main

import (
	"fmt"
	"math"
	"path/filepath"
	"sort"
)

// SpamAggregatedModel implements both AggregatedModel and CalibratedAggregatedModel interfaces
type SpamAggregatedModel struct {
	modelDir string
	config   ModelConfig
	
	// Internal models (these could be anything - sklearn, tensorflow, custom)
	bayesModel  interface{} // Your actual Bayes model
	neuralNet   interface{} // Your actual Neural Network
	rulesEngine interface{} // Your actual Rules Engine
	
	// Weights to be optimized
	weights []float64
	
	// Calibration parameters (simple beta calibration)
	calibrationFitted bool
	posMean          float64
	negMean          float64
}

// LoadModels loads the actual ML models
func (m *SpamAggregatedModel) LoadModels() error {
	// In real implementation, you would load your actual models here
	// For example:
	// m.bayesModel = pickle.Load(filepath.Join(m.modelDir, m.config.ModelPaths.BayesModel))
	// m.neuralNet = keras.LoadModel(filepath.Join(m.modelDir, m.config.ModelPaths.NeuralNet))
	// m.rulesEngine = json.Load(filepath.Join(m.modelDir, m.config.ModelPaths.RulesBased))
	
	// For now, we'll use mock models
	m.bayesModel = &MockBayesModel{modelPath: filepath.Join(m.modelDir, m.config.ModelPaths.BayesModel)}
	m.neuralNet = &MockNeuralNet{modelPath: filepath.Join(m.modelDir, m.config.ModelPaths.NeuralNet)}
	m.rulesEngine = &MockRulesEngine{modelPath: filepath.Join(m.modelDir, m.config.ModelPaths.RulesBased)}
	
	return nil
}

// Predict combines predictions from all models using weighted Naive Bayes
func (m *SpamAggregatedModel) Predict(samples [][]float64) ([]float64, error) {
	n := len(samples)
	results := make([]float64, n)
	
	// Get predictions from each model
	// In real implementation, these would call your actual models
	bayesPreds := m.getBayesPredictions(samples)
	nnPreds := m.getNeuralNetPredictions(samples)
	rulesPreds := m.getRulesPredictions(samples)
	
	// Combine using weighted Naive Bayes multiplication
	for i := 0; i < n; i++ {
		results[i] = 1.0
		
		// Apply each model's prediction with its weight
		results[i] *= math.Pow(bayesPreds[i], m.weights[0])
		results[i] *= math.Pow(nnPreds[i], m.weights[1])
		results[i] *= math.Pow(rulesPreds[i], m.weights[2])
		
		// Ensure in [0,1] range
		if results[i] > 1.0 {
			results[i] = 1.0
		} else if results[i] < 0.0 {
			results[i] = 0.0
		}
	}
	
	return results, nil
}

// GetWeights returns current weights
func (m *SpamAggregatedModel) GetWeights() []float64 {
	weightsCopy := make([]float64, len(m.weights))
	copy(weightsCopy, m.weights)
	return weightsCopy
}

// SetWeights updates weights
func (m *SpamAggregatedModel) SetWeights(weights []float64) error {
	if len(weights) != 3 {
		return fmt.Errorf("expected 3 weights, got %d", len(weights))
	}
	copy(m.weights, weights)
	return nil
}

// GetNumModels returns number of models
func (m *SpamAggregatedModel) GetNumModels() int {
	return 3
}

// GetModelNames returns model names
func (m *SpamAggregatedModel) GetModelNames() []string {
	return []string{"BayesFilter", "NeuralNetwork", "RulesEngine"}
}

// Helper methods to get predictions from each model
func (m *SpamAggregatedModel) getBayesPredictions(samples [][]float64) []float64 {
	// In real implementation: return m.bayesModel.PredictProba(samples)
	model := m.bayesModel.(*MockBayesModel)
	return model.Predict(samples)
}

func (m *SpamAggregatedModel) getNeuralNetPredictions(samples [][]float64) []float64 {
	// In real implementation: return m.neuralNet.Predict(samples)
	model := m.neuralNet.(*MockNeuralNet)
	return model.Predict(samples)
}

func (m *SpamAggregatedModel) getRulesPredictions(samples [][]float64) []float64 {
	// In real implementation: return m.rulesEngine.Score(samples)
	model := m.rulesEngine.(*MockRulesEngine)
	return model.Predict(samples)
}

// Mock models for demonstration
type MockBayesModel struct {
	modelPath string
}

func (m *MockBayesModel) Predict(samples [][]float64) []float64 {
	// Mock implementation - returns percentage values
	preds := make([]float64, len(samples))
	for i := range samples {
		// Simple mock logic based on first feature
		if len(samples[i]) > 0 {
			preds[i] = math.Min(1.0, math.Max(0.0, samples[i][0]*0.8+0.1))
		}
	}
	return preds
}

type MockNeuralNet struct {
	modelPath string
}

func (m *MockNeuralNet) Predict(samples [][]float64) []float64 {
	// Mock implementation
	preds := make([]float64, len(samples))
	for i := range samples {
		// Different logic for variety
		avg := 0.0
		for _, f := range samples[i] {
			avg += f
		}
		avg /= float64(len(samples[i]))
		preds[i] = 1.0 / (1.0 + math.Exp(-2*(avg-0.5))) // Sigmoid
	}
	return preds
}

type MockRulesEngine struct {
	modelPath string
}

func (m *MockRulesEngine) Predict(samples [][]float64) []float64 {
	// Mock implementation
	preds := make([]float64, len(samples))
	for i := range samples {
		// Rule-based logic
		if len(samples[i]) > 1 && samples[i][0] > 0.7 && samples[i][1] > 0.6 {
			preds[i] = 0.9
		} else if len(samples[i]) > 0 && samples[i][0] < 0.3 {
			preds[i] = 0.1
		} else {
			preds[i] = 0.5
		}
	}
	return preds
}

// PredictWithCalibration returns both raw and calibrated scores
// This implements the CalibratedAggregatedModel interface
func (m *SpamAggregatedModel) PredictWithCalibration(samples [][]float64) (raw []float64, calibrated []float64, err error) {
	// Get raw predictions using the standard Predict method
	raw, err = m.Predict(samples)
	if err != nil {
		return nil, nil, err
	}
	
	// Apply calibration if fitted
	calibrated = make([]float64, len(raw))
	if m.calibrationFitted {
		for i, score := range raw {
			calibrated[i] = m.calibrateScore(score)
		}
	} else {
		// If not calibrated, just use simple min-max normalization
		min, max := m.findMinMax(raw)
		for i, score := range raw {
			if max > min {
				calibrated[i] = (score - min) / (max - min)
			} else {
				calibrated[i] = 0.5
			}
		}
	}
	
	return raw, calibrated, nil
}

// GetCalibrationMethod returns the calibration method name
func (m *SpamAggregatedModel) GetCalibrationMethod() string {
	if m.calibrationFitted {
		return "Beta Calibration"
	}
	return "Min-Max Normalization"
}

// FitCalibration fits the calibration parameters on validation data
// This is optional - the model can work without being calibrated
func (m *SpamAggregatedModel) FitCalibration(samples [][]float64, labels []float64) error {
	// Get raw predictions
	preds, err := m.Predict(samples)
	if err != nil {
		return err
	}
	
	// Calculate positive and negative score means
	var posSum, negSum float64
	var posCount, negCount int
	
	for i, pred := range preds {
		if labels[i] > 0.5 {
			posSum += pred
			posCount++
		} else {
			negSum += pred
			negCount++
		}
	}
	
	if posCount > 0 {
		m.posMean = posSum / float64(posCount)
	} else {
		m.posMean = 0.8
	}
	
	if negCount > 0 {
		m.negMean = negSum / float64(negCount)
	} else {
		m.negMean = 0.2
	}
	
	m.calibrationFitted = true
	return nil
}

// calibrateScore applies beta calibration to a single score
func (m *SpamAggregatedModel) calibrateScore(score float64) float64 {
	if !m.calibrationFitted {
		return score
	}
	
	// Beta calibration: map negative mean to 0.2 and positive mean to 0.8
	if score <= m.negMean {
		return score / m.negMean * 0.2
	} else if score >= m.posMean {
		return 0.8 + (score-m.posMean)/(1-m.posMean) * 0.2
	} else {
		// Linear interpolation between negMean and posMean
		t := (score - m.negMean) / (m.posMean - m.negMean)
		return 0.2 + t * 0.6
	}
}

// findMinMax finds the min and max values in a slice
func (m *SpamAggregatedModel) findMinMax(values []float64) (min, max float64) {
	if len(values) == 0 {
		return 0, 1
	}
	
	min, max = values[0], values[0]
	for _, v := range values[1:] {
		if v < min {
			min = v
		}
		if v > max {
			max = v
		}
	}
	return min, max
}