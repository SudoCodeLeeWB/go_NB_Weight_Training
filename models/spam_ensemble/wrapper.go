package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
)

func init() {
	// Register this wrapper when the package is imported
	framework.RegisterModelWrapper("spam_ensemble", &SpamEnsembleWrapper{})
}

// SpamEnsembleWrapper implements the ModelWrapper interface
type SpamEnsembleWrapper struct {
	modelDir  string
	ensemble  *SpamAggregatedModel
	config    ModelConfig
}

// ModelConfig for this specific model
type ModelConfig struct {
	ModelPaths struct {
		BayesModel  string `json:"bayes_model"`
		NeuralNet   string `json:"neural_net"`
		RulesBased  string `json:"rules_based"`
	} `json:"model_paths"`
	InitialWeights []float64 `json:"initial_weights"`
}

// LoadModel loads the model configuration and initializes the ensemble
func (w *SpamEnsembleWrapper) LoadModel(modelDir string) error {
	w.modelDir = modelDir
	
	// Load model config
	configPath := filepath.Join(modelDir, "model_config.json")
	configData, err := os.ReadFile(configPath)
	if err != nil {
		return fmt.Errorf("failed to read model config: %w", err)
	}
	
	if err := json.Unmarshal(configData, &w.config); err != nil {
		return fmt.Errorf("failed to parse model config: %w", err)
	}
	
	// Initialize the aggregated model
	w.ensemble = &SpamAggregatedModel{
		modelDir: modelDir,
		config:   w.config,
		weights:  w.config.InitialWeights,
	}
	
	// Load individual models
	if err := w.ensemble.LoadModels(); err != nil {
		return fmt.Errorf("failed to load models: %w", err)
	}
	
	return nil
}

// GetAggregatedModel returns the AggregatedModel implementation
func (w *SpamEnsembleWrapper) GetAggregatedModel() framework.AggregatedModel {
	return w.ensemble
}

// GetInfo returns model information
func (w *SpamEnsembleWrapper) GetInfo() framework.ModelInfo {
	return framework.ModelInfo{
		Name:        "Spam Detection Ensemble",
		Version:     "1.0.0",
		Description: "Combines Bayes, Neural Net, and Rules-based spam classifiers",
		Models:      []string{"BayesFilter", "NeuralNetwork", "RulesEngine"},
	}
}