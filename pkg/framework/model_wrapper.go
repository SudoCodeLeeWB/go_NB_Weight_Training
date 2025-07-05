package framework

import (
	"fmt"
	"os"
	"plugin"
	"path/filepath"
)

// ModelWrapper is the interface that each model directory must implement
// This allows dynamic loading of models without recompiling the main framework
type ModelWrapper interface {
	// LoadModel initializes the model from the directory
	LoadModel(modelDir string) error
	
	// GetAggregatedModel returns the AggregatedModel implementation
	GetAggregatedModel() AggregatedModel
	
	// GetInfo returns model information
	GetInfo() ModelInfo
}

// ModelInfo contains metadata about the model
type ModelInfo struct {
	Name        string   `json:"name"`
	Version     string   `json:"version"`
	Description string   `json:"description"`
	Models      []string `json:"models"` // Names of internal models
}

// BaseModelWrapper provides common functionality for model wrappers
type BaseModelWrapper struct {
	modelDir string
	info     ModelInfo
}

// LoadModelWrapper loads a model wrapper from a directory
// The directory should contain a compiled plugin or a wrapper.go file
func LoadModelWrapper(modelDir string) (ModelWrapper, error) {
	// Option 1: Try to load as a plugin (for compiled models)
	pluginPath := filepath.Join(modelDir, "model.so")
	if fileExists(pluginPath) {
		return loadPlugin(pluginPath)
	}
	
	// Option 2: For development, we'll use a registry pattern
	// Models register themselves when imported
	wrapperName := filepath.Base(modelDir)
	if wrapper, exists := modelRegistry[wrapperName]; exists {
		wrapper.LoadModel(modelDir)
		return wrapper, nil
	}
	
	return nil, fmt.Errorf("no model wrapper found in %s", modelDir)
}

// Model registry for development mode
var modelRegistry = make(map[string]ModelWrapper)

// RegisterModelWrapper registers a model wrapper
func RegisterModelWrapper(name string, wrapper ModelWrapper) {
	modelRegistry[name] = wrapper
}

// Helper to check if file exists
func fileExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// Load plugin (for production use)
func loadPlugin(path string) (ModelWrapper, error) {
	p, err := plugin.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open plugin: %w", err)
	}
	
	symbol, err := p.Lookup("GetModelWrapper")
	if err != nil {
		return nil, fmt.Errorf("plugin missing GetModelWrapper function: %w", err)
	}
	
	getWrapper, ok := symbol.(func() ModelWrapper)
	if !ok {
		return nil, fmt.Errorf("GetModelWrapper has wrong signature")
	}
	
	return getWrapper(), nil
}