package framework

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// ModelPersistence handles saving and loading models
type ModelPersistence struct {
	ModelNames []string             `json:"model_names"`
	Weights    []float64            `json:"weights"`
	Config     *Config              `json:"config"`
	Metadata   map[string]interface{} `json:"metadata,omitempty"`
}

// SaveEnsemble saves the ensemble configuration and weights
func SaveEnsemble(ensemble *EnsembleModel, config *Config, path string) error {
	persistence := &ModelPersistence{
		ModelNames: make([]string, len(ensemble.Models)),
		Weights:    ensemble.Weights,
		Config:     config,
		Metadata: map[string]interface{}{
			"version":    "1.0",
			"created_at": time.Now().Format(time.RFC3339),
		},
	}
	
	for i, model := range ensemble.Models {
		persistence.ModelNames[i] = model.GetName()
	}
	
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()
	
	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(persistence)
}

// LoadEnsemble loads ensemble configuration
// Note: You need to provide the actual model instances
func LoadEnsemble(path string, models []Model) (*EnsembleModel, *Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()
	
	var persistence ModelPersistence
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&persistence); err != nil {
		return nil, nil, fmt.Errorf("failed to decode: %w", err)
	}
	
	// Verify model names match
	if len(models) != len(persistence.ModelNames) {
		return nil, nil, fmt.Errorf("model count mismatch: expected %d, got %d", 
			len(persistence.ModelNames), len(models))
	}
	
	for i, model := range models {
		if model.GetName() != persistence.ModelNames[i] {
			return nil, nil, fmt.Errorf("model name mismatch at index %d: expected %s, got %s",
				i, persistence.ModelNames[i], model.GetName())
		}
	}
	
	ensemble := &EnsembleModel{
		Models:  models,
		Weights: persistence.Weights,
	}
	
	return ensemble, persistence.Config, nil
}