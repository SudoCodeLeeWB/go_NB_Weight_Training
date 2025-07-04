package framework

import (
	"encoding/json"
	"fmt"
	"os"
	"time"
)

// WeightsSave represents the structure for saving weights
type WeightsSave struct {
	Weights   []float64          `json:"weights"`
	Metadata  map[string]float64 `json:"metadata,omitempty"`
	Timestamp string             `json:"timestamp"`
}

// SaveWeights saves the trained weights to a JSON file
func SaveWeights(weights []float64, path string) error {
	save := WeightsSave{
		Weights:   weights,
		Timestamp: time.Now().Format(time.RFC3339),
	}
	
	data, err := json.MarshalIndent(save, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal weights: %w", err)
	}
	
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write weights: %w", err)
	}
	
	return nil
}

// LoadWeights loads weights from a JSON file
func LoadWeights(path string) ([]float64, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read weights file: %w", err)
	}
	
	var save WeightsSave
	if err := json.Unmarshal(data, &save); err != nil {
		return nil, fmt.Errorf("failed to unmarshal weights: %w", err)
	}
	
	return save.Weights, nil
}