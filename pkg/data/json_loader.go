package data

import (
	"encoding/json"
	"fmt"
	"os"
)

// JSONDataset represents the structure of a JSON dataset file
type JSONDataset struct {
	Features [][]float64 `json:"features"`
	Labels   []int       `json:"labels"`
	Metadata *JSONMetadata `json:"metadata,omitempty"`
}

// JSONMetadata contains optional metadata about the dataset
type JSONMetadata struct {
	FeatureNames []string `json:"feature_names,omitempty"`
	Description  string   `json:"description,omitempty"`
	NumSamples   int      `json:"num_samples,omitempty"`
	NumFeatures  int      `json:"num_features,omitempty"`
}

// LoadJSONData loads a dataset from a JSON file
func LoadJSONData(filepath string) (*Dataset, error) {
	// Read file
	data, err := os.ReadFile(filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to read JSON file: %w", err)
	}
	
	// Parse JSON
	var jsonData JSONDataset
	if err := json.Unmarshal(data, &jsonData); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}
	
	// Validate
	if len(jsonData.Features) == 0 {
		return nil, fmt.Errorf("no features found in JSON dataset")
	}
	
	if len(jsonData.Features) != len(jsonData.Labels) {
		return nil, fmt.Errorf("features and labels length mismatch: %d vs %d", 
			len(jsonData.Features), len(jsonData.Labels))
	}
	
	// Convert to samples
	samples := make([]Sample, len(jsonData.Features))
	for i := range jsonData.Features {
		samples[i] = Sample{
			Features: jsonData.Features[i],
			Label:    jsonData.Labels[i],
		}
	}
	
	return NewDataset(samples), nil
}

// SaveJSONData saves a dataset to a JSON file
func SaveJSONData(dataset *Dataset, filepath string) error {
	// Extract features and labels
	features := make([][]float64, dataset.NumSamples)
	labels := make([]int, dataset.NumSamples)
	
	for i, sample := range dataset.Samples {
		features[i] = sample.Features
		labels[i] = sample.Label
	}
	
	// Create JSON structure
	jsonData := JSONDataset{
		Features: features,
		Labels:   labels,
		Metadata: &JSONMetadata{
			NumSamples:  dataset.NumSamples,
			NumFeatures: dataset.NumFeatures,
		},
	}
	
	// Marshal to JSON
	data, err := json.MarshalIndent(jsonData, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal JSON: %w", err)
	}
	
	// Write to file
	if err := os.WriteFile(filepath, data, 0644); err != nil {
		return fmt.Errorf("failed to write JSON file: %w", err)
	}
	
	return nil
}

// LoadData is updated to support both CSV and JSON
func LoadData(filepath string) (*Dataset, error) {
	// Check file extension
	if len(filepath) > 5 && filepath[len(filepath)-5:] == ".json" {
		return LoadJSONData(filepath)
	}
	// Default to CSV
	loader := NewCSVLoader()
	return loader.Load(filepath)
}