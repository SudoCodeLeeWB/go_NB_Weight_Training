package data

import (
	"encoding/csv"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Loader interface for different data formats
type Loader interface {
	Load(path string) (*Dataset, error)
}

// CSVLoader loads data from CSV files
type CSVLoader struct {
	HasHeader   bool
	LabelColumn int // Index of label column (-1 for last column)
	Delimiter   rune
}

// NewCSVLoader creates a new CSV loader with default settings
func NewCSVLoader() *CSVLoader {
	return &CSVLoader{
		HasHeader:   true,
		LabelColumn: -1, // Last column by default
		Delimiter:   ',',
	}
}

// Load implements the Loader interface for CSV
func (cl *CSVLoader) Load(path string) (*Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()
	
	reader := csv.NewReader(file)
	reader.Comma = cl.Delimiter
	reader.TrimLeadingSpace = true
	
	// Read all records
	records, err := reader.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("failed to read CSV: %w", err)
	}
	
	if len(records) == 0 {
		return nil, fmt.Errorf("empty CSV file")
	}
	
	// Skip header if present
	startIdx := 0
	if cl.HasHeader {
		startIdx = 1
	}
	
	samples := make([]Sample, 0, len(records)-startIdx)
	
	for i := startIdx; i < len(records); i++ {
		record := records[i]
		if len(record) < 2 {
			continue // Skip invalid rows
		}
		
		// Determine label column
		labelCol := cl.LabelColumn
		if labelCol < 0 {
			labelCol = len(record) - 1
		}
		
		// Parse features
		features := make([]float64, 0, len(record)-1)
		for j, val := range record {
			if j == labelCol {
				continue // Skip label column
			}
			
			f, err := strconv.ParseFloat(strings.TrimSpace(val), 64)
			if err != nil {
				return nil, fmt.Errorf("failed to parse feature at row %d, col %d: %w", i, j, err)
			}
			features = append(features, f)
		}
		
		// Parse label
		label, err := strconv.ParseFloat(strings.TrimSpace(record[labelCol]), 64)
		if err != nil {
			return nil, fmt.Errorf("failed to parse label at row %d: %w", i, err)
		}
		
		// Ensure label is binary
		if label != 0.0 && label != 1.0 {
			return nil, fmt.Errorf("non-binary label %.2f at row %d", label, i)
		}
		
		samples = append(samples, Sample{
			Features: features,
			Label:    label,
			ID:       fmt.Sprintf("sample_%d", i-startIdx),
		})
	}
	
	return NewDataset(samples), nil
}

// JSONLoader loads data from JSON files
type JSONLoader struct{}

// JSONData represents the expected JSON structure
type JSONData struct {
	Samples []struct {
		Features []float64 `json:"features"`
		Label    float64   `json:"label"`
		ID       string    `json:"id,omitempty"`
	} `json:"samples"`
}

// NewJSONLoader creates a new JSON loader
func NewJSONLoader() *JSONLoader {
	return &JSONLoader{}
}

// Load implements the Loader interface for JSON
func (jl *JSONLoader) Load(path string) (*Dataset, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()
	
	var data JSONData
	decoder := json.NewDecoder(file)
	if err := decoder.Decode(&data); err != nil {
		return nil, fmt.Errorf("failed to parse JSON: %w", err)
	}
	
	samples := make([]Sample, len(data.Samples))
	for i, s := range data.Samples {
		// Validate label
		if s.Label != 0.0 && s.Label != 1.0 {
			return nil, fmt.Errorf("non-binary label %.2f for sample %s", s.Label, s.ID)
		}
		
		samples[i] = Sample{
			Features: s.Features,
			Label:    s.Label,
			ID:       s.ID,
		}
		
		if samples[i].ID == "" {
			samples[i].ID = fmt.Sprintf("sample_%d", i)
		}
	}
	
	return NewDataset(samples), nil
}

// LoadData is a convenience function that loads data based on file extension
func LoadData(path string) (*Dataset, error) {
	ext := strings.ToLower(strings.TrimPrefix(getFileExtension(path), "."))
	
	switch ext {
	case "csv":
		loader := NewCSVLoader()
		return loader.Load(path)
	case "json":
		loader := NewJSONLoader()
		return loader.Load(path)
	default:
		return nil, fmt.Errorf("unsupported file format: %s", ext)
	}
}

// getFileExtension returns the file extension
func getFileExtension(path string) string {
	parts := strings.Split(path, ".")
	if len(parts) > 1 {
		return "." + parts[len(parts)-1]
	}
	return ""
}

// SaveDataset saves a dataset to CSV format
func SaveDataset(dataset *Dataset, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer file.Close()
	
	writer := csv.NewWriter(file)
	defer writer.Flush()
	
	// Write header
	header := make([]string, dataset.NumFeatures+1)
	for i := 0; i < dataset.NumFeatures; i++ {
		header[i] = fmt.Sprintf("feature_%d", i)
	}
	header[dataset.NumFeatures] = "label"
	
	if err := writer.Write(header); err != nil {
		return fmt.Errorf("failed to write header: %w", err)
	}
	
	// Write samples
	for _, sample := range dataset.Samples {
		record := make([]string, dataset.NumFeatures+1)
		for i, f := range sample.Features {
			record[i] = strconv.FormatFloat(f, 'f', -1, 64)
		}
		record[dataset.NumFeatures] = strconv.FormatFloat(sample.Label, 'f', -1, 64)
		
		if err := writer.Write(record); err != nil {
			return fmt.Errorf("failed to write record: %w", err)
		}
	}
	
	return nil
}