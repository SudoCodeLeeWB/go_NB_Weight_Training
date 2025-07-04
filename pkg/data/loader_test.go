package data

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestCSVLoader(t *testing.T) {
	// Create temporary CSV file
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")
	
	csvContent := `feature1,feature2,feature3,label
0.1,0.2,0.3,0
0.4,0.5,0.6,1
0.7,0.8,0.9,1
0.2,0.3,0.4,0`
	
	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	require.NoError(t, err)
	
	// Test loading
	loader := NewCSVLoader()
	dataset, err := loader.Load(csvPath)
	require.NoError(t, err)
	
	// Verify dataset
	assert.Equal(t, 4, dataset.NumSamples)
	assert.Equal(t, 3, dataset.NumFeatures)
	assert.Equal(t, 2, dataset.ClassCounts[0.0])
	assert.Equal(t, 2, dataset.ClassCounts[1.0])
	
	// Verify first sample
	assert.Equal(t, []float64{0.1, 0.2, 0.3}, dataset.Samples[0].Features)
	assert.Equal(t, 0.0, dataset.Samples[0].Label)
	
	// Verify last sample
	assert.Equal(t, []float64{0.2, 0.3, 0.4}, dataset.Samples[3].Features)
	assert.Equal(t, 0.0, dataset.Samples[3].Label)
}

func TestCSVLoaderNoHeader(t *testing.T) {
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")
	
	csvContent := `0.1,0.2,0.3,0
0.4,0.5,0.6,1`
	
	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	require.NoError(t, err)
	
	loader := &CSVLoader{
		HasHeader:   false,
		LabelColumn: -1,
		Delimiter:   ',',
	}
	
	dataset, err := loader.Load(csvPath)
	require.NoError(t, err)
	
	assert.Equal(t, 2, dataset.NumSamples)
	assert.Equal(t, 3, dataset.NumFeatures)
}

func TestCSVLoaderCustomDelimiter(t *testing.T) {
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")
	
	csvContent := `feature1;feature2;label
0.1;0.2;0
0.4;0.5;1`
	
	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	require.NoError(t, err)
	
	loader := &CSVLoader{
		HasHeader:   true,
		LabelColumn: -1,
		Delimiter:   ';',
	}
	
	dataset, err := loader.Load(csvPath)
	require.NoError(t, err)
	
	assert.Equal(t, 2, dataset.NumSamples)
	assert.Equal(t, 2, dataset.NumFeatures)
}

func TestCSVLoaderLabelInMiddle(t *testing.T) {
	tmpDir := t.TempDir()
	csvPath := filepath.Join(tmpDir, "test.csv")
	
	csvContent := `feature1,label,feature2
0.1,0,0.2
0.4,1,0.5`
	
	err := os.WriteFile(csvPath, []byte(csvContent), 0644)
	require.NoError(t, err)
	
	loader := &CSVLoader{
		HasHeader:   true,
		LabelColumn: 1,
		Delimiter:   ',',
	}
	
	dataset, err := loader.Load(csvPath)
	require.NoError(t, err)
	
	assert.Equal(t, 2, dataset.NumSamples)
	assert.Equal(t, 2, dataset.NumFeatures)
	assert.Equal(t, []float64{0.1, 0.2}, dataset.Samples[0].Features)
}

func TestCSVLoaderErrors(t *testing.T) {
	loader := NewCSVLoader()
	
	// Test non-existent file
	_, err := loader.Load("/non/existent/file.csv")
	assert.Error(t, err)
	
	// Test empty CSV
	tmpDir := t.TempDir()
	emptyPath := filepath.Join(tmpDir, "empty.csv")
	err = os.WriteFile(emptyPath, []byte(""), 0644)
	require.NoError(t, err)
	
	_, err = loader.Load(emptyPath)
	assert.Error(t, err)
	
	// Test invalid float
	invalidPath := filepath.Join(tmpDir, "invalid.csv")
	err = os.WriteFile(invalidPath, []byte("a,b,c\n1,2,not_a_number"), 0644)
	require.NoError(t, err)
	
	_, err = loader.Load(invalidPath)
	assert.Error(t, err)
	
	// Test non-binary label
	nonBinaryPath := filepath.Join(tmpDir, "nonbinary.csv")
	err = os.WriteFile(nonBinaryPath, []byte("f1,label\n0.5,2"), 0644)
	require.NoError(t, err)
	
	_, err = loader.Load(nonBinaryPath)
	assert.Error(t, err)
}

func TestJSONLoader(t *testing.T) {
	tmpDir := t.TempDir()
	jsonPath := filepath.Join(tmpDir, "test.json")
	
	jsonContent := `{
		"samples": [
			{"features": [0.1, 0.2, 0.3], "label": 0, "id": "sample1"},
			{"features": [0.4, 0.5, 0.6], "label": 1, "id": "sample2"},
			{"features": [0.7, 0.8, 0.9], "label": 1},
			{"features": [0.2, 0.3, 0.4], "label": 0}
		]
	}`
	
	err := os.WriteFile(jsonPath, []byte(jsonContent), 0644)
	require.NoError(t, err)
	
	loader := NewJSONLoader()
	dataset, err := loader.Load(jsonPath)
	require.NoError(t, err)
	
	assert.Equal(t, 4, dataset.NumSamples)
	assert.Equal(t, 3, dataset.NumFeatures)
	assert.Equal(t, "sample1", dataset.Samples[0].ID)
	assert.Equal(t, "sample2", dataset.Samples[1].ID)
	assert.Equal(t, "sample_2", dataset.Samples[2].ID) // Auto-generated ID
}

func TestJSONLoaderErrors(t *testing.T) {
	loader := NewJSONLoader()
	
	// Test invalid JSON
	tmpDir := t.TempDir()
	invalidPath := filepath.Join(tmpDir, "invalid.json")
	err := os.WriteFile(invalidPath, []byte("{invalid json}"), 0644)
	require.NoError(t, err)
	
	_, err = loader.Load(invalidPath)
	assert.Error(t, err)
	
	// Test non-binary label
	nonBinaryPath := filepath.Join(tmpDir, "nonbinary.json")
	jsonContent := `{
		"samples": [
			{"features": [0.1, 0.2], "label": 2}
		]
	}`
	err = os.WriteFile(nonBinaryPath, []byte(jsonContent), 0644)
	require.NoError(t, err)
	
	_, err = loader.Load(nonBinaryPath)
	assert.Error(t, err)
}

func TestLoadData(t *testing.T) {
	tmpDir := t.TempDir()
	
	// Test CSV loading
	csvPath := filepath.Join(tmpDir, "test.csv")
	err := os.WriteFile(csvPath, []byte("f1,label\n0.5,1"), 0644)
	require.NoError(t, err)
	
	dataset, err := LoadData(csvPath)
	require.NoError(t, err)
	assert.Equal(t, 1, dataset.NumSamples)
	
	// Test JSON loading
	jsonPath := filepath.Join(tmpDir, "test.json")
	jsonContent := `{"samples": [{"features": [0.5], "label": 1}]}`
	err = os.WriteFile(jsonPath, []byte(jsonContent), 0644)
	require.NoError(t, err)
	
	dataset, err = LoadData(jsonPath)
	require.NoError(t, err)
	assert.Equal(t, 1, dataset.NumSamples)
	
	// Test unsupported format
	txtPath := filepath.Join(tmpDir, "test.txt")
	err = os.WriteFile(txtPath, []byte("some text"), 0644)
	require.NoError(t, err)
	
	_, err = LoadData(txtPath)
	assert.Error(t, err)
}

func TestSaveDataset(t *testing.T) {
	// Create a dataset
	samples := []Sample{
		{Features: []float64{0.1, 0.2}, Label: 0, ID: "s1"},
		{Features: []float64{0.3, 0.4}, Label: 1, ID: "s2"},
	}
	dataset := NewDataset(samples)
	
	// Save dataset
	tmpDir := t.TempDir()
	savePath := filepath.Join(tmpDir, "saved.csv")
	
	err := SaveDataset(dataset, savePath)
	require.NoError(t, err)
	
	// Load it back
	loader := NewCSVLoader()
	loaded, err := loader.Load(savePath)
	require.NoError(t, err)
	
	assert.Equal(t, dataset.NumSamples, loaded.NumSamples)
	assert.Equal(t, dataset.NumFeatures, loaded.NumFeatures)
	
	// Check data integrity
	for i := range dataset.Samples {
		assert.Equal(t, dataset.Samples[i].Features, loaded.Samples[i].Features)
		assert.Equal(t, dataset.Samples[i].Label, loaded.Samples[i].Label)
	}
}

func TestDatasetMethods(t *testing.T) {
	samples := []Sample{
		{Features: []float64{0.1, 0.2}, Label: 0},
		{Features: []float64{0.3, 0.4}, Label: 1},
		{Features: []float64{0.5, 0.6}, Label: 1},
		{Features: []float64{0.7, 0.8}, Label: 0},
	}
	dataset := NewDataset(samples)
	
	// Test GetFeatures
	features := dataset.GetFeatures()
	assert.Equal(t, 4, len(features))
	assert.Equal(t, []float64{0.1, 0.2}, features[0])
	
	// Test GetLabels
	labels := dataset.GetLabels()
	assert.Equal(t, []float64{0, 1, 1, 0}, labels)
	
	// Test Subset
	subset := dataset.Subset([]int{0, 2})
	assert.Equal(t, 2, subset.NumSamples)
	assert.Equal(t, 0.0, subset.Samples[0].Label)
	assert.Equal(t, 1.0, subset.Samples[1].Label)
	
	// Test ClassBalance
	balance := dataset.ClassBalance()
	assert.Equal(t, 0.5, balance) // 2 positive out of 4
}

func TestEmptyDataset(t *testing.T) {
	dataset := NewDataset([]Sample{})
	
	assert.Equal(t, 0, dataset.NumSamples)
	assert.Equal(t, 0, dataset.NumFeatures)
	assert.Equal(t, 0, len(dataset.ClassCounts))
	assert.Equal(t, 0.0, dataset.ClassBalance())
}

func BenchmarkCSVLoader(b *testing.B) {
	// Create a larger CSV file
	tmpDir := b.TempDir()
	csvPath := filepath.Join(tmpDir, "bench.csv")
	
	// Generate CSV content
	content := "f1,f2,f3,f4,f5,label\n"
	for i := 0; i < 1000; i++ {
		content += "0.1,0.2,0.3,0.4,0.5,1\n"
	}
	
	err := os.WriteFile(csvPath, []byte(content), 0644)
	require.NoError(b, err)
	
	loader := NewCSVLoader()
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = loader.Load(csvPath)
	}
}

func TestGetFileExtension(t *testing.T) {
	tests := []struct {
		path     string
		expected string
	}{
		{"file.csv", ".csv"},
		{"file.CSV", ".CSV"},
		{"path/to/file.json", ".json"},
		{"file.tar.gz", ".gz"},
		{"noextension", ""},
		{"", ""},
	}
	
	for _, tt := range tests {
		result := getFileExtension(tt.path)
		assert.Equal(t, tt.expected, result)
	}
}