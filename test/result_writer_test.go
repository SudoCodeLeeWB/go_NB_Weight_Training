package test

import (
	"os"
	"path/filepath"
	"testing"
	"time"
	
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestResultWriter(t *testing.T) {
	// Create temporary directory
	tempDir := t.TempDir()
	
	// Create result writer
	writer, err := framework.NewResultWriter(tempDir)
	require.NoError(t, err)
	
	// Check that directory was created with timestamp
	resultDir := writer.GetResultDir()
	assert.Contains(t, resultDir, "results_")
	assert.Contains(t, resultDir, time.Now().Format("2006-01-02"))
	
	// Verify directory exists
	info, err := os.Stat(resultDir)
	require.NoError(t, err)
	assert.True(t, info.IsDir())
	
	// Create mock result
	result := &framework.TrainingResult{
		BestWeights: []float64{0.8, 1.2, 0.5},
		FinalMetrics: map[string]float64{
			"pr_auc": 0.95,
			"roc_auc": 0.92,
		},
		TrainMetrics: map[string]float64{
			"pr_auc": 0.97,
		},
		ValMetrics: map[string]float64{
			"pr_auc": 0.95,
		},
		TotalEpochs: 50,
		TrainingTime: 5 * time.Minute,
		Converged: true,
	}
	
	// Create config
	config := framework.DefaultConfig()
	config.Visualization.Enabled = true
	config.Visualization.OutputDir = tempDir
	
	// Save results
	err = writer.SaveTrainingResult(result, config)
	require.NoError(t, err)
	
	// Check files were created
	expectedFiles := []string{
		"config.json",
		"training_result.json",
		"best_weights.json",
		"summary.txt",
	}
	
	for _, filename := range expectedFiles {
		path := filepath.Join(resultDir, filename)
		_, err := os.Stat(path)
		assert.NoError(t, err, "File %s should exist", filename)
	}
	
	// Read summary file
	summaryPath := filepath.Join(resultDir, "summary.txt")
	content, err := os.ReadFile(summaryPath)
	require.NoError(t, err)
	
	summary := string(content)
	assert.Contains(t, summary, "Training Summary")
	assert.Contains(t, summary, "Best Weights:")
	assert.Contains(t, summary, "Model 0: 0.8000")
	assert.Contains(t, summary, "pr_auc: 0.9500")
}

func TestMultipleResults(t *testing.T) {
	// Create temporary directory
	tempDir := t.TempDir()
	
	// Create first result
	writer1, err := framework.NewResultWriter(tempDir)
	require.NoError(t, err)
	dir1 := writer1.GetResultDir()
	
	// Wait a bit to ensure different timestamp
	time.Sleep(2 * time.Second)
	
	// Create second result
	writer2, err := framework.NewResultWriter(tempDir)
	require.NoError(t, err)
	dir2 := writer2.GetResultDir()
	
	// Directories should be different
	assert.NotEqual(t, dir1, dir2)
	
	// Both should exist
	_, err = os.Stat(dir1)
	assert.NoError(t, err)
	_, err = os.Stat(dir2)
	assert.NoError(t, err)
}