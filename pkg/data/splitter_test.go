package data

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func createTestDataset(n int, posRatio float64) *Dataset {
	samples := make([]Sample, n)
	numPos := int(float64(n) * posRatio)
	
	for i := 0; i < n; i++ {
		label := 0.0
		if i < numPos {
			label = 1.0
		}
		samples[i] = Sample{
			Features: []float64{float64(i), float64(i) * 2},
			Label:    label,
			ID:       fmt.Sprintf("sample_%d", i),
		}
	}
	
	return NewDataset(samples)
}

func TestRandomSplitter(t *testing.T) {
	dataset := createTestDataset(100, 0.5)
	
	tests := []struct {
		name     string
		testSize float64
		seed     int64
	}{
		{"20% test split", 0.2, 42},
		{"30% test split", 0.3, 42},
		{"50% test split", 0.5, 42},
	}
	
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			splitter := NewRandomSplitter(tt.testSize, tt.seed)
			split, err := splitter.Split(dataset)
			require.NoError(t, err)
			
			// Check sizes
			expectedTestSize := int(float64(dataset.NumSamples) * tt.testSize)
			assert.Equal(t, expectedTestSize, split.Test.NumSamples)
			assert.Equal(t, dataset.NumSamples-expectedTestSize, split.Train.NumSamples)
			
			// Check no overlap
			testIDs := make(map[string]bool)
			for _, s := range split.Test.Samples {
				testIDs[s.ID] = true
			}
			
			for _, s := range split.Train.Samples {
				assert.False(t, testIDs[s.ID], "Sample %s appears in both train and test", s.ID)
			}
			
			// Check reproducibility
			splitter2 := NewRandomSplitter(tt.testSize, tt.seed)
			split2, err := splitter2.Split(dataset)
			require.NoError(t, err)
			
			// Should produce same split with same seed
			for i := range split.Test.Samples {
				assert.Equal(t, split.Test.Samples[i].ID, split2.Test.Samples[i].ID)
			}
		})
	}
}

func TestRandomSplitterErrors(t *testing.T) {
	dataset := createTestDataset(10, 0.5)
	
	// Test invalid test size
	splitter := NewRandomSplitter(0, 42)
	_, err := splitter.Split(dataset)
	assert.Error(t, err)
	
	splitter = NewRandomSplitter(1, 42)
	_, err = splitter.Split(dataset)
	assert.Error(t, err)
	
	splitter = NewRandomSplitter(1.5, 42)
	_, err = splitter.Split(dataset)
	assert.Error(t, err)
}

func TestStratifiedSplitter(t *testing.T) {
	// Create imbalanced dataset
	dataset := createTestDataset(100, 0.3) // 30% positive
	
	splitter := NewStratifiedSplitter(0.2, 42)
	split, err := splitter.Split(dataset)
	require.NoError(t, err)
	
	// Check that class distribution is preserved
	trainBalance := split.Train.ClassBalance()
	testBalance := split.Test.ClassBalance()
	originalBalance := dataset.ClassBalance()
	
	// Allow small tolerance due to rounding
	assert.InDelta(t, originalBalance, trainBalance, 0.1)
	assert.InDelta(t, originalBalance, testBalance, 0.1)
	
	// Check sizes
	assert.Equal(t, 20, split.Test.NumSamples)
	assert.Equal(t, 80, split.Train.NumSamples)
}

func TestStratifiedSplitterEdgeCases(t *testing.T) {
	// Test with very small dataset
	samples := []Sample{
		{Features: []float64{1}, Label: 0},
		{Features: []float64{2}, Label: 0},
		{Features: []float64{3}, Label: 1},
		{Features: []float64{4}, Label: 1},
	}
	dataset := NewDataset(samples)
	
	splitter := NewStratifiedSplitter(0.5, 42)
	split, err := splitter.Split(dataset)
	require.NoError(t, err)
	
	// Each class should have one sample in test
	assert.Equal(t, 1, split.Test.ClassCounts[0.0])
	assert.Equal(t, 1, split.Test.ClassCounts[1.0])
}

func TestKFoldCV(t *testing.T) {
	dataset := createTestDataset(100, 0.5)
	
	tests := []struct {
		k       int
		shuffle bool
	}{
		{5, false},
		{5, true},
		{10, true},
		{2, true},
	}
	
	for _, tt := range tests {
		t.Run(fmt.Sprintf("k=%d,shuffle=%v", tt.k, tt.shuffle), func(t *testing.T) {
			cv := NewKFoldCV(tt.k, tt.shuffle, 42)
			folds, err := cv.GetFolds(dataset)
			require.NoError(t, err)
			
			assert.Len(t, folds, tt.k)
			
			// Check that each sample appears exactly once in test across all folds
			testCounts := make(map[int]int)
			trainCounts := make(map[int]int)
			
			for _, fold := range folds {
				// Check fold sizes
				expectedTestSize := dataset.NumSamples / tt.k
				assert.InDelta(t, expectedTestSize, len(fold.TestIndices), 1)
				
				// Count appearances
				for _, idx := range fold.TestIndices {
					testCounts[idx]++
				}
				for _, idx := range fold.TrainIndices {
					trainCounts[idx]++
				}
			}
			
			// Each sample should appear exactly once in test
			for i := 0; i < dataset.NumSamples; i++ {
				assert.Equal(t, 1, testCounts[i], "Sample %d should appear once in test", i)
				assert.Equal(t, tt.k-1, trainCounts[i], "Sample %d should appear k-1 times in train", i)
			}
		})
	}
}

func TestKFoldCVErrors(t *testing.T) {
	dataset := createTestDataset(10, 0.5)
	
	// Test k < 2
	cv := NewKFoldCV(1, false, 42)
	_, err := cv.GetFolds(dataset)
	assert.Error(t, err)
	
	// Test k > n_samples
	cv = NewKFoldCV(20, false, 42)
	_, err = cv.GetFolds(dataset)
	assert.Error(t, err)
}

func TestStratifiedKFoldCV(t *testing.T) {
	// Create imbalanced dataset
	dataset := createTestDataset(100, 0.3)
	
	cv := NewStratifiedKFoldCV(5, true, 42)
	folds, err := cv.GetFolds(dataset)
	require.NoError(t, err)
	
	// Check stratification
	for i, fold := range folds {
		trainData := dataset.Subset(fold.TrainIndices)
		testData := dataset.Subset(fold.TestIndices)
		
		trainBalance := trainData.ClassBalance()
		testBalance := testData.ClassBalance()
		originalBalance := dataset.ClassBalance()
		
		// Allow some tolerance
		assert.InDelta(t, originalBalance, trainBalance, 0.1, 
			"Fold %d train balance should match original", i)
		assert.InDelta(t, originalBalance, testBalance, 0.1,
			"Fold %d test balance should match original", i)
	}
}

func TestStratifiedKFoldCVErrors(t *testing.T) {
	// Test with too few samples per class
	samples := []Sample{
		{Features: []float64{1}, Label: 0},
		{Features: []float64{2}, Label: 0},
		{Features: []float64{3}, Label: 1}, // Only 1 sample of class 1
	}
	dataset := NewDataset(samples)
	
	cv := NewStratifiedKFoldCV(5, false, 42)
	_, err := cv.GetFolds(dataset)
	assert.Error(t, err) // Should fail because class 1 has < 5 samples
}

func TestFoldConsistency(t *testing.T) {
	dataset := createTestDataset(50, 0.5)
	
	// Test that indices are sorted
	cv := NewKFoldCV(5, true, 42)
	folds, err := cv.GetFolds(dataset)
	require.NoError(t, err)
	
	for i, fold := range folds {
		// Check if indices are sorted
		for j := 1; j < len(fold.TrainIndices); j++ {
			assert.Less(t, fold.TrainIndices[j-1], fold.TrainIndices[j],
				"Fold %d train indices should be sorted", i)
		}
		for j := 1; j < len(fold.TestIndices); j++ {
			assert.Less(t, fold.TestIndices[j-1], fold.TestIndices[j],
				"Fold %d test indices should be sorted", i)
		}
	}
}

func BenchmarkStratifiedSplit(b *testing.B) {
	dataset := createTestDataset(10000, 0.3)
	splitter := NewStratifiedSplitter(0.2, 42)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = splitter.Split(dataset)
	}
}

func BenchmarkStratifiedKFold(b *testing.B) {
	dataset := createTestDataset(1000, 0.3)
	cv := NewStratifiedKFoldCV(5, true, 42)
	
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = cv.GetFolds(dataset)
	}
}

func TestSplitterSeedReproducibility(t *testing.T) {
	dataset := createTestDataset(100, 0.5)
	
	// Test that different seeds produce different results
	splitter1 := NewRandomSplitter(0.2, 42)
	splitter2 := NewRandomSplitter(0.2, 43)
	
	split1, _ := splitter1.Split(dataset)
	split2, _ := splitter2.Split(dataset)
	
	// Count how many samples are in the same position
	sameCount := 0
	for i := range split1.Test.Samples {
		if split1.Test.Samples[i].ID == split2.Test.Samples[i].ID {
			sameCount++
		}
	}
	
	// Should not be identical
	assert.NotEqual(t, len(split1.Test.Samples), sameCount,
		"Different seeds should produce different splits")
}

func TestEmptyDatasetSplitting(t *testing.T) {
	emptyDataset := NewDataset([]Sample{})
	
	// Random splitter
	splitter := NewRandomSplitter(0.2, 42)
	split, err := splitter.Split(emptyDataset)
	require.NoError(t, err)
	assert.Equal(t, 0, split.Train.NumSamples)
	assert.Equal(t, 0, split.Test.NumSamples)
	
	// K-fold
	cv := NewKFoldCV(5, false, 42)
	_, err = cv.GetFolds(emptyDataset)
	assert.Error(t, err) // Should error because k > n_samples
}