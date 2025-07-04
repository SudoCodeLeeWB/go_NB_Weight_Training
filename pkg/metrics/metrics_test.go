package metrics

import (
	"math"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestConfusionMatrix(t *testing.T) {
	tests := []struct {
		name        string
		predictions []float64
		labels      []float64
		threshold   float64
		expectedCM  ConfusionMatrix
	}{
		{
			name:        "Perfect predictions",
			predictions: []float64{0.9, 0.8, 0.1, 0.2},
			labels:      []float64{1, 1, 0, 0},
			threshold:   0.5,
			expectedCM: ConfusionMatrix{
				TruePositives:  2,
				TrueNegatives:  2,
				FalsePositives: 0,
				FalseNegatives: 0,
			},
		},
		{
			name:        "All false positives",
			predictions: []float64{0.9, 0.8, 0.7, 0.6},
			labels:      []float64{0, 0, 0, 0},
			threshold:   0.5,
			expectedCM: ConfusionMatrix{
				TruePositives:  0,
				TrueNegatives:  0,
				FalsePositives: 4,
				FalseNegatives: 0,
			},
		},
		{
			name:        "Mixed results",
			predictions: []float64{0.9, 0.3, 0.7, 0.2},
			labels:      []float64{1, 1, 0, 0},
			threshold:   0.5,
			expectedCM: ConfusionMatrix{
				TruePositives:  1,
				TrueNegatives:  2,
				FalsePositives: 1,
				FalseNegatives: 1,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			cm := CalculateConfusionMatrix(tt.predictions, tt.labels, tt.threshold)
			assert.Equal(t, tt.expectedCM, *cm)
		})
	}
}

func TestConfusionMatrixMetrics(t *testing.T) {
	cm := &ConfusionMatrix{
		TruePositives:  30,
		TrueNegatives:  40,
		FalsePositives: 10,
		FalseNegatives: 20,
	}

	// Test Precision
	expectedPrecision := 30.0 / (30.0 + 10.0)
	assert.InDelta(t, expectedPrecision, cm.Precision(), 1e-6)

	// Test Recall
	expectedRecall := 30.0 / (30.0 + 20.0)
	assert.InDelta(t, expectedRecall, cm.Recall(), 1e-6)

	// Test F1Score
	expectedF1 := 2 * (expectedPrecision * expectedRecall) / (expectedPrecision + expectedRecall)
	assert.InDelta(t, expectedF1, cm.F1Score(), 1e-6)

	// Test FPR
	expectedFPR := 10.0 / (10.0 + 40.0)
	assert.InDelta(t, expectedFPR, cm.FPR(), 1e-6)

	// Test TPR (should equal recall)
	assert.Equal(t, cm.Recall(), cm.TPR())
}

func TestEdgeCases(t *testing.T) {
	// Test zero division cases
	t.Run("No positive predictions", func(t *testing.T) {
		cm := &ConfusionMatrix{
			TruePositives:  0,
			TrueNegatives:  50,
			FalsePositives: 0,
			FalseNegatives: 50,
		}
		assert.Equal(t, 0.0, cm.Precision())
	})

	t.Run("No actual positives", func(t *testing.T) {
		cm := &ConfusionMatrix{
			TruePositives:  0,
			TrueNegatives:  50,
			FalsePositives: 50,
			FalseNegatives: 0,
		}
		assert.Equal(t, 0.0, cm.Recall())
	})

	t.Run("No actual negatives", func(t *testing.T) {
		cm := &ConfusionMatrix{
			TruePositives:  50,
			TrueNegatives:  0,
			FalsePositives: 0,
			FalseNegatives: 50,
		}
		assert.Equal(t, 0.0, cm.FPR())
	})
}

func TestPRAUC(t *testing.T) {
	tests := []struct {
		name        string
		predictions []float64
		labels      []float64
		minAUC      float64
		maxAUC      float64
	}{
		{
			name:        "Perfect classifier",
			predictions: []float64{0.9, 0.8, 0.7, 0.1, 0.2, 0.3},
			labels:      []float64{1, 1, 1, 0, 0, 0},
			minAUC:      0.99,
			maxAUC:      1.0,
		},
		{
			name:        "Random predictions",
			predictions: []float64{0.5, 0.6, 0.4, 0.5, 0.6, 0.4},
			labels:      []float64{1, 0, 1, 0, 1, 0},
			minAUC:      0.4,
			maxAUC:      0.6,
		},
		{
			name:        "Worst case",
			predictions: []float64{0.1, 0.2, 0.3, 0.9, 0.8, 0.7},
			labels:      []float64{1, 1, 1, 0, 0, 0},
			minAUC:      0.0,
			maxAUC:      0.5,
		},
	}

	prauc := &PRAUC{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			auc, err := prauc.Calculate(tt.predictions, tt.labels)
			assert.NoError(t, err)
			assert.GreaterOrEqual(t, auc, tt.minAUC)
			assert.LessOrEqual(t, auc, tt.maxAUC)
		})
	}
}

func TestROCAUC(t *testing.T) {
	tests := []struct {
		name        string
		predictions []float64
		labels      []float64
		minAUC      float64
		maxAUC      float64
	}{
		{
			name:        "Perfect classifier",
			predictions: []float64{0.9, 0.8, 0.7, 0.1, 0.2, 0.3},
			labels:      []float64{1, 1, 1, 0, 0, 0},
			minAUC:      0.99,
			maxAUC:      1.0,
		},
		{
			name:        "Random predictions",
			predictions: []float64{0.5, 0.6, 0.4, 0.5, 0.6, 0.4},
			labels:      []float64{1, 0, 1, 0, 1, 0},
			minAUC:      0.4,
			maxAUC:      0.6,
		},
		{
			name:        "Inverse predictions",
			predictions: []float64{0.1, 0.2, 0.3, 0.9, 0.8, 0.7},
			labels:      []float64{1, 1, 1, 0, 0, 0},
			minAUC:      0.0,
			maxAUC:      0.1,
		},
	}

	rocauc := &ROCAUC{}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			auc, err := rocauc.Calculate(tt.predictions, tt.labels)
			assert.NoError(t, err)
			assert.GreaterOrEqual(t, auc, tt.minAUC)
			assert.LessOrEqual(t, auc, tt.maxAUC)
		})
	}
}

func TestPRCurve(t *testing.T) {
	predictions := []float64{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2}
	labels := []float64{1, 1, 0, 1, 0, 0, 1, 0}

	precisions, recalls := CalculatePRCurve(predictions, labels)

	// Check that curves start and end correctly
	assert.Equal(t, 1.0, precisions[0], "Precision should start at 1.0")
	assert.Equal(t, 0.0, recalls[0], "Recall should start at 0.0")

	// Check monotonicity
	for i := 1; i < len(recalls); i++ {
		assert.GreaterOrEqual(t, recalls[i], recalls[i-1], "Recall should be non-decreasing")
	}

	// Check that we reach recall of 1.0
	assert.Equal(t, 1.0, recalls[len(recalls)-1], "Recall should end at 1.0")
}

func TestROCCurve(t *testing.T) {
	predictions := []float64{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2}
	labels := []float64{1, 1, 0, 1, 0, 0, 1, 0}

	fprs, tprs := CalculateROCCurve(predictions, labels)

	// Check that curves start and end correctly
	assert.Equal(t, 0.0, fprs[0], "FPR should start at 0.0")
	assert.Equal(t, 0.0, tprs[0], "TPR should start at 0.0")
	assert.Equal(t, 1.0, fprs[len(fprs)-1], "FPR should end at 1.0")
	assert.Equal(t, 1.0, tprs[len(tprs)-1], "TPR should end at 1.0")

	// Check monotonicity
	for i := 1; i < len(fprs); i++ {
		assert.GreaterOrEqual(t, fprs[i], fprs[i-1], "FPR should be non-decreasing")
		assert.GreaterOrEqual(t, tprs[i], tprs[i-1], "TPR should be non-decreasing")
	}
}

func TestMetricErrors(t *testing.T) {
	prauc := &PRAUC{}
	rocauc := &ROCAUC{}

	// Test mismatched lengths
	predictions := []float64{0.5, 0.6}
	labels := []float64{1}

	_, err := prauc.Calculate(predictions, labels)
	assert.Error(t, err)

	_, err = rocauc.Calculate(predictions, labels)
	assert.Error(t, err)

	// Test empty inputs
	_, err = prauc.Calculate([]float64{}, []float64{})
	assert.Error(t, err)

	_, err = rocauc.Calculate([]float64{}, []float64{})
	assert.Error(t, err)
}

func TestTrapezoidalArea(t *testing.T) {
	tests := []struct {
		x1, x2, y1, y2 float64
		expected       float64
	}{
		{0, 1, 0, 1, 0.5},
		{0, 1, 1, 1, 1.0},
		{0, 0.5, 0.5, 1, 0.375},
		{0.5, 1, 0, 0.5, 0.125},
	}

	for _, tt := range tests {
		result := TrapezoidalArea(tt.x1, tt.x2, tt.y1, tt.y2)
		assert.InDelta(t, tt.expected, result, 1e-6)
	}
}

func BenchmarkPRAUC(b *testing.B) {
	// Create test data
	n := 1000
	predictions := make([]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		predictions[i] = float64(i) / float64(n)
		if i < n/2 {
			labels[i] = 1
		} else {
			labels[i] = 0
		}
	}

	prauc := &PRAUC{}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = prauc.Calculate(predictions, labels)
	}
}

func BenchmarkROCAUC(b *testing.B) {
	// Create test data
	n := 1000
	predictions := make([]float64, n)
	labels := make([]float64, n)
	for i := 0; i < n; i++ {
		predictions[i] = float64(i) / float64(n)
		if i < n/2 {
			labels[i] = 1
		} else {
			labels[i] = 0
		}
	}

	rocauc := &ROCAUC{}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		_, _ = rocauc.Calculate(predictions, labels)
	}
}

// Helper function to check if two float slices are approximately equal
func assertFloatSlicesEqual(t *testing.T, expected, actual []float64, delta float64) {
	assert.Equal(t, len(expected), len(actual), "Slices should have same length")
	for i := range expected {
		assert.InDelta(t, expected[i], actual[i], delta, "Values at index %d should be equal", i)
	}
}

func TestPrecisionAtRecall(t *testing.T) {
	predictions := []float64{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2}
	labels := []float64{1, 1, 0, 1, 0, 0, 1, 0}

	tests := []struct {
		targetRecall float64
		minPrecision float64
	}{
		{0.0, 1.0},    // At zero recall, precision should be 1
		{0.25, 0.8},   // At 25% recall
		{0.5, 0.6},    // At 50% recall
		{0.75, 0.4},   // At 75% recall
		{1.0, 0.0},    // At 100% recall
	}

	for _, tt := range tests {
		precision := PrecisionAtRecall(predictions, labels, tt.targetRecall)
		assert.GreaterOrEqual(t, precision, tt.minPrecision)
	}
}

func TestOptimalThresholdROC(t *testing.T) {
	predictions := []float64{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2}
	labels := []float64{1, 1, 0, 1, 0, 0, 1, 0}

	threshold, tpr, fpr := OptimalThresholdROC(predictions, labels)

	// Check that threshold is valid
	assert.GreaterOrEqual(t, threshold, 0.0)
	assert.LessOrEqual(t, threshold, 1.0)

	// Check that TPR and FPR are valid
	assert.GreaterOrEqual(t, tpr, 0.0)
	assert.LessOrEqual(t, tpr, 1.0)
	assert.GreaterOrEqual(t, fpr, 0.0)
	assert.LessOrEqual(t, fpr, 1.0)

	// Youden's J should be maximized
	j := tpr - fpr
	assert.Greater(t, j, 0.0)
}

func TestPartialROCAUC(t *testing.T) {
	predictions := []float64{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2}
	labels := []float64{1, 1, 0, 1, 0, 0, 1, 0}

	// Test partial AUC at different FPR limits
	tests := []struct {
		maxFPR      float64
		minPartialAUC float64
		maxPartialAUC float64
	}{
		{0.1, 0.0, 0.1},
		{0.5, 0.0, 0.5},
		{1.0, 0.0, 1.0},
	}

	for _, tt := range tests {
		partialAUC := PartialROCAUC(predictions, labels, tt.maxFPR)
		assert.GreaterOrEqual(t, partialAUC, tt.minPartialAUC)
		assert.LessOrEqual(t, partialAUC, tt.maxPartialAUC)
	}
}

func TestAveragePrecision(t *testing.T) {
	predictions := []float64{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2}
	labels := []float64{1, 1, 0, 1, 0, 0, 1, 0}

	ap := AveragePrecision(predictions, labels)
	
	// Average precision should be between 0 and 1
	assert.GreaterOrEqual(t, ap, 0.0)
	assert.LessOrEqual(t, ap, 1.0)

	// It should equal PR-AUC
	prauc := &PRAUC{}
	praucValue, _ := prauc.Calculate(predictions, labels)
	assert.Equal(t, praucValue, ap)
}

func TestCalculateThresholdMetrics(t *testing.T) {
	predictions := []float64{0.9, 0.7, 0.3, 0.1}
	labels := []float64{1, 1, 0, 0}

	metrics := CalculateThresholdMetrics(predictions, labels)

	// Should have metrics for each unique threshold plus 0 and 1
	assert.GreaterOrEqual(t, len(metrics), len(predictions))

	// First threshold should be 0, last should be 1
	assert.Equal(t, 0.0, metrics[0].Threshold)
	assert.Equal(t, 1.0, metrics[len(metrics)-1].Threshold)

	// At threshold 0, all predictions are positive
	assert.Equal(t, 1.0, metrics[0].Recall) // All positives caught
	assert.Equal(t, 0.5, metrics[0].Precision) // 2 TP, 2 FP

	// At threshold 1, all predictions are negative
	lastIdx := len(metrics) - 1
	assert.Equal(t, 0.0, metrics[lastIdx].Recall) // No positives caught
	assert.Equal(t, 0.0, metrics[lastIdx].Precision) // No positive predictions
}

func TestEdgeCasesSingleClass(t *testing.T) {
	// Test when all samples are positive
	t.Run("All positive samples", func(t *testing.T) {
		predictions := []float64{0.9, 0.8, 0.7, 0.6}
		labels := []float64{1, 1, 1, 1}

		fprs, tprs := CalculateROCCurve(predictions, labels)
		assert.Equal(t, []float64{0.0, 1.0}, fprs)
		assert.Equal(t, []float64{0.0, 1.0}, tprs)
	})

	// Test when all samples are negative
	t.Run("All negative samples", func(t *testing.T) {
		predictions := []float64{0.9, 0.8, 0.7, 0.6}
		labels := []float64{0, 0, 0, 0}

		fprs, tprs := CalculateROCCurve(predictions, labels)
		assert.Equal(t, []float64{0.0, 1.0}, fprs)
		assert.Equal(t, []float64{0.0, 1.0}, tprs)
	})
}

func TestNaNAndInfHandling(t *testing.T) {
	// Test F1 score when precision and recall are both 0
	cm := &ConfusionMatrix{
		TruePositives:  0,
		TrueNegatives:  10,
		FalsePositives: 0,
		FalseNegatives: 10,
	}
	
	f1 := cm.F1Score()
	assert.False(t, math.IsNaN(f1))
	assert.False(t, math.IsInf(f1, 0))
	assert.Equal(t, 0.0, f1)
}