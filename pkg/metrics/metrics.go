package metrics

import (
	"sort"
)

// Metrics interface defines common metrics operations
type Metrics interface {
	Calculate(predictions, labels []float64) (float64, error)
	GetName() string
}

// ThresholdMetric represents metrics at a specific threshold
type ThresholdMetric struct {
	Threshold  float64
	Precision  float64
	Recall     float64
	FPR        float64 // False Positive Rate
	TPR        float64 // True Positive Rate (same as Recall)
	F1Score    float64
}

// ConfusionMatrix holds classification results
type ConfusionMatrix struct {
	TruePositives  int
	TrueNegatives  int
	FalsePositives int
	FalseNegatives int
}

// Calculate confusion matrix for binary classification
func CalculateConfusionMatrix(predictions, labels []float64, threshold float64) *ConfusionMatrix {
	cm := &ConfusionMatrix{}
	
	for i := range predictions {
		predicted := 0.0
		if predictions[i] >= threshold {
			predicted = 1.0
		}
		
		actual := labels[i]
		
		if predicted == 1.0 && actual == 1.0 {
			cm.TruePositives++
		} else if predicted == 0.0 && actual == 0.0 {
			cm.TrueNegatives++
		} else if predicted == 1.0 && actual == 0.0 {
			cm.FalsePositives++
		} else if predicted == 0.0 && actual == 1.0 {
			cm.FalseNegatives++
		}
	}
	
	return cm
}

// Precision calculates precision from confusion matrix
func (cm *ConfusionMatrix) Precision() float64 {
	if cm.TruePositives+cm.FalsePositives == 0 {
		return 0.0
	}
	return float64(cm.TruePositives) / float64(cm.TruePositives+cm.FalsePositives)
}

// Recall calculates recall from confusion matrix
func (cm *ConfusionMatrix) Recall() float64 {
	if cm.TruePositives+cm.FalseNegatives == 0 {
		return 0.0
	}
	return float64(cm.TruePositives) / float64(cm.TruePositives+cm.FalseNegatives)
}

// F1Score calculates F1 score from confusion matrix
func (cm *ConfusionMatrix) F1Score() float64 {
	precision := cm.Precision()
	recall := cm.Recall()
	
	if precision+recall == 0 {
		return 0.0
	}
	
	return 2 * (precision * recall) / (precision + recall)
}

// FPR calculates false positive rate
func (cm *ConfusionMatrix) FPR() float64 {
	if cm.FalsePositives+cm.TrueNegatives == 0 {
		return 0.0
	}
	return float64(cm.FalsePositives) / float64(cm.FalsePositives+cm.TrueNegatives)
}

// TPR calculates true positive rate (same as recall)
func (cm *ConfusionMatrix) TPR() float64 {
	return cm.Recall()
}

// CalculateThresholdMetrics calculates metrics at different thresholds
func CalculateThresholdMetrics(predictions, labels []float64) []ThresholdMetric {
	// Create unique thresholds from predictions
	uniqueThresholds := make(map[float64]bool)
	uniqueThresholds[0.0] = true
	uniqueThresholds[1.0] = true
	
	for _, pred := range predictions {
		uniqueThresholds[pred] = true
	}
	
	// Convert to sorted slice
	thresholds := make([]float64, 0, len(uniqueThresholds))
	for t := range uniqueThresholds {
		thresholds = append(thresholds, t)
	}
	sort.Float64s(thresholds)
	
	// Calculate metrics at each threshold
	metrics := make([]ThresholdMetric, len(thresholds))
	for i, threshold := range thresholds {
		cm := CalculateConfusionMatrix(predictions, labels, threshold)
		metrics[i] = ThresholdMetric{
			Threshold:  threshold,
			Precision:  cm.Precision(),
			Recall:     cm.Recall(),
			FPR:        cm.FPR(),
			TPR:        cm.TPR(),
			F1Score:    cm.F1Score(),
		}
	}
	
	return metrics
}

// TrapezoidalArea calculates area using trapezoidal rule
func TrapezoidalArea(x1, x2, y1, y2 float64) float64 {
	return 0.5 * (x2 - x1) * (y1 + y2)
}