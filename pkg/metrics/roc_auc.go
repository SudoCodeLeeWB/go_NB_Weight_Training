package metrics

import (
	"fmt"
	"sort"
)

// ROCAUC implements ROC Area Under Curve metric
type ROCAUC struct{}

// Calculate computes ROC-AUC from predictions and labels
func (r *ROCAUC) Calculate(predictions, labels []float64) (float64, error) {
	if len(predictions) != len(labels) {
		return 0, fmt.Errorf("predictions and labels must have same length")
	}
	
	if len(predictions) == 0 {
		return 0, fmt.Errorf("empty predictions")
	}
	
	// Get ROC curve points
	fprs, tprs := CalculateROCCurve(predictions, labels)
	
	// Calculate AUC using trapezoidal rule
	auc := calculateAUC(fprs, tprs)
	
	return auc, nil
}

// GetName returns the metric name
func (r *ROCAUC) GetName() string {
	return "ROC-AUC"
}

// CalculateROCCurve calculates ROC curve points (FPR, TPR)
func CalculateROCCurve(predictions, labels []float64) ([]float64, []float64) {
	// Create pairs of (prediction, label) and sort by prediction descending
	type pair struct {
		pred  float64
		label float64
	}
	
	pairs := make([]pair, len(predictions))
	for i := range predictions {
		pairs[i] = pair{pred: predictions[i], label: labels[i]}
	}
	
	// Sort by prediction descending
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].pred > pairs[j].pred
	})
	
	// Count total positives and negatives
	totalPositives := 0.0
	totalNegatives := 0.0
	for _, label := range labels {
		if label > 0.5 {
			totalPositives++
		} else {
			totalNegatives++
		}
	}
	
	// Handle edge cases
	if totalPositives == 0 || totalNegatives == 0 {
		// If all samples are of one class, ROC curve is undefined
		return []float64{0.0, 1.0}, []float64{0.0, 1.0}
	}
	
	// Calculate FPR and TPR at each threshold
	fprs := []float64{0.0}
	tprs := []float64{0.0}
	
	tp := 0.0 // True positives
	fp := 0.0 // False positives
	
	prevPred := pairs[0].pred + 1 // Initialize to value higher than any prediction
	
	for _, p := range pairs {
		// When prediction value changes, calculate metrics
		if p.pred != prevPred && (tp > 0 || fp > 0) {
			fpr := fp / totalNegatives
			tpr := tp / totalPositives
			
			// Add point to curve
			fprs = append(fprs, fpr)
			tprs = append(tprs, tpr)
			
			prevPred = p.pred
		}
		
		// Update counts
		if p.label > 0.5 {
			tp++
		} else {
			fp++
		}
	}
	
	// Add final point (1, 1)
	fprs = append(fprs, 1.0)
	tprs = append(tprs, 1.0)
	
	return fprs, tprs
}

// CalculateROCPoint calculates a single point on the ROC curve
func CalculateROCPoint(predictions, labels []float64, threshold float64) (fpr, tpr float64) {
	cm := CalculateConfusionMatrix(predictions, labels, threshold)
	return cm.FPR(), cm.TPR()
}

// OptimalThresholdROC finds the optimal threshold based on Youden's J statistic
func OptimalThresholdROC(predictions, labels []float64) (float64, float64, float64) {
	metrics := CalculateThresholdMetrics(predictions, labels)
	
	bestThreshold := 0.5
	bestJ := -1.0
	bestTPR := 0.0
	bestFPR := 0.0
	
	for _, m := range metrics {
		// Youden's J = TPR - FPR
		j := m.TPR - m.FPR
		if j > bestJ {
			bestJ = j
			bestThreshold = m.Threshold
			bestTPR = m.TPR
			bestFPR = m.FPR
		}
	}
	
	return bestThreshold, bestTPR, bestFPR
}

// PartialROCAUC calculates partial AUC for a specific FPR range
func PartialROCAUC(predictions, labels []float64, maxFPR float64) float64 {
	fprs, tprs := CalculateROCCurve(predictions, labels)
	
	// Find points within FPR range
	partialFPRs := []float64{0.0}
	partialTPRs := []float64{0.0}
	
	for i := 1; i < len(fprs); i++ {
		if fprs[i] <= maxFPR {
			partialFPRs = append(partialFPRs, fprs[i])
			partialTPRs = append(partialTPRs, tprs[i])
		} else {
			// Interpolate to get TPR at maxFPR
			if fprs[i-1] < maxFPR {
				t := (maxFPR - fprs[i-1]) / (fprs[i] - fprs[i-1])
				interpolatedTPR := tprs[i-1] + t*(tprs[i]-tprs[i-1])
				partialFPRs = append(partialFPRs, maxFPR)
				partialTPRs = append(partialTPRs, interpolatedTPR)
			}
			break
		}
	}
	
	// Calculate partial AUC
	return calculateAUC(partialFPRs, partialTPRs)
}