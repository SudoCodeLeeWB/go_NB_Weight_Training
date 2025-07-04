package metrics

import (
	"fmt"
	"sort"
)

// PRAUC implements Precision-Recall Area Under Curve metric
type PRAUC struct{}

// Calculate computes PR-AUC from predictions and labels
func (p *PRAUC) Calculate(predictions, labels []float64) (float64, error) {
	if len(predictions) != len(labels) {
		return 0, fmt.Errorf("predictions and labels must have same length")
	}
	
	if len(predictions) == 0 {
		return 0, fmt.Errorf("empty predictions")
	}
	
	// Get precision-recall curve points
	precisions, recalls := CalculatePRCurve(predictions, labels)
	
	// Calculate AUC using trapezoidal rule
	auc := calculateAUC(recalls, precisions)
	
	return auc, nil
}

// GetName returns the metric name
func (p *PRAUC) GetName() string {
	return "PR-AUC"
}

// CalculatePRCurve calculates precision-recall curve points
func CalculatePRCurve(predictions, labels []float64) ([]float64, []float64) {
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
	
	// Calculate precision and recall at each threshold
	precisions := []float64{1.0} // Start with precision 1 at recall 0
	recalls := []float64{0.0}
	
	tp := 0.0 // True positives
	fp := 0.0 // False positives
	
	// Count total positives
	totalPositives := 0.0
	for _, label := range labels {
		if label > 0.5 {
			totalPositives++
		}
	}
	
	// If no positive samples, return default curve
	if totalPositives == 0 {
		return []float64{0.0, 1.0}, []float64{0.0, 0.0}
	}
	
	// Process each prediction
	prevPred := pairs[0].pred + 1 // Initialize to value higher than any prediction
	
	for _, p := range pairs {
		// When prediction value changes, calculate metrics
		if p.pred != prevPred && (tp > 0 || fp > 0) {
			precision := tp / (tp + fp)
			recall := tp / totalPositives
			
			// Add point to curve
			precisions = append(precisions, precision)
			recalls = append(recalls, recall)
			
			prevPred = p.pred
		}
		
		// Update counts
		if p.label > 0.5 {
			tp++
		} else {
			fp++
		}
	}
	
	// Add final point
	if tp > 0 || fp > 0 {
		precision := tp / (tp + fp)
		recall := tp / totalPositives
		precisions = append(precisions, precision)
		recalls = append(recalls, recall)
	}
	
	// Ensure curve ends at recall=1 if not already
	if recalls[len(recalls)-1] < 1.0 {
		// Calculate precision at recall=1
		finalPrecision := totalPositives / float64(len(predictions))
		precisions = append(precisions, finalPrecision)
		recalls = append(recalls, 1.0)
	}
	
	return precisions, recalls
}

// calculateAUC computes area under curve using trapezoidal rule
func calculateAUC(x, y []float64) float64 {
	if len(x) != len(y) || len(x) < 2 {
		return 0.0
	}
	
	auc := 0.0
	for i := 1; i < len(x); i++ {
		// Trapezoidal area: 0.5 * (x2-x1) * (y1+y2)
		auc += TrapezoidalArea(x[i-1], x[i], y[i-1], y[i])
	}
	
	return auc
}

// PrecisionAtRecall calculates precision at a specific recall level
func PrecisionAtRecall(predictions, labels []float64, targetRecall float64) float64 {
	precisions, recalls := CalculatePRCurve(predictions, labels)
	
	// Find precision at target recall
	for i := 1; i < len(recalls); i++ {
		if recalls[i] >= targetRecall {
			// Linear interpolation
			if recalls[i] == recalls[i-1] {
				return precisions[i]
			}
			
			t := (targetRecall - recalls[i-1]) / (recalls[i] - recalls[i-1])
			return precisions[i-1] + t*(precisions[i]-precisions[i-1])
		}
	}
	
	// If target recall not reached, return last precision
	return precisions[len(precisions)-1]
}

// AveragePrecision calculates average precision (equivalent to PR-AUC)
func AveragePrecision(predictions, labels []float64) float64 {
	prauc := &PRAUC{}
	ap, _ := prauc.Calculate(predictions, labels)
	return ap
}