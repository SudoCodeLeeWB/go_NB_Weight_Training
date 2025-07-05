package framework

import (
	"fmt"
	"math"
	"sort"

	"github.com/iwonbin/go-nb-weight-training/pkg/metrics"
)

// CompareCalibrationMethods tests multiple calibration methods on raw scores
func CompareCalibrationMethods(rawScores []float64, labels []float64, 
	optimizationMetric string, thresholdMetric string) (*CalibrationComparisonResult, error) {
	
	if len(rawScores) != len(labels) {
		return nil, fmt.Errorf("scores and labels must have same length")
	}
	
	// Calculate raw score distribution
	rawDist := calculateScoreDistribution(rawScores, labels)
	
	// Methods to test
	methods := []string{"beta", "isotonic", "platt", "none"}
	comparisons := make([]CalibrationComparison, 0, len(methods))
	
	// Test each calibration method
	for _, method := range methods {
		comparison, err := testCalibrationMethod(rawScores, labels, method, 
			optimizationMetric, thresholdMetric)
		if err != nil {
			// Log warning but continue with other methods
			fmt.Printf("Warning: calibration method %s failed: %v\n", method, err)
			continue
		}
		comparisons = append(comparisons, *comparison)
	}
	
	if len(comparisons) == 0 {
		return nil, fmt.Errorf("all calibration methods failed")
	}
	
	// Find best method based on optimization metric
	bestMethod, bestScore := findBestCalibrationMethod(comparisons, optimizationMetric)
	
	return &CalibrationComparisonResult{
		RawScoreDistribution:   rawDist,
		CalibrationComparisons: comparisons,
		BestMethod:            bestMethod,
		BestScore:             bestScore,
	}, nil
}

// testCalibrationMethod tests a single calibration method
func testCalibrationMethod(rawScores, labels []float64, method string,
	optimizationMetric, thresholdMetric string) (*CalibrationComparison, error) {
	
	// Fit calibration
	calibrationFunc, err := fitCalibrationMethod(rawScores, labels, method)
	if err != nil {
		return nil, err
	}
	
	// Apply calibration
	calibratedScores := make([]float64, len(rawScores))
	for i, score := range rawScores {
		calibratedScores[i] = calibrationFunc(score)
	}
	
	// Calculate score distribution
	scoreDist := calculateScoreDistribution(calibratedScores, labels)
	
	// Find optimal threshold
	optThreshold, _ := FindOptimalThreshold(calibratedScores, labels, thresholdMetric)
	
	// Calculate metrics at optimal threshold
	cm := metrics.CalculateConfusionMatrix(calibratedScores, labels, optThreshold)
	metricsAtThreshold := map[string]float64{
		"precision": cm.Precision(),
		"recall":    cm.Recall(),
		"f1_score":  cm.F1Score(),
		"accuracy":  cm.Accuracy(),
		"threshold": optThreshold,
	}
	
	// Calculate PR and ROC curves
	precisions, recalls := metrics.CalculatePRCurve(calibratedScores, labels)
	prAUC := metrics.AveragePrecision(calibratedScores, labels)
	prCurve := &CurveData{
		X:   recalls,
		Y:   precisions,
		AUC: prAUC,
	}
	
	fprs, tprs := metrics.CalculateROCCurve(calibratedScores, labels)
	rocAUC, _ := (&metrics.ROCAUC{}).Calculate(calibratedScores, labels)
	rocCurve := &CurveData{
		X:   fprs,
		Y:   tprs,
		AUC: rocAUC,
	}
	
	return &CalibrationComparison{
		Method:             method,
		OptimalThreshold:   optThreshold,
		ThresholdMetric:    thresholdMetric,
		MetricsAtThreshold: metricsAtThreshold,
		ScoreDistribution:  scoreDist,
		CalibrationFunc:    calibrationFunc,
		PRCurve:           prCurve,
		ROCCurve:          rocCurve,
	}, nil
}

// fitCalibrationMethod fits a calibration function based on method name
func fitCalibrationMethod(scores, labels []float64, method string) (func(float64) float64, error) {
	// Calculate score range for all methods
	scoreRange := fitScoreRange(scores)
	
	switch method {
	case "beta":
		return fitBetaCalibration(scores, labels), nil
	case "isotonic":
		return fitIsotonicRegression(scores, labels), nil
	case "platt":
		return fitPlattScaling(scores, labels), nil
	case "none":
		// Simple min-max normalization
		return func(score float64) float64 {
			if scoreRange.MaxScore == scoreRange.MinScore {
				return 0.5
			}
			normalized := (score - scoreRange.MinScore) / 
			              (scoreRange.MaxScore - scoreRange.MinScore)
			if normalized < 0 {
				return 0
			}
			if normalized > 1 {
				return 1
			}
			return normalized
		}, nil
	default:
		return nil, fmt.Errorf("unknown calibration method: %s", method)
	}
}

// calculateScoreDistribution calculates distribution statistics
func calculateScoreDistribution(scores, labels []float64) ScoreDistribution {
	n := len(scores)
	if n == 0 {
		return ScoreDistribution{}
	}
	
	// Sort scores for percentiles
	sorted := make([]float64, n)
	copy(sorted, scores)
	sort.Float64s(sorted)
	
	// Calculate mean
	sum := 0.0
	for _, s := range scores {
		sum += s
	}
	mean := sum / float64(n)
	
	// Calculate std
	variance := 0.0
	for _, s := range scores {
		diff := s - mean
		variance += diff * diff
	}
	std := math.Sqrt(variance / float64(n))
	
	// Calculate percentiles
	percentiles := make(map[int]float64)
	for _, p := range []int{0, 25, 50, 75, 100} {
		idx := int(float64(p) / 100.0 * float64(n-1))
		if idx >= n {
			idx = n - 1
		}
		percentiles[p] = sorted[idx]
	}
	
	// Create histogram with 20 bins
	histogram := createHistogram(scores, 20)
	
	return ScoreDistribution{
		Min:         sorted[0],
		Max:         sorted[n-1],
		Mean:        mean,
		Std:         std,
		Percentiles: percentiles,
		Histogram:   histogram,
	}
}

// createHistogram creates histogram bins
func createHistogram(scores []float64, numBins int) []HistogramBin {
	if len(scores) == 0 || numBins <= 0 {
		return nil
	}
	
	// Find min and max
	min, max := scores[0], scores[0]
	for _, s := range scores[1:] {
		if s < min {
			min = s
		}
		if s > max {
			max = s
		}
	}
	
	// Handle case where all scores are the same
	if max == min {
		return []HistogramBin{{
			Start: min,
			End:   min,
			Count: len(scores),
			Ratio: 1.0,
		}}
	}
	
	// Create bins
	binWidth := (max - min) / float64(numBins)
	bins := make([]HistogramBin, numBins)
	
	for i := range bins {
		bins[i].Start = min + float64(i)*binWidth
		bins[i].End = min + float64(i+1)*binWidth
		if i == numBins-1 {
			bins[i].End = max // Ensure last bin includes max
		}
	}
	
	// Count scores in each bin
	for _, score := range scores {
		binIdx := int((score - min) / binWidth)
		if binIdx >= numBins {
			binIdx = numBins - 1
		}
		bins[binIdx].Count++
	}
	
	// Calculate ratios
	total := float64(len(scores))
	for i := range bins {
		bins[i].Ratio = float64(bins[i].Count) / total
	}
	
	return bins
}

// findBestCalibrationMethod finds the best method based on optimization metric
func findBestCalibrationMethod(comparisons []CalibrationComparison, metric string) (string, float64) {
	if len(comparisons) == 0 {
		return "", 0
	}
	
	bestMethod := comparisons[0].Method
	bestScore := 0.0
	
	for _, comp := range comparisons {
		var score float64
		switch metric {
		case "pr_auc":
			score = comp.PRCurve.AUC
		case "roc_auc":
			score = comp.ROCCurve.AUC
		case "precision":
			score = comp.MetricsAtThreshold["precision"]
		case "recall":
			score = comp.MetricsAtThreshold["recall"]
		case "f1_score":
			score = comp.MetricsAtThreshold["f1_score"]
		default:
			score = comp.PRCurve.AUC
		}
		
		if score > bestScore {
			bestScore = score
			bestMethod = comp.Method
		}
	}
	
	return bestMethod, bestScore
}

// TestModelCalibration tests a model's own calibration against other methods
func TestModelCalibration(model CalibratedAggregatedModel, features [][]float64, labels []float64,
	optimizationMetric, thresholdMetric string) (*CalibrationComparisonResult, error) {
	
	// Get raw and calibrated scores from model
	rawScores, modelCalibratedScores, err := model.PredictWithCalibration(features)
	if err != nil {
		return nil, fmt.Errorf("failed to get predictions: %w", err)
	}
	
	// Compare standard calibration methods on raw scores
	result, err := CompareCalibrationMethods(rawScores, labels, optimizationMetric, thresholdMetric)
	if err != nil {
		return nil, err
	}
	
	// Store validation data for potential visualization
	result.ValidationFeatures = features
	result.ValidationLabels = labels
	
	// Test model's own calibration
	modelCalibration, err := testModelProvidedCalibration(modelCalibratedScores, labels, 
		model.GetCalibrationMethod(), optimizationMetric, thresholdMetric)
	if err == nil {
		result.ModelProvidedCalibration = modelCalibration
		
		// Check if model's calibration is better than the best standard method
		modelScore := 0.0
		switch optimizationMetric {
		case "pr_auc":
			modelScore = modelCalibration.PRCurve.AUC
		case "roc_auc":
			modelScore = modelCalibration.ROCCurve.AUC
		default:
			modelScore = modelCalibration.PRCurve.AUC
		}
		
		if modelScore > result.BestScore {
			result.BestMethod = fmt.Sprintf("Model's %s", model.GetCalibrationMethod())
			result.BestScore = modelScore
		}
	}
	
	return result, nil
}

// testModelProvidedCalibration tests the model's own calibration
func testModelProvidedCalibration(calibratedScores, labels []float64, methodName string,
	optimizationMetric, thresholdMetric string) (*CalibrationComparison, error) {
	
	// Calculate score distribution
	scoreDist := calculateScoreDistribution(calibratedScores, labels)
	
	// Find optimal threshold
	optThreshold, _ := FindOptimalThreshold(calibratedScores, labels, thresholdMetric)
	
	// Calculate metrics at optimal threshold
	cm := metrics.CalculateConfusionMatrix(calibratedScores, labels, optThreshold)
	metricsAtThreshold := map[string]float64{
		"precision": cm.Precision(),
		"recall":    cm.Recall(),
		"f1_score":  cm.F1Score(),
		"accuracy":  cm.Accuracy(),
		"threshold": optThreshold,
	}
	
	// Calculate PR and ROC curves
	precisions, recalls := metrics.CalculatePRCurve(calibratedScores, labels)
	prAUC := metrics.AveragePrecision(calibratedScores, labels)
	prCurve := &CurveData{
		X:   recalls,
		Y:   precisions,
		AUC: prAUC,
	}
	
	fprs, tprs := metrics.CalculateROCCurve(calibratedScores, labels)
	rocAUC, _ := (&metrics.ROCAUC{}).Calculate(calibratedScores, labels)
	rocCurve := &CurveData{
		X:   fprs,
		Y:   tprs,
		AUC: rocAUC,
	}
	
	return &CalibrationComparison{
		Method:             methodName,
		OptimalThreshold:   optThreshold,
		ThresholdMetric:    thresholdMetric,
		MetricsAtThreshold: metricsAtThreshold,
		ScoreDistribution:  scoreDist,
		PRCurve:           prCurve,
		ROCCurve:          rocCurve,
	}, nil
}