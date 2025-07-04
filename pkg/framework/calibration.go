package framework

import (
	"math"
	"sort"
)

// CalibratedEnsemble wraps ensemble with probability calibration
type CalibratedEnsemble struct {
	*EnsembleModel
	UseLogSpace      bool
	ScoreRange       ScoreRange
	CalibrationFunc  func(float64) float64
	CalibrationMethod string // "platt", "isotonic", "beta", "none"
}

// ScoreRange tracks min/max scores for normalization
type ScoreRange struct {
	MinScore   float64
	MaxScore   float64
	Percentiles []float64  // For percentile-based calibration
}

// PredictCalibrated returns calibrated probabilities in [0,1]
func (ce *CalibratedEnsemble) PredictCalibrated(samples [][]float64) ([]float64, error) {
	n := len(samples)
	scores := make([]float64, n)
	
	if ce.UseLogSpace {
		// Compute in log space to avoid underflow
		for i := range samples {
			logScore := 0.0
			
			for idx, model := range ce.Models {
				predictions, err := model.Predict(samples[i:i+1])
				if err != nil {
					return nil, err
				}
				
				p := predictions[0]
				// Clamp to avoid log(0)
				if p <= 0 {
					p = 1e-10
				} else if p >= 1 {
					p = 1 - 1e-10
				}
				
				logScore += ce.Weights[idx] * math.Log(p)
			}
			
			scores[i] = logScore
		}
	} else {
		// Regular multiplication (will underflow for many models)
		predictions, err := ce.EnsembleModel.Predict(samples)
		if err != nil {
			return nil, err
		}
		scores = predictions
	}
	
	// Apply calibration
	calibrated := make([]float64, n)
	for i, score := range scores {
		if ce.CalibrationFunc != nil {
			calibrated[i] = ce.CalibrationFunc(score)
		} else {
			// Default: min-max normalization
			calibrated[i] = ce.normalizeScore(score)
		}
	}
	
	return calibrated, nil
}

// FitCalibration fits calibration on validation data
func (ce *CalibratedEnsemble) FitCalibration(samples [][]float64, labels []float64) error {
	// Get raw scores
	scores := make([]float64, len(samples))
	
	if ce.UseLogSpace {
		for i := range samples {
			logScore := 0.0
			
			for idx, model := range ce.Models {
				predictions, err := model.Predict(samples[i:i+1])
				if err != nil {
					return err
				}
				
				p := predictions[0]
				if p <= 0 {
					p = 1e-10
				} else if p >= 1 {
					p = 1 - 1e-10
				}
				
				logScore += ce.Weights[idx] * math.Log(p)
			}
			
			scores[i] = logScore
		}
	} else {
		predictions, err := ce.EnsembleModel.Predict(samples)
		if err != nil {
			return err
		}
		scores = predictions
	}
	
	// Fit score range
	ce.ScoreRange = fitScoreRange(scores)
	
	// Fit calibration function based on method
	switch ce.CalibrationMethod {
	case "platt":
		ce.CalibrationFunc = fitPlattScaling(scores, labels)
	case "isotonic":
		ce.CalibrationFunc = fitIsotonicRegression(scores, labels)
	case "beta":
		ce.CalibrationFunc = fitBetaCalibration(scores, labels)
	case "none":
		// Just use min-max normalization
		ce.CalibrationFunc = nil
	default:
		// Default to beta calibration (less aggressive than Platt)
		ce.CalibrationMethod = "beta"
		ce.CalibrationFunc = fitBetaCalibration(scores, labels)
	}
	
	return nil
}

// normalizeScore applies min-max normalization
func (ce *CalibratedEnsemble) normalizeScore(score float64) float64 {
	if ce.ScoreRange.MaxScore == ce.ScoreRange.MinScore {
		return 0.5
	}
	
	normalized := (score - ce.ScoreRange.MinScore) / 
	              (ce.ScoreRange.MaxScore - ce.ScoreRange.MinScore)
	
	// Clamp to [0,1]
	if normalized < 0 {
		return 0
	}
	if normalized > 1 {
		return 1
	}
	return normalized
}

// fitScoreRange computes score statistics
func fitScoreRange(scores []float64) ScoreRange {
	if len(scores) == 0 {
		return ScoreRange{MinScore: 0, MaxScore: 1}
	}
	
	// Copy and sort for percentiles
	sorted := make([]float64, len(scores))
	copy(sorted, scores)
	sort.Float64s(sorted)
	
	// Compute percentiles
	percentiles := make([]float64, 101)
	for i := 0; i <= 100; i++ {
		idx := int(float64(i) / 100.0 * float64(len(sorted)-1))
		percentiles[i] = sorted[idx]
	}
	
	return ScoreRange{
		MinScore:    sorted[0],
		MaxScore:    sorted[len(sorted)-1],
		Percentiles: percentiles,
	}
}

// fitPlattScaling fits sigmoid calibration
func fitPlattScaling(scores, labels []float64) func(float64) float64 {
	// Simple Platt scaling: sigmoid(ax + b)
	// This is a simplified version - in production use proper optimization
	
	// Sort by score
	type pair struct {
		score float64
		label float64
	}
	pairs := make([]pair, len(scores))
	for i := range scores {
		pairs[i] = pair{scores[i], labels[i]}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].score < pairs[j].score
	})
	
	// Find score that best separates classes
	bestThreshold := pairs[len(pairs)/2].score
	bestF1 := 0.0
	
	for i := 1; i < len(pairs)-1; i++ {
		threshold := (pairs[i].score + pairs[i-1].score) / 2
		
		tp, fp, fn := 0.0, 0.0, 0.0
		for _, p := range pairs {
			if p.score >= threshold {
				if p.label > 0.5 {
					tp++
				} else {
					fp++
				}
			} else {
				if p.label > 0.5 {
					fn++
				}
			}
		}
		
		precision := tp / (tp + fp + 1e-10)
		recall := tp / (tp + fn + 1e-10)
		f1 := 2 * precision * recall / (precision + recall + 1e-10)
		
		if f1 > bestF1 {
			bestF1 = f1
			bestThreshold = threshold
		}
	}
	
	// Fit sigmoid around threshold
	// a controls steepness, b controls center
	a := -4.0 / (pairs[len(pairs)-1].score - pairs[0].score)
	b := -a * bestThreshold
	
	return func(score float64) float64 {
		return 1.0 / (1.0 + math.Exp(a*score+b))
	}
}

// FindOptimalThreshold finds best threshold for a given metric
func FindOptimalThreshold(predictions, labels []float64, metric string) (float64, float64) {
	type result struct {
		threshold float64
		score     float64
		precision float64
		recall    float64
		f1        float64
		mcc       float64  // Matthews Correlation Coefficient
		distance  float64  // Distance from perfect point
	}
	
	// Try different thresholds
	thresholds := []float64{}
	for i := 0; i <= 100; i++ {
		thresholds = append(thresholds, float64(i)/100.0)
	}
	
	results := make([]result, len(thresholds))
	
	for i, threshold := range thresholds {
		tp, tn, fp, fn := 0.0, 0.0, 0.0, 0.0
		
		for j, pred := range predictions {
			if pred >= threshold {
				if labels[j] > 0.5 {
					tp++
				} else {
					fp++
				}
			} else {
				if labels[j] > 0.5 {
					fn++
				} else {
					tn++
				}
			}
		}
		
		precision := tp / (tp + fp + 1e-10)
		recall := tp / (tp + fn + 1e-10)
		f1 := 2 * precision * recall / (precision + recall + 1e-10)
		accuracy := (tp + tn) / float64(len(predictions))
		
		// Matthews Correlation Coefficient
		mcc := (tp*tn - fp*fn) / math.Sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn) + 1e-10)
		
		// Distance from perfect point (1, 1) in PR space
		distance := math.Sqrt(math.Pow(1-precision, 2) + math.Pow(1-recall, 2))
		
		r := result{
			threshold: threshold,
			precision: precision,
			recall:    recall,
			f1:        f1,
			mcc:       mcc,
			distance:  distance,
		}
		
		switch metric {
		case "f1":
			r.score = f1
		case "precision":
			// For precision, only consider if recall is at least 10%
			if recall >= 0.1 {
				r.score = precision
			} else {
				r.score = 0 // Ignore thresholds with very low recall
			}
		case "recall":
			r.score = recall
		case "accuracy":
			r.score = accuracy
		case "mcc":
			r.score = mcc
		case "pr_distance":
			// For distance, lower is better, so we negate
			r.score = -distance
		case "precision_at_recall":
			// Find threshold that gives maximum precision at 50% recall
			if recall >= 0.5 {
				r.score = precision
			} else {
				r.score = 0
			}
		default:
			r.score = f1
		}
		
		results[i] = r
	}
	
	// Find best
	best := results[0]
	for _, r := range results[1:] {
		if r.score > best.score {
			best = r
		}
	}
	
	return best.threshold, best.score
}

// fitBetaCalibration fits a beta distribution-based calibration
// This is less aggressive than Platt scaling and preserves more of the original distribution
func fitBetaCalibration(scores, labels []float64) func(float64) float64 {
	// Calculate positive and negative score means
	var posSum, negSum float64
	var posCount, negCount int
	
	for i, score := range scores {
		if labels[i] > 0.5 {
			posSum += score
			posCount++
		} else {
			negSum += score
			negCount++
		}
	}
	
	posMean := posSum / float64(posCount+1)
	negMean := negSum / float64(negCount+1)
	
	// Simple linear interpolation between means
	return func(score float64) float64 {
		// Map negative mean to 0.2 and positive mean to 0.8
		// This preserves more of the original distribution
		if score <= negMean {
			return score / negMean * 0.2
		} else if score >= posMean {
			return 0.8 + (score-posMean)/(1-posMean) * 0.2
		} else {
			// Linear interpolation between negMean and posMean
			t := (score - negMean) / (posMean - negMean)
			return 0.2 + t * 0.6
		}
	}
}

// fitIsotonicRegression fits isotonic regression calibration
// This is a non-parametric method that preserves monotonicity
func fitIsotonicRegression(scores, labels []float64) func(float64) float64 {
	// Sort scores and labels together
	type pair struct {
		score float64
		label float64
	}
	pairs := make([]pair, len(scores))
	for i := range scores {
		pairs[i] = pair{scores[i], labels[i]}
	}
	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].score < pairs[j].score
	})
	
	// Pool adjacent violators algorithm
	n := len(pairs)
	values := make([]float64, n)
	weights := make([]float64, n)
	
	for i := range pairs {
		values[i] = pairs[i].label
		weights[i] = 1.0
	}
	
	// PAV algorithm
	for i := 1; i < n; i++ {
		if values[i] < values[i-1] {
			// Violation found, pool
			poolStart := i - 1
			poolSum := values[poolStart] * weights[poolStart] + values[i] * weights[i]
			poolWeight := weights[poolStart] + weights[i]
			poolValue := poolSum / poolWeight
			
			// Check if we need to pool more
			for poolStart > 0 && values[poolStart-1] > poolValue {
				poolStart--
				poolSum += values[poolStart] * weights[poolStart]
				poolWeight += weights[poolStart]
				poolValue = poolSum / poolWeight
			}
			
			// Apply pooled value
			for j := poolStart; j <= i; j++ {
				values[j] = poolValue
				weights[j] = poolWeight / float64(i-poolStart+1)
			}
		}
	}
	
	// Create interpolation function
	calibScores := make([]float64, n)
	calibValues := make([]float64, n)
	for i := range pairs {
		calibScores[i] = pairs[i].score
		calibValues[i] = values[i]
	}
	
	return func(score float64) float64 {
		// Binary search for interpolation
		idx := sort.SearchFloat64s(calibScores, score)
		
		if idx == 0 {
			return calibValues[0]
		} else if idx == n {
			return calibValues[n-1]
		} else {
			// Linear interpolation
			t := (score - calibScores[idx-1]) / (calibScores[idx] - calibScores[idx-1])
			return calibValues[idx-1] + t*(calibValues[idx]-calibValues[idx-1])
		}
	}
}