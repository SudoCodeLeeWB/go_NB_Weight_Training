package framework

// CalibratedAggregatedModel extends AggregatedModel with calibration support.
// This optional interface allows models to provide both raw and calibrated scores,
// enabling the framework to compare different calibration methods.
type CalibratedAggregatedModel interface {
	AggregatedModel
	
	// PredictWithCalibration returns both raw and calibrated scores
	// Input: samples [][]float64 - raw features
	// Output: 
	//   - raw_scores []float64 - raw weighted Naive Bayes scores (can be very small)
	//   - calibrated_scores []float64 - model's own calibrated scores [0,1]
	//   - error
	PredictWithCalibration(samples [][]float64) (raw []float64, calibrated []float64, err error)
	
	// GetCalibrationMethod returns the name/description of calibration method used by the model
	// This helps users understand what calibration the model is already applying
	GetCalibrationMethod() string
}

// CalibrationComparison holds results for a single calibration method
type CalibrationComparison struct {
	Method            string                    // Calibration method name
	OptimalThreshold  float64                   // Best threshold for this calibration
	ThresholdMetric   string                    // Metric used to find threshold
	MetricsAtThreshold map[string]float64       // Metrics at optimal threshold
	ScoreDistribution  ScoreDistribution        // Distribution statistics
	CalibrationFunc    func(float64) float64    // The calibration function (for visualization)
	PRCurve           *CurveData                // PR curve for this calibration
	ROCCurve          *CurveData                // ROC curve for this calibration
}

// ScoreDistribution holds statistics about score distributions
type ScoreDistribution struct {
	Min         float64
	Max         float64
	Mean        float64
	Std         float64
	Percentiles map[int]float64  // 0, 25, 50, 75, 100
	Histogram   []HistogramBin   // For visualization
}

// HistogramBin represents a bin in the score histogram
type HistogramBin struct {
	Start float64
	End   float64
	Count int
	Ratio float64  // Proportion of total samples
}

// CalibrationComparisonResult holds the complete calibration comparison
type CalibrationComparisonResult struct {
	// Raw score distribution (before any calibration)
	RawScoreDistribution ScoreDistribution
	
	// Comparisons for each calibration method tested
	CalibrationComparisons []CalibrationComparison
	
	// Best method based on optimization metric
	BestMethod string
	BestScore  float64
	
	// Model's own calibration (if CalibratedAggregatedModel is implemented)
	ModelProvidedCalibration *CalibrationComparison
	
	// Validation data used for comparison
	ValidationFeatures [][]float64
	ValidationLabels   []float64
}