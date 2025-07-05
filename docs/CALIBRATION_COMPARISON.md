# Calibration Comparison Feature

This framework now supports comparing different calibration methods to help you choose the best way to calibrate your aggregated model's scores.

## What is Calibration?

When using Naive Bayes multiplication to combine multiple models, the raw scores can become very small. For example:
- 3 models with scores [0.8, 0.8, 0.8] → 0.8³ = 0.512
- 8 models with scores [0.5, 0.5, ...] → 0.5⁸ = 0.0039

These tiny scores make it difficult to use standard thresholds (like 0.5) for classification. Calibration maps these scores back to a meaningful [0,1] range.

## How to Use Calibration Comparison

### 1. Implement the Extended Interface (Optional)

Your model can optionally implement `CalibratedAggregatedModel` to provide both raw and calibrated scores:

```go
type CalibratedAggregatedModel interface {
    AggregatedModel  // Extends the basic interface
    
    // Returns both raw and calibrated predictions
    PredictWithCalibration(samples [][]float64) (raw, calibrated []float64, err error)
    
    // Returns the calibration method name
    GetCalibrationMethod() string
}
```

### 2. Example Implementation

See `models/spam_ensemble/aggregated_model.go` for a complete example:

```go
func (m *SpamAggregatedModel) PredictWithCalibration(samples [][]float64) (raw, calibrated []float64, err error) {
    // Get raw Naive Bayes scores
    raw, err = m.Predict(samples)
    if err != nil {
        return nil, nil, err
    }
    
    // Apply your calibration method
    calibrated = make([]float64, len(raw))
    for i, score := range raw {
        calibrated[i] = m.calibrateScore(score)
    }
    
    return raw, calibrated, nil
}
```

### 3. What the Framework Does

When your model implements `CalibratedAggregatedModel`, the framework automatically:

1. **Tests Multiple Calibration Methods** on your raw scores:
   - **Beta Calibration**: Preserves score distribution, maps class means to [0.2, 0.8]
   - **Isotonic Regression**: Non-parametric method for complex patterns
   - **Platt Scaling**: Sigmoid transformation (can be aggressive)
   - **Min-Max**: Simple normalization to [0,1]

2. **Compares Your Calibration** (if provided) against these methods

3. **Finds Optimal Thresholds** for each calibration method

4. **Provides Comprehensive Metrics**:
   - PR-AUC and ROC-AUC
   - Precision, Recall, F1-Score at optimal threshold
   - Score distributions (min, max, percentiles)

5. **Generates Visualizations**:
   - Box plots of score distributions
   - Performance comparison charts
   - PR curves for each method

## Interpreting Results

The framework will show you:

```
Calibration Comparison Results:
Best method: Beta (pr_auc = 0.8234)

Model's calibration (Beta Calibration):
  Optimal threshold: 0.4821
  Precision: 0.8543
  Recall: 0.7632
  F1-Score: 0.8062

Framework calibration methods:

beta calibration:
  PR-AUC: 0.8234
  Optimal threshold: 0.4985
  Precision: 0.8612
  Recall: 0.7544
  Score range: [0.000123, 0.987654]

isotonic calibration:
  PR-AUC: 0.8156
  Optimal threshold: 0.5123
  Precision: 0.8421
  Recall: 0.7723
  Score range: [0.001234, 0.998765]
```

## Choosing a Calibration Method

- **Beta Calibration**: Best default choice, preserves relative ordering
- **Isotonic Regression**: Good for complex, non-linear patterns
- **Platt Scaling**: Use when you want aggressive probability mapping
- **Your Own Method**: Implement based on domain knowledge

## Benefits

1. **Better Thresholds**: Find meaningful thresholds instead of using arbitrary 0.5
2. **Improved Metrics**: Better precision/recall trade-offs
3. **Visual Understanding**: See how scores are distributed
4. **Informed Decisions**: Choose calibration based on your specific needs

## Without Calibration Comparison

If your model only implements the basic `AggregatedModel` interface, the framework still works normally - it just won't perform calibration comparison.

## Example Report

The HTML report will include:
- Score distribution box plots
- Performance comparison bar charts
- PR curves for each calibration method
- Detailed metrics table

This helps you make an informed decision about which calibration method works best for your specific use case.