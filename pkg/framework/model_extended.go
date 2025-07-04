package framework

// ExtendedModel adds single prediction capability to the Model interface
type ExtendedModel interface {
	Model
	// PredictSingle predicts for a single sample
	PredictSingle(sample []float64) (float64, error)
}

// SinglePredictAdapter wraps a Model to provide single prediction capability
type SinglePredictAdapter struct {
	Model
}

// PredictSingle implements single prediction by calling batch predict
func (spa *SinglePredictAdapter) PredictSingle(sample []float64) (float64, error) {
	// Create a batch with single sample
	batch := [][]float64{sample}
	
	// Get predictions
	predictions, err := spa.Model.Predict(batch)
	if err != nil {
		return 0, err
	}
	
	if len(predictions) == 0 {
		return 0, ErrInvalidPrediction
	}
	
	return predictions[0], nil
}

// BatchPredictAdapter provides efficient batch prediction for models that only implement PredictSingle
type BatchPredictAdapter struct {
	ExtendedModel
}

// Predict implements batch prediction using single predictions
func (bpa *BatchPredictAdapter) Predict(samples [][]float64) ([]float64, error) {
	predictions := make([]float64, len(samples))
	
	for i, sample := range samples {
		pred, err := bpa.ExtendedModel.PredictSingle(sample)
		if err != nil {
			return nil, err
		}
		predictions[i] = pred
	}
	
	return predictions, nil
}

// ToExtended converts a Model to ExtendedModel
func ToExtended(model Model) ExtendedModel {
	// Check if already extended
	if em, ok := model.(ExtendedModel); ok {
		return em
	}
	
	// Wrap with adapter
	return &SinglePredictAdapter{Model: model}
}

// ExtendedEnsembleModel extends EnsembleModel with single prediction
type ExtendedEnsembleModel struct {
	*EnsembleModel
}

// PredictSingle predicts for a single sample
func (eem *ExtendedEnsembleModel) PredictSingle(sample []float64) (float64, error) {
	// Create batch with single sample
	batch := [][]float64{sample}
	
	// Use batch prediction
	predictions, err := eem.Predict(batch)
	if err != nil {
		return 0, err
	}
	
	if len(predictions) == 0 {
		return 0, ErrInvalidPrediction
	}
	
	return predictions[0], nil
}

// NewExtendedEnsemble creates an extended ensemble model
func NewExtendedEnsemble(models []Model, weights []float64) *ExtendedEnsembleModel {
	return &ExtendedEnsembleModel{
		EnsembleModel: &EnsembleModel{
			Models:  models,
			Weights: weights,
		},
	}
}