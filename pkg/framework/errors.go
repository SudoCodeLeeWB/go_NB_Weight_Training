package framework

import "errors"

// Common errors
var (
	// Model errors
	ErrNoModels            = errors.New("no models provided")
	ErrWeightModelMismatch = errors.New("number of weights must match number of models")
	ErrInvalidPrediction   = errors.New("invalid prediction values")
	
	// Data errors
	ErrNoData           = errors.New("no data provided")
	ErrInvalidData      = errors.New("invalid data format")
	ErrMismatchedLength = errors.New("mismatched data lengths")
	ErrNoLabels         = errors.New("no labels provided")
	
	// Training errors
	ErrTrainingFailed = errors.New("training failed")
	ErrNoImprovement  = errors.New("no improvement in metric")
	
	// Configuration errors
	ErrInvalidConfig = errors.New("invalid configuration")
	
	// File errors
	ErrFileNotFound = errors.New("file not found")
	ErrInvalidFormat = errors.New("invalid file format")
)