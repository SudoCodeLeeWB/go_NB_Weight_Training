package framework

import (
	"fmt"
	"time"

	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/metrics"
	"github.com/iwonbin/go-nb-weight-training/pkg/optimizer"
)

// TrainAggregatedModel optimizes the weights of an AggregatedModel
// This is the main entry point for users who have implemented the AggregatedModel interface
func TrainAggregatedModel(dataset *data.Dataset, model AggregatedModel, config *Config) (*TrainingResult, error) {
	trainer := NewTrainerForAggregated(config)
	return trainer.Train(dataset, model)
}

// TrainerForAggregated is a trainer specifically for AggregatedModel
// It treats the model as a complete black box and only optimizes weights
type TrainerForAggregated struct {
	config        *Config
	callbacks     *CallbackList
	optimizer     optimizer.Optimizer
	earlyStopping *EarlyStopping
}

// NewTrainerForAggregated creates a new trainer for AggregatedModel
func NewTrainerForAggregated(config *Config) *TrainerForAggregated {
	if config == nil {
		config = DefaultConfig()
	}
	
	// Validate config
	if err := config.Validate(); err != nil {
		panic(fmt.Sprintf("Invalid config: %v", err))
	}
	
	// Create callbacks
	callbacks := NewCallbackList()
	
	// Add progress logger if verbose
	if config.TrainingConfig.Verbose {
		callbacks.Add(NewProgressLogger(config.TrainingConfig.LogInterval))
	}
	
	// Create early stopping if configured
	var es *EarlyStopping
	if config.EarlyStopping != nil {
		es = NewEarlyStopping(config.EarlyStopping)
		callbacks.Add(es)
	}
	
	// Create optimizer
	var opt optimizer.Optimizer
	switch config.OptimizerConfig.Type {
	case "differential_evolution":
		opt = optimizer.NewDifferentialEvolution()
	case "random_search":
		opt = optimizer.NewRandomSearch()
	default:
		opt = optimizer.NewDifferentialEvolution()
	}
	
	return &TrainerForAggregated{
		config:        config,
		callbacks:     callbacks,
		optimizer:     opt,
		earlyStopping: es,
	}
}

// Train optimizes the weights of an AggregatedModel
func (t *TrainerForAggregated) Train(dataset *data.Dataset, model AggregatedModel) (*TrainingResult, error) {
	// Validate dataset
	if err := ValidateDataset(dataset); err != nil {
		return nil, fmt.Errorf("dataset validation failed: %w", err)
	}
	
	// Validate model
	if model.GetNumModels() == 0 {
		return nil, fmt.Errorf("aggregated model has no models to optimize")
	}
	
	startTime := time.Now()
	
	// Initialize result
	result := &TrainingResult{
		MetricHistory: make(map[string][]float64),
		FinalMetrics:  make(map[string]float64),
		TrainMetrics:  make(map[string]float64),
		ValMetrics:    make(map[string]float64),
		CVResults:     []CVFoldResult{},
	}
	
	// Store original weights to restore if needed
	originalWeights := model.GetWeights()
	
	// Notify callbacks
	t.callbacks.OnTrainBegin(t.config)
	
	// Perform training based on configuration
	if t.config.DataConfig.KFolds > 1 {
		// Cross-validation
		cvResult, err := t.crossValidate(dataset, model)
		if err != nil {
			return nil, err
		}
		result.CVResults = cvResult.Folds
		result.BestWeights = cvResult.BestWeights
		result.FinalMetrics = cvResult.AverageMetrics
	} else {
		// Simple train-validation split
		trainResult, err := t.trainSingleSplit(dataset, model)
		if err != nil {
			return nil, err
		}
		result = trainResult
	}
	
	// Apply best weights to model
	if err := model.SetWeights(result.BestWeights); err != nil {
		// If setting weights fails, restore original
		model.SetWeights(originalWeights)
		return nil, fmt.Errorf("failed to set best weights: %w", err)
	}
	
	// Calculate final performance curves
	if result.BestWeights != nil {
		t.calculateFinalCurves(dataset, model, result)
	}
	
	// Set final info
	result.TrainingTime = time.Since(startTime)
	result.Converged = t.earlyStopping != nil && !t.earlyStopping.stopped
	
	// Notify callbacks
	t.callbacks.OnTrainEnd(result)
	
	// Save results if visualization is enabled
	if t.config.Visualization.Enabled {
		if err := t.saveResults(result); err != nil {
			return result, err // Return result even if saving fails
		}
	}
	
	return result, nil
}

// trainSingleSplit trains on a single train-validation split
func (t *TrainerForAggregated) trainSingleSplit(dataset *data.Dataset, model AggregatedModel) (*TrainingResult, error) {
	// Split data
	split, err := t.createDataSplit(dataset)
	if err != nil {
		return nil, err
	}
	
	// Create objective function
	objectiveFunc := t.createObjectiveFunction(split.Train, split.Test, model)
	
	// Configure optimizer
	numWeights := model.GetNumModels()
	optConfig := t.createOptimizerConfig(numWeights)
	optConfig.Callback = t.createOptCallback(model)
	
	// Optimize weights
	optResult, err := t.optimizer.Optimize(objectiveFunc, numWeights, optConfig)
	if err != nil {
		return nil, fmt.Errorf("optimization failed: %w", err)
	}
	
	// Get best weights (from early stopping if applicable)
	bestWeights := t.getBestWeights(optResult)
	
	// Create result
	result := &TrainingResult{
		BestWeights:   bestWeights,
		TotalEpochs:   optResult.Iterations,
		Converged:     optResult.Converged,
		WeightHistory: optResult.WeightHistory,
		MetricHistory: map[string][]float64{
			t.config.TrainingConfig.OptimizationMetric: optResult.ScoreHistory,
		},
	}
	
	// Calculate final metrics with best weights
	model.SetWeights(bestWeights)
	result.TrainMetrics = t.evaluateMetrics(model, split.Train)
	result.ValMetrics = t.evaluateMetrics(model, split.Test)
	result.FinalMetrics = result.ValMetrics
	
	return result, nil
}

// crossValidate performs k-fold cross-validation
func (t *TrainerForAggregated) crossValidate(dataset *data.Dataset, model AggregatedModel) (*CrossValidationResult, error) {
	// Create cross-validator
	cv := t.createCrossValidator()
	
	// Get folds
	folds, err := cv.GetFolds(dataset)
	if err != nil {
		return nil, fmt.Errorf("failed to create folds: %w", err)
	}
	
	// Train on each fold
	cvResults := make([]CVFoldResult, len(folds))
	bestScore := -1e9
	bestFoldIndex := 0
	
	// Save original weights
	originalWeights := model.GetWeights()
	
	for i, fold := range folds {
		fmt.Printf("\nTraining fold %d/%d\n", i+1, len(folds))
		
		// Reset weights for each fold
		model.SetWeights(originalWeights)
		
		// Get train/val data for this fold
		trainData := dataset.Subset(fold.TrainIndices)
		valData := dataset.Subset(fold.TestIndices)
		
		// Train on this fold
		foldResult, err := t.trainFold(trainData, valData, model, i)
		if err != nil {
			return nil, fmt.Errorf("failed to train fold %d: %w", i, err)
		}
		
		cvResults[i] = foldResult
		
		// Track best fold
		score := foldResult.ValMetrics[t.config.TrainingConfig.OptimizationMetric]
		if score > bestScore {
			bestScore = score
			bestFoldIndex = i
		}
	}
	
	// Calculate average metrics
	avgMetrics := t.averageMetrics(cvResults)
	
	return &CrossValidationResult{
		Folds:          cvResults,
		AverageMetrics: avgMetrics,
		BestWeights:    cvResults[bestFoldIndex].Weights,
		BestFoldIndex:  bestFoldIndex,
	}, nil
}

// Helper methods

func (t *TrainerForAggregated) createDataSplit(dataset *data.Dataset) (*data.Split, error) {
	var splitter data.Splitter
	if t.config.DataConfig.Stratified {
		splitter = data.NewStratifiedSplitter(
			t.config.DataConfig.ValidationSplit,
			t.config.DataConfig.RandomSeed,
		)
	} else {
		splitter = data.NewRandomSplitter(
			t.config.DataConfig.ValidationSplit,
			t.config.DataConfig.RandomSeed,
		)
	}
	return splitter.Split(dataset)
}

func (t *TrainerForAggregated) createCrossValidator() data.CrossValidator {
	if t.config.DataConfig.Stratified {
		return data.NewStratifiedKFoldCV(
			t.config.DataConfig.KFolds,
			true,
			t.config.DataConfig.RandomSeed,
		)
	}
	return data.NewKFoldCV(
		t.config.DataConfig.KFolds,
		true,
		t.config.DataConfig.RandomSeed,
	)
}

func (t *TrainerForAggregated) createOptimizerConfig(numWeights int) *optimizer.Config {
	return &optimizer.Config{
		MaxIterations:  t.config.TrainingConfig.MaxEpochs,
		Tolerance:      1e-6,
		RandomSeed:     t.config.DataConfig.RandomSeed,
		MinWeight:      t.config.OptimizerConfig.MinWeight,
		MaxWeight:      t.config.OptimizerConfig.MaxWeight,
		PopulationSize: t.config.OptimizerConfig.PopulationSize,
		MutationFactor: t.config.OptimizerConfig.MutationFactor,
		CrossoverProb:  t.config.OptimizerConfig.CrossoverProb,
		EnforceNonZero: t.config.OptimizerConfig.EnforceNonZero,
	}
}

func (t *TrainerForAggregated) createObjectiveFunction(trainData, valData *data.Dataset, model AggregatedModel) optimizer.ObjectiveFunc {
	return func(weights []float64) float64 {
		// Set weights on the model
		if err := model.SetWeights(weights); err != nil {
			return 0.0
		}
		
		// Get predictions on validation set
		features := valData.GetFeatures()
		labels := valData.GetLabels()
		
		predictions, err := model.Predict(features)
		if err != nil {
			return 0.0
		}
		
		// Calculate metric
		var metric metrics.Metrics
		switch t.config.TrainingConfig.OptimizationMetric {
		case "pr_auc":
			metric = &metrics.PRAUC{}
		case "roc_auc":
			metric = &metrics.ROCAUC{}
		default:
			metric = &metrics.PRAUC{}
		}
		
		score, err := metric.Calculate(predictions, labels)
		if err != nil {
			return 0.0
		}
		
		return score
	}
}

func (t *TrainerForAggregated) createOptCallback(model AggregatedModel) optimizer.ProgressCallback {
	return func(iteration int, bestScore float64, bestWeights []float64) {
		metrics := map[string]float64{
			t.config.TrainingConfig.OptimizationMetric: bestScore,
		}
		
		// Update early stopping
		if t.earlyStopping != nil && bestWeights != nil {
			t.earlyStopping.UpdateWeights(iteration, bestScore, bestWeights)
		}
		
		// Log progress
		if t.config.TrainingConfig.Verbose && iteration%t.config.TrainingConfig.LogInterval == 0 {
			fmt.Printf("Epoch %d: %s=%.4f\n", 
				iteration, t.config.TrainingConfig.OptimizationMetric, bestScore)
		}
		
		t.callbacks.OnEpochEnd(iteration, metrics)
	}
}

func (t *TrainerForAggregated) getBestWeights(optResult *optimizer.Result) []float64 {
	bestWeights := optResult.BestWeights
	if t.earlyStopping != nil && t.earlyStopping.GetBestWeights() != nil {
		bestWeights = t.earlyStopping.GetBestWeights()
		if t.config.TrainingConfig.Verbose {
			fmt.Printf("Restoring best weights from early stopping (epoch %d)\n", t.earlyStopping.bestEpoch)
		}
	}
	return bestWeights
}

func (t *TrainerForAggregated) evaluateMetrics(model AggregatedModel, dataset *data.Dataset) map[string]float64 {
	features := dataset.GetFeatures()
	labels := dataset.GetLabels()
	
	predictions, _ := model.Predict(features)
	
	// Calculate various metrics
	prAUC, _ := (&metrics.PRAUC{}).Calculate(predictions, labels)
	rocAUC, _ := (&metrics.ROCAUC{}).Calculate(predictions, labels)
	
	// Calculate metrics at threshold 0.5
	cm := metrics.CalculateConfusionMatrix(predictions, labels, 0.5)
	
	return map[string]float64{
		"pr_auc":    prAUC,
		"roc_auc":   rocAUC,
		"precision": cm.Precision(),
		"recall":    cm.Recall(),
		"f1_score":  cm.F1Score(),
	}
}

func (t *TrainerForAggregated) trainFold(trainData, valData *data.Dataset, model AggregatedModel, foldIdx int) (CVFoldResult, error) {
	// Create objective function for this fold
	objectiveFunc := t.createObjectiveFunction(trainData, valData, model)
	
	// Configure optimizer
	numWeights := model.GetNumModels()
	optConfig := t.createOptimizerConfig(numWeights)
	optConfig.RandomSeed += int64(foldIdx) // Different seed for each fold
	
	// Optimize
	optResult, err := t.optimizer.Optimize(objectiveFunc, numWeights, optConfig)
	if err != nil {
		return CVFoldResult{}, err
	}
	
	// Evaluate with best weights
	model.SetWeights(optResult.BestWeights)
	
	return CVFoldResult{
		FoldIndex:    foldIdx,
		TrainMetrics: t.evaluateMetrics(model, trainData),
		ValMetrics:   t.evaluateMetrics(model, valData),
		Weights:      optResult.BestWeights,
	}, nil
}

func (t *TrainerForAggregated) averageMetrics(folds []CVFoldResult) map[string]float64 {
	if len(folds) == 0 {
		return map[string]float64{}
	}
	
	// Initialize sums
	sums := make(map[string]float64)
	for name := range folds[0].ValMetrics {
		sums[name] = 0.0
	}
	
	// Sum metrics
	for _, fold := range folds {
		for name, value := range fold.ValMetrics {
			sums[name] += value
		}
	}
	
	// Calculate averages
	avg := make(map[string]float64)
	for name, sum := range sums {
		avg[name] = sum / float64(len(folds))
	}
	
	return avg
}

func (t *TrainerForAggregated) calculateFinalCurves(dataset *data.Dataset, model AggregatedModel, result *TrainingResult) {
	features := dataset.GetFeatures()
	labels := dataset.GetLabels()
	
	predictions, _ := model.Predict(features)
	
	// Calculate PR curve
	precisions, recalls := metrics.CalculatePRCurve(predictions, labels)
	prAUC := metrics.AveragePrecision(predictions, labels)
	result.PRCurve = &CurveData{
		X:   recalls,
		Y:   precisions,
		AUC: prAUC,
	}
	
	// Calculate ROC curve
	fprs, tprs := metrics.CalculateROCCurve(predictions, labels)
	rocAUC, _ := (&metrics.ROCAUC{}).Calculate(predictions, labels)
	result.ROCCurve = &CurveData{
		X:   fprs,
		Y:   tprs,
		AUC: rocAUC,
	}
}

func (t *TrainerForAggregated) saveResults(result *TrainingResult) error {
	// Always use ./output directory
	outputDir := "./output"
	resultWriter, err := NewResultWriter(outputDir)
	if err != nil {
		return fmt.Errorf("failed to create result writer: %w", err)
	}
	
	// Save all results
	if err := resultWriter.SaveTrainingResult(result, t.config); err != nil {
		return fmt.Errorf("failed to save results: %w", err)
	}
	
	// Save visualization info
	if err := resultWriter.SaveVisualizationInfo(result, t.config); err != nil {
		return fmt.Errorf("failed to save visualization info: %w", err)
	}
	
	// Store result directory for later report generation
	result.OutputDir = resultWriter.GetResultDir()
	
	fmt.Printf("\nResults saved to: %s\n", resultWriter.GetResultDir())
	return nil
}