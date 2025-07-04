package framework

import (
	"fmt"
	"math"
	"time"

	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/metrics"
	"github.com/iwonbin/go-nb-weight-training/pkg/optimizer"
)

// Trainer is the main training orchestrator
type Trainer struct {
	config      *Config
	callbacks   *CallbackList
	optimizer   optimizer.Optimizer
	models      []Model
	earlyStopping *EarlyStopping
}

// TrainingResult holds the results of training
type TrainingResult struct {
	// Final model weights
	BestWeights []float64
	
	// Performance metrics
	FinalMetrics    map[string]float64
	TrainMetrics    map[string]float64
	ValMetrics      map[string]float64
	
	// History
	MetricHistory   map[string][]float64
	WeightHistory   [][]float64
	
	// Cross-validation results
	CVResults       []CVFoldResult
	
	// PR and ROC curves
	PRCurve         *CurveData
	ROCCurve        *CurveData
	
	// Training info
	TotalEpochs     int
	TrainingTime    time.Duration
	Converged       bool
}

// CVFoldResult holds results for a single CV fold
type CVFoldResult struct {
	FoldIndex    int
	TrainMetrics map[string]float64
	ValMetrics   map[string]float64
	Weights      []float64
}

// CurveData holds data for plotting curves
type CurveData struct {
	X      []float64 // Recall for PR, FPR for ROC
	Y      []float64 // Precision for PR, TPR for ROC
	AUC    float64
}

// NewTrainer creates a new trainer
func NewTrainer(config *Config) *Trainer {
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
	
	return &Trainer{
		config:        config,
		callbacks:     callbacks,
		optimizer:     opt,
		earlyStopping: es,
	}
}

// Train trains the ensemble on the given dataset
func (t *Trainer) Train(dataset *data.Dataset, models []Model) (*TrainingResult, error) {
	// Validate dataset
	if err := ValidateDataset(dataset); err != nil {
		return nil, fmt.Errorf("dataset validation failed: %w", err)
	}
	
	// Validate models
	if err := ValidateModels(models); err != nil {
		return nil, fmt.Errorf("model validation failed: %w", err)
	}
	
	t.models = models
	startTime := time.Now()
	
	// Initialize result
	result := &TrainingResult{
		MetricHistory: make(map[string][]float64),
		FinalMetrics:  make(map[string]float64),
		TrainMetrics:  make(map[string]float64),
		ValMetrics:    make(map[string]float64),
		CVResults:     []CVFoldResult{},
	}
	
	// Notify callbacks
	t.callbacks.OnTrainBegin(t.config)
	
	// Perform cross-validation or simple train-val split
	if t.config.DataConfig.KFolds > 1 {
		cvResult, err := t.crossValidate(dataset)
		if err != nil {
			return nil, err
		}
		result.CVResults = cvResult.Folds
		result.BestWeights = cvResult.BestWeights
		result.FinalMetrics = cvResult.AverageMetrics
	} else {
		// Simple train-validation split
		trainResult, err := t.trainSingleSplit(dataset)
		if err != nil {
			return nil, err
		}
		result = trainResult
	}
	
	// Calculate final curves on full dataset if needed
	if result.BestWeights != nil {
		ensemble := &EnsembleModel{
			Models:  t.models,
			Weights: result.BestWeights,
		}
		
		features := dataset.GetFeatures()
		labels := dataset.GetLabels()
		
		predictions, _ := ensemble.Predict(features)
		
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
	
	// Set final info
	result.TrainingTime = time.Since(startTime)
	result.Converged = t.earlyStopping != nil && !t.earlyStopping.stopped
	
	// Notify callbacks
	t.callbacks.OnTrainEnd(result)
	
	// Save results if visualization is enabled
	if t.config.Visualization.Enabled {
		resultWriter, err := NewResultWriter(t.config.Visualization.OutputDir)
		if err != nil {
			return result, fmt.Errorf("failed to create result writer: %w", err)
		}
		
		// Save all results
		if err := resultWriter.SaveTrainingResult(result, t.config); err != nil {
			return result, fmt.Errorf("failed to save results: %w", err)
		}
		
		// Save visualization info
		if err := resultWriter.SaveVisualizationInfo(result, t.config); err != nil {
			return result, fmt.Errorf("failed to save visualization info: %w", err)
		}
		
		fmt.Printf("\nResults saved to: %s\n", resultWriter.GetResultDir())
	}
	
	return result, nil
}

// trainSingleSplit trains on a single train-validation split
func (t *Trainer) trainSingleSplit(dataset *data.Dataset) (*TrainingResult, error) {
	// Create splitter
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
	
	// Split data
	split, err := splitter.Split(dataset)
	if err != nil {
		return nil, fmt.Errorf("failed to split data: %w", err)
	}
	
	// Create objective function
	objectiveFunc := t.createObjectiveFunction(split.Train, split.Test)
	
	// Optimize weights
	optConfig := &optimizer.Config{
		MaxIterations:  t.config.TrainingConfig.MaxEpochs,
		Tolerance:      1e-6,
		RandomSeed:     t.config.DataConfig.RandomSeed,
		MinWeight:      t.config.OptimizerConfig.MinWeight,
		MaxWeight:      t.config.OptimizerConfig.MaxWeight,
		PopulationSize: t.config.OptimizerConfig.PopulationSize,
		MutationFactor: t.config.OptimizerConfig.MutationFactor,
		CrossoverProb:  t.config.OptimizerConfig.CrossoverProb,
		Callback:       t.createOptCallback(),
	}
	
	optResult, err := t.optimizer.Optimize(objectiveFunc, len(t.models), optConfig)
	if err != nil {
		return nil, fmt.Errorf("optimization failed: %w", err)
	}
	
	// Check if early stopping has better weights
	bestWeights := optResult.BestWeights
	if t.earlyStopping != nil && t.earlyStopping.GetBestWeights() != nil {
		bestWeights = t.earlyStopping.GetBestWeights()
		if t.config.TrainingConfig.Verbose {
			fmt.Printf("Restoring best weights from early stopping (epoch %d)\n", t.earlyStopping.bestEpoch)
		}
	}
	
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
	
	// Calculate final metrics
	ensemble := &EnsembleModel{
		Models:  t.models,
		Weights: optResult.BestWeights,
	}
	
	trainMetrics := t.evaluateMetrics(ensemble, split.Train)
	valMetrics := t.evaluateMetrics(ensemble, split.Test)
	
	result.TrainMetrics = trainMetrics
	result.ValMetrics = valMetrics
	result.FinalMetrics = valMetrics
	
	return result, nil
}

// CrossValidationResult holds cross-validation results
type CrossValidationResult struct {
	Folds          []CVFoldResult
	AverageMetrics map[string]float64
	BestWeights    []float64
	BestFoldIndex  int
}

// crossValidate performs k-fold cross-validation
func (t *Trainer) crossValidate(dataset *data.Dataset) (*CrossValidationResult, error) {
	// Create cross-validator
	var cv data.CrossValidator
	if t.config.DataConfig.Stratified {
		cv = data.NewStratifiedKFoldCV(
			t.config.DataConfig.KFolds,
			true,
			t.config.DataConfig.RandomSeed,
		)
	} else {
		cv = data.NewKFoldCV(
			t.config.DataConfig.KFolds,
			true,
			t.config.DataConfig.RandomSeed,
		)
	}
	
	// Get folds
	folds, err := cv.GetFolds(dataset)
	if err != nil {
		return nil, fmt.Errorf("failed to create folds: %w", err)
	}
	
	// Train on each fold
	cvResults := make([]CVFoldResult, len(folds))
	bestScore := -1e9
	bestFoldIndex := 0
	
	for i, fold := range folds {
		fmt.Printf("\nTraining fold %d/%d\n", i+1, len(folds))
		
		trainData := dataset.Subset(fold.TrainIndices)
		valData := dataset.Subset(fold.TestIndices)
		
		// Create objective function for this fold
		objectiveFunc := t.createObjectiveFunction(trainData, valData)
		
		// Optimize
		optConfig := &optimizer.Config{
			MaxIterations:  t.config.TrainingConfig.MaxEpochs,
			Tolerance:      1e-6,
			RandomSeed:     t.config.DataConfig.RandomSeed + int64(i),
			MinWeight:      t.config.OptimizerConfig.MinWeight,
			MaxWeight:      t.config.OptimizerConfig.MaxWeight,
			PopulationSize: t.config.OptimizerConfig.PopulationSize,
			MutationFactor: t.config.OptimizerConfig.MutationFactor,
			CrossoverProb:  t.config.OptimizerConfig.CrossoverProb,
		}
		
		optResult, err := t.optimizer.Optimize(objectiveFunc, len(t.models), optConfig)
		if err != nil {
			return nil, fmt.Errorf("optimization failed for fold %d: %w", i, err)
		}
		
		// Evaluate metrics
		ensemble := &EnsembleModel{
			Models:  t.models,
			Weights: optResult.BestWeights,
		}
		
		trainMetrics := t.evaluateMetrics(ensemble, trainData)
		valMetrics := t.evaluateMetrics(ensemble, valData)
		
		cvResults[i] = CVFoldResult{
			FoldIndex:    i,
			TrainMetrics: trainMetrics,
			ValMetrics:   valMetrics,
			Weights:      optResult.BestWeights,
		}
		
		// Track best fold
		score := valMetrics[t.config.TrainingConfig.OptimizationMetric]
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

// createObjectiveFunction creates the objective function for optimization
func (t *Trainer) createObjectiveFunction(trainData, valData *data.Dataset) optimizer.ObjectiveFunc {
	return func(weights []float64) float64 {
		// Create ensemble with current weights
		ensemble := &EnsembleModel{
			Models:  t.models,
			Weights: weights,
		}
		
		// Get predictions on validation set
		features := valData.GetFeatures()
		labels := valData.GetLabels()
		
		predictions, err := ensemble.Predict(features)
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

// createOptCallback creates optimizer callback with batch processing
func (t *Trainer) createOptCallback() optimizer.ProgressCallback {
	var lastBestScore float64
	batchLosses := []float64{}
	
	return func(iteration int, bestScore float64, bestWeights []float64) {
		// Calculate batch loss (improvement)
		batchLoss := 0.0
		if iteration > 0 {
			batchLoss = math.Abs(bestScore - lastBestScore)
		}
		lastBestScore = bestScore
		batchLosses = append(batchLosses, batchLoss)
		
		// Calculate moving average of losses
		avgLoss := 0.0
		window := 5
		start := len(batchLosses) - window
		if start < 0 {
			start = 0
		}
		for i := start; i < len(batchLosses); i++ {
			avgLoss += batchLosses[i]
		}
		if len(batchLosses[start:]) > 0 {
			avgLoss /= float64(len(batchLosses[start:]))
		}
		
		metrics := map[string]float64{
			t.config.TrainingConfig.OptimizationMetric: bestScore,
			"batch_loss": batchLoss,
			"avg_loss": avgLoss,
		}
		
		// Pass weights to callbacks (for early stopping)
		// We need to handle this separately since metrics is map[string]float64
		if t.earlyStopping != nil && bestWeights != nil {
			// Update early stopping directly with current weights if score improved
			t.earlyStopping.UpdateWeights(iteration, bestScore, bestWeights)
		}
		
		// Log batch progress
		if t.config.TrainingConfig.Verbose && iteration%t.config.TrainingConfig.BatchSize == 0 {
			batchNum := iteration / t.config.TrainingConfig.BatchSize
			fmt.Printf("Batch %d: score=%.4f, loss=%.6f, avg_loss=%.6f\n", 
				batchNum, bestScore, batchLoss, avgLoss)
		}
		
		t.callbacks.OnEpochEnd(iteration, metrics)
	}
}

// evaluateMetrics evaluates all metrics on a dataset
func (t *Trainer) evaluateMetrics(model Model, dataset *data.Dataset) map[string]float64 {
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

// averageMetrics calculates average metrics across CV folds
func (t *Trainer) averageMetrics(folds []CVFoldResult) map[string]float64 {
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

// TrainFromCSV is a convenience method to train from a CSV file
func (t *Trainer) TrainFromCSV(csvPath string, models []Model) (*TrainingResult, error) {
	// Load data
	loader := data.NewCSVLoader()
	dataset, err := loader.Load(csvPath)
	if err != nil {
		return nil, fmt.Errorf("failed to load data: %w", err)
	}
	
	// Train
	return t.Train(dataset, models)
}

