package framework

import (
	"fmt"
	"time"
)

// Callback interface for training callbacks
type Callback interface {
	OnTrainBegin(config *Config)
	OnTrainEnd(result *TrainingResult)
	OnEpochBegin(epoch int)
	OnEpochEnd(epoch int, metrics map[string]float64)
	OnBatchBegin(batch int)
	OnBatchEnd(batch int, loss float64)
}

// CallbackList manages multiple callbacks
type CallbackList struct {
	callbacks []Callback
}

// NewCallbackList creates a new callback list
func NewCallbackList(callbacks ...Callback) *CallbackList {
	return &CallbackList{callbacks: callbacks}
}

// Add adds a callback to the list
func (cl *CallbackList) Add(callback Callback) {
	cl.callbacks = append(cl.callbacks, callback)
}

// OnTrainBegin calls OnTrainBegin on all callbacks
func (cl *CallbackList) OnTrainBegin(config *Config) {
	for _, cb := range cl.callbacks {
		cb.OnTrainBegin(config)
	}
}

// OnTrainEnd calls OnTrainEnd on all callbacks
func (cl *CallbackList) OnTrainEnd(result *TrainingResult) {
	for _, cb := range cl.callbacks {
		cb.OnTrainEnd(result)
	}
}

// OnEpochBegin calls OnEpochBegin on all callbacks
func (cl *CallbackList) OnEpochBegin(epoch int) {
	for _, cb := range cl.callbacks {
		cb.OnEpochBegin(epoch)
	}
}

// OnEpochEnd calls OnEpochEnd on all callbacks
func (cl *CallbackList) OnEpochEnd(epoch int, metrics map[string]float64) {
	for _, cb := range cl.callbacks {
		cb.OnEpochEnd(epoch, metrics)
	}
}

// OnBatchBegin calls OnBatchBegin on all callbacks
func (cl *CallbackList) OnBatchBegin(batch int) {
	for _, cb := range cl.callbacks {
		cb.OnBatchBegin(batch)
	}
}

// OnBatchEnd calls OnBatchEnd on all callbacks
func (cl *CallbackList) OnBatchEnd(batch int, loss float64) {
	for _, cb := range cl.callbacks {
		cb.OnBatchEnd(batch, loss)
	}
}

// EarlyStopping implements early stopping callback
type EarlyStopping struct {
	patience     int
	minDelta     float64
	mode         string // "max" or "min"
	monitor      string
	verbose      bool
	bestScore    float64
	counter      int
	stopped      bool
	bestEpoch    int
	bestWeights  []float64
	trainer      *Trainer  // Reference to trainer to restore weights
}

// NewEarlyStopping creates a new early stopping callback
func NewEarlyStopping(config *EarlyStoppingConfig) *EarlyStopping {
	es := &EarlyStopping{
		patience:  config.Patience,
		minDelta:  config.MinDelta,
		mode:      config.Mode,
		monitor:   config.Monitor,
		verbose:   true,
		stopped:   false,
		counter:   0,
		bestEpoch: 0,
	}
	
	// Initialize best score based on mode
	if es.mode == "max" {
		es.bestScore = -1e9
	} else {
		es.bestScore = 1e9
	}
	
	return es
}

// OnTrainBegin implements Callback interface
func (es *EarlyStopping) OnTrainBegin(config *Config) {
	if es.verbose {
		fmt.Printf("Early stopping: monitoring %s (mode=%s, patience=%d)\n", 
			es.monitor, es.mode, es.patience)
	}
}

// OnTrainEnd implements Callback interface
func (es *EarlyStopping) OnTrainEnd(result *TrainingResult) {
	if es.verbose && es.stopped {
		fmt.Printf("Early stopping triggered. Best epoch: %d\n", es.bestEpoch)
	}
}

// OnEpochBegin implements Callback interface
func (es *EarlyStopping) OnEpochBegin(epoch int) {}

// OnEpochEnd implements Callback interface
func (es *EarlyStopping) OnEpochEnd(epoch int, metrics map[string]float64) {
	score, ok := metrics[es.monitor]
	if !ok {
		return
	}
	
	// Check for improvement
	improved := false
	if es.mode == "max" {
		improved = score > es.bestScore+es.minDelta
	} else {
		improved = score < es.bestScore-es.minDelta
	}
	
	if !improved {
		es.counter++
		if es.counter >= es.patience {
			es.stopped = true
			if es.verbose {
				fmt.Printf("Early stopping patience exhausted at epoch %d\n", epoch)
			}
		}
	}
}

// UpdateWeights updates the best weights if the score has improved
func (es *EarlyStopping) UpdateWeights(epoch int, score float64, weights []float64) {
	improved := false
	if es.mode == "max" {
		improved = score > es.bestScore+es.minDelta
	} else {
		improved = score < es.bestScore-es.minDelta
	}
	
	if improved {
		es.bestScore = score
		es.counter = 0
		es.bestEpoch = epoch
		
		// Save best weights - make a copy
		es.bestWeights = make([]float64, len(weights))
		copy(es.bestWeights, weights)
		
		if es.verbose {
			fmt.Printf("Improved %s: %.4f (saving weights)\n", es.monitor, score)
		}
	}
}

// OnBatchBegin implements Callback interface
func (es *EarlyStopping) OnBatchBegin(batch int) {}

// OnBatchEnd implements Callback interface
func (es *EarlyStopping) OnBatchEnd(batch int, loss float64) {}

// ShouldStop returns whether training should stop
func (es *EarlyStopping) ShouldStop() bool {
	return es.stopped
}

// GetBestWeights returns the best weights found
func (es *EarlyStopping) GetBestWeights() []float64 {
	return es.bestWeights
}

// ProgressLogger implements a simple progress logging callback
type ProgressLogger struct {
	logInterval int
	startTime   time.Time
}

// NewProgressLogger creates a new progress logger
func NewProgressLogger(logInterval int) *ProgressLogger {
	return &ProgressLogger{
		logInterval: logInterval,
	}
}

// OnTrainBegin implements Callback interface
func (pl *ProgressLogger) OnTrainBegin(config *Config) {
	pl.startTime = time.Now()
	fmt.Println("Training started...")
}

// OnTrainEnd implements Callback interface
func (pl *ProgressLogger) OnTrainEnd(result *TrainingResult) {
	duration := time.Since(pl.startTime)
	fmt.Printf("Training completed in %v\n", duration)
	fmt.Printf("Final PR-AUC: %.4f\n", result.FinalMetrics["pr_auc"])
}

// OnEpochBegin implements Callback interface
func (pl *ProgressLogger) OnEpochBegin(epoch int) {}

// OnEpochEnd implements Callback interface
func (pl *ProgressLogger) OnEpochEnd(epoch int, metrics map[string]float64) {
	if epoch%pl.logInterval == 0 {
		fmt.Printf("Epoch %d: ", epoch)
		for name, value := range metrics {
			fmt.Printf("%s=%.4f ", name, value)
		}
		fmt.Println()
	}
}

// OnBatchBegin implements Callback interface
func (pl *ProgressLogger) OnBatchBegin(batch int) {}

// OnBatchEnd implements Callback interface
func (pl *ProgressLogger) OnBatchEnd(batch int, loss float64) {}