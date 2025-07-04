package framework

import (
	"fmt"
	"math"
	"runtime"
	"sync"
)

// BatchProcessor handles efficient batch predictions
type BatchProcessor struct {
	Model       Model
	BatchSize   int
	NumWorkers  int
	UseParallel bool
}

// NewBatchProcessor creates an optimized batch processor
func NewBatchProcessor(model Model, batchSize int) *BatchProcessor {
	numWorkers := runtime.NumCPU()
	if numWorkers > 4 {
		numWorkers = 4 // Limit to avoid overhead
	}
	
	return &BatchProcessor{
		Model:       model,
		BatchSize:   batchSize,
		NumWorkers:  numWorkers,
		UseParallel: true,
	}
}

// PredictBatch processes large datasets efficiently
func (bp *BatchProcessor) PredictBatch(samples [][]float64) ([]float64, error) {
	if !bp.UseParallel || len(samples) < bp.BatchSize*2 {
		// For small datasets, use sequential processing
		return bp.Model.Predict(samples)
	}
	
	// Parallel batch processing
	results := make([]float64, len(samples))
	errors := make([]error, bp.NumWorkers)
	
	// Create work channel
	type workItem struct {
		start int
		end   int
	}
	work := make(chan workItem, bp.NumWorkers*2)
	
	// Worker function
	var wg sync.WaitGroup
	for w := 0; w < bp.NumWorkers; w++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			
			for item := range work {
				batch := samples[item.start:item.end]
				predictions, err := bp.Model.Predict(batch)
				
				if err != nil {
					errors[workerID] = err
					return
				}
				
				// Copy results
				copy(results[item.start:item.end], predictions)
			}
		}(w)
	}
	
	// Submit work
	for i := 0; i < len(samples); i += bp.BatchSize {
		end := i + bp.BatchSize
		if end > len(samples) {
			end = len(samples)
		}
		work <- workItem{start: i, end: end}
	}
	close(work)
	
	// Wait for completion
	wg.Wait()
	
	// Check for errors
	for _, err := range errors {
		if err != nil {
			return nil, err
		}
	}
	
	return results, nil
}

// StreamPredict handles streaming predictions for large datasets
type StreamPredictor struct {
	Model     Model
	BatchSize int
}

// PredictStream processes samples as they come without loading all into memory
func (sp *StreamPredictor) PredictStream(
	sampleChan <-chan []float64,
	resultChan chan<- PredictionResult,
) {
	batch := make([][]float64, 0, sp.BatchSize)
	indices := make([]int, 0, sp.BatchSize)
	index := 0
	
	for sample := range sampleChan {
		batch = append(batch, sample)
		indices = append(indices, index)
		index++
		
		if len(batch) >= sp.BatchSize {
			sp.processBatch(batch, indices, resultChan)
			batch = batch[:0]
			indices = indices[:0]
		}
	}
	
	// Process remaining samples
	if len(batch) > 0 {
		sp.processBatch(batch, indices, resultChan)
	}
	
	close(resultChan)
}

func (sp *StreamPredictor) processBatch(
	batch [][]float64,
	indices []int,
	resultChan chan<- PredictionResult,
) {
	predictions, err := sp.Model.Predict(batch)
	
	for i, pred := range predictions {
		resultChan <- PredictionResult{
			Index:      indices[i],
			Prediction: pred,
			Error:      err,
		}
	}
}

// PredictionResult holds streaming prediction result
type PredictionResult struct {
	Index      int
	Prediction float64
	Error      error
}

// MemoryEfficientEnsemble handles large-scale ensemble predictions
type MemoryEfficientEnsemble struct {
	Models      []Model
	Weights     []float64
	BatchSize   int
	UseLogSpace bool
}

// Predict processes in chunks to avoid memory issues
func (mee *MemoryEfficientEnsemble) Predict(samples [][]float64) ([]float64, error) {
	n := len(samples)
	results := make([]float64, n)
	
	// Process in chunks
	for start := 0; start < n; start += mee.BatchSize {
		end := start + mee.BatchSize
		if end > n {
			end = n
		}
		
		chunk := samples[start:end]
		chunkSize := end - start
		
		if mee.UseLogSpace {
			// Initialize log scores
			for i := 0; i < chunkSize; i++ {
				results[start+i] = 0.0
			}
			
			// Process each model
			for modelIdx, model := range mee.Models {
				predictions, err := model.Predict(chunk)
				if err != nil {
					return nil, fmt.Errorf("model %d prediction failed: %w", modelIdx, err)
				}
				
				// Add weighted log probabilities
				weight := mee.Weights[modelIdx]
				for i, pred := range predictions {
					if pred <= 0 {
						pred = 1e-10
					} else if pred >= 1 {
						pred = 1 - 1e-10
					}
					results[start+i] += weight * math.Log(pred)
				}
			}
			
			// Convert back from log space if needed
			for i := 0; i < chunkSize; i++ {
				results[start+i] = math.Exp(results[start+i])
			}
		} else {
			// Regular multiplication
			for i := 0; i < chunkSize; i++ {
				results[start+i] = 1.0
			}
			
			for modelIdx, model := range mee.Models {
				predictions, err := model.Predict(chunk)
				if err != nil {
					return nil, fmt.Errorf("model %d prediction failed: %w", modelIdx, err)
				}
				
				weight := mee.Weights[modelIdx]
				for i, pred := range predictions {
					results[start+i] *= math.Pow(pred, weight)
				}
			}
		}
	}
	
	return results, nil
}

func (mee *MemoryEfficientEnsemble) GetName() string {
	return "MemoryEfficientEnsemble"
}