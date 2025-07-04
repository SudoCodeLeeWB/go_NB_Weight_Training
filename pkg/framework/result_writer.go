package framework

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"
)

// ResultWriter handles saving training results with timestamp-based directories
type ResultWriter struct {
	baseDir    string
	resultDir  string
	timestamp  string
}

// NewResultWriter creates a new result writer with timestamp directory
func NewResultWriter(baseDir string) (*ResultWriter, error) {
	// Create timestamp
	timestamp := time.Now().Format("2006-01-02_15-04-05")
	
	// Create result directory path
	resultDir := filepath.Join(baseDir, fmt.Sprintf("results_%s", timestamp))
	
	// Create directories
	if err := os.MkdirAll(resultDir, 0755); err != nil {
		return nil, fmt.Errorf("failed to create result directory: %w", err)
	}
	
	return &ResultWriter{
		baseDir:   baseDir,
		resultDir: resultDir,
		timestamp: timestamp,
	}, nil
}

// GetResultDir returns the current result directory
func (rw *ResultWriter) GetResultDir() string {
	return rw.resultDir
}

// SaveTrainingResult saves the complete training result
func (rw *ResultWriter) SaveTrainingResult(result *TrainingResult, config *Config) error {
	// Save configuration
	configPath := filepath.Join(rw.resultDir, "config.json")
	if err := config.SaveConfig(configPath); err != nil {
		return fmt.Errorf("failed to save config: %w", err)
	}
	
	// Save training results
	resultPath := filepath.Join(rw.resultDir, "training_result.json")
	if err := rw.saveResultJSON(result, resultPath); err != nil {
		return fmt.Errorf("failed to save results: %w", err)
	}
	
	// Save weights separately for easy access
	weightsPath := filepath.Join(rw.resultDir, "best_weights.json")
	if err := rw.saveWeights(result.BestWeights, weightsPath); err != nil {
		return fmt.Errorf("failed to save weights: %w", err)
	}
	
	// Create summary file
	summaryPath := filepath.Join(rw.resultDir, "summary.txt")
	if err := rw.saveSummary(result, config, summaryPath); err != nil {
		return fmt.Errorf("failed to save summary: %w", err)
	}
	
	return nil
}

// saveResultJSON saves the training result as JSON
func (rw *ResultWriter) saveResultJSON(result *TrainingResult, path string) error {
	// Create a simplified version for JSON serialization
	jsonResult := map[string]interface{}{
		"timestamp":       rw.timestamp,
		"best_weights":    result.BestWeights,
		"final_metrics":   result.FinalMetrics,
		"train_metrics":   result.TrainMetrics,
		"val_metrics":     result.ValMetrics,
		"total_epochs":    result.TotalEpochs,
		"training_time":   result.TrainingTime.String(),
		"converged":       result.Converged,
		"metric_history":  result.MetricHistory,
		"weight_history":  result.WeightHistory,
	}
	
	// Add CV results if available
	if len(result.CVResults) > 0 {
		cvResults := make([]map[string]interface{}, len(result.CVResults))
		for i, fold := range result.CVResults {
			cvResults[i] = map[string]interface{}{
				"fold_index":    fold.FoldIndex,
				"train_metrics": fold.TrainMetrics,
				"val_metrics":   fold.ValMetrics,
				"weights":       fold.Weights,
			}
		}
		jsonResult["cv_results"] = cvResults
	}
	
	// Add curve data if available
	if result.PRCurve != nil {
		jsonResult["pr_auc"] = result.PRCurve.AUC
	}
	if result.ROCCurve != nil {
		jsonResult["roc_auc"] = result.ROCCurve.AUC
	}
	
	// Write JSON
	data, err := json.MarshalIndent(jsonResult, "", "  ")
	if err != nil {
		return err
	}
	
	return os.WriteFile(path, data, 0644)
}

// saveWeights saves just the weights for easy loading
func (rw *ResultWriter) saveWeights(weights []float64, path string) error {
	data, err := json.MarshalIndent(weights, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0644)
}

// saveSummary creates a human-readable summary
func (rw *ResultWriter) saveSummary(result *TrainingResult, config *Config, path string) error {
	summary := fmt.Sprintf("Training Summary\n")
	summary += fmt.Sprintf("================\n\n")
	summary += fmt.Sprintf("Timestamp: %s\n", rw.timestamp)
	summary += fmt.Sprintf("Training Time: %v\n", result.TrainingTime)
	summary += fmt.Sprintf("Total Epochs: %d\n", result.TotalEpochs)
	summary += fmt.Sprintf("Converged: %v\n\n", result.Converged)
	
	summary += fmt.Sprintf("Configuration:\n")
	summary += fmt.Sprintf("--------------\n")
	summary += fmt.Sprintf("Optimization Metric: %s\n", config.TrainingConfig.OptimizationMetric)
	summary += fmt.Sprintf("Max Epochs: %d\n", config.TrainingConfig.MaxEpochs)
	summary += fmt.Sprintf("K-Folds: %d\n", config.DataConfig.KFolds)
	if config.EarlyStopping != nil {
		summary += fmt.Sprintf("Early Stopping Patience: %d\n", config.EarlyStopping.Patience)
	}
	summary += fmt.Sprintf("\n")
	
	summary += fmt.Sprintf("Best Weights:\n")
	summary += fmt.Sprintf("-------------\n")
	for i, w := range result.BestWeights {
		summary += fmt.Sprintf("Model %d: %.4f\n", i, w)
	}
	summary += fmt.Sprintf("\n")
	
	summary += fmt.Sprintf("Final Metrics:\n")
	summary += fmt.Sprintf("--------------\n")
	for metric, value := range result.FinalMetrics {
		summary += fmt.Sprintf("%s: %.4f\n", metric, value)
	}
	
	if len(result.TrainMetrics) > 0 {
		summary += fmt.Sprintf("\nTraining Metrics:\n")
		summary += fmt.Sprintf("-----------------\n")
		for metric, value := range result.TrainMetrics {
			summary += fmt.Sprintf("%s: %.4f\n", metric, value)
		}
	}
	
	if len(result.ValMetrics) > 0 {
		summary += fmt.Sprintf("\nValidation Metrics:\n")
		summary += fmt.Sprintf("-------------------\n")
		for metric, value := range result.ValMetrics {
			summary += fmt.Sprintf("%s: %.4f\n", metric, value)
		}
	}
	
	return os.WriteFile(path, []byte(summary), 0644)
}

// SaveVisualizationInfo saves information for visualization generation
func (rw *ResultWriter) SaveVisualizationInfo(result *TrainingResult, config *Config) error {
	if !config.Visualization.Enabled {
		return nil
	}
	
	// Save visualization data for later processing
	vizDataPath := filepath.Join(rw.resultDir, "visualization_data.json")
	
	vizData := map[string]interface{}{
		"output_dir": rw.resultDir,
		"formats":    config.Visualization.Formats,
		"dpi":        config.Visualization.DPI,
		"generate_report": config.Visualization.GenerateReport,
		"has_pr_curve": result.PRCurve != nil,
		"has_roc_curve": result.ROCCurve != nil,
		"num_models": len(result.BestWeights),
		"optimization_metric": config.TrainingConfig.OptimizationMetric,
	}
	
	// Add curve data if available
	if result.PRCurve != nil {
		vizData["pr_curve_data"] = map[string]interface{}{
			"x": result.PRCurve.X,
			"y": result.PRCurve.Y,
			"auc": result.PRCurve.AUC,
		}
	}
	
	if result.ROCCurve != nil {
		vizData["roc_curve_data"] = map[string]interface{}{
			"x": result.ROCCurve.X,
			"y": result.ROCCurve.Y,
			"auc": result.ROCCurve.AUC,
		}
	}
	
	// Add metric history
	if history, ok := result.MetricHistory[config.TrainingConfig.OptimizationMetric]; ok {
		vizData["metric_history"] = history
	}
	
	// Save visualization data
	data, err := json.MarshalIndent(vizData, "", "  ")
	if err != nil {
		return err
	}
	
	return os.WriteFile(vizDataPath, data, 0644)
}