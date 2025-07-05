package framework

import (
	"encoding/json"
	"fmt"
	"os"
)

// Config holds all configuration for the training framework
type Config struct {
	// Data configuration
	DataConfig DataConfig `json:"data_config"`
	
	// Training configuration
	TrainingConfig TrainingConfig `json:"training_config"`
	
	// Optimization configuration
	OptimizerConfig OptimizerConfig `json:"optimizer_config"`
	
	// Early stopping configuration
	EarlyStopping *EarlyStoppingConfig `json:"early_stopping,omitempty"`
	
	// Visualization configuration
	Visualization VisualizationConfig `json:"visualization"`
}

// DataConfig holds data-related configuration
type DataConfig struct {
	// Path to training data
	TrainPath string `json:"train_path"`
	
	// Validation split ratio (0-1)
	ValidationSplit float64 `json:"validation_split"`
	
	// Calibration split ratio (0-1) - only used with UseThreeWaySplit
	CalibrationSplit float64 `json:"calibration_split"`
	
	// Whether to use three-way split (train/calibration/test)
	UseThreeWaySplit bool `json:"use_three_way_split"`
	
	// Number of folds for cross-validation
	KFolds int `json:"k_folds"`
	
	// Whether to use stratified splitting
	Stratified bool `json:"stratified"`
	
	// Random seed for reproducibility
	RandomSeed int64 `json:"random_seed"`
}

// TrainingConfig holds training-related configuration
type TrainingConfig struct {
	// Maximum number of epochs
	MaxEpochs int `json:"max_epochs"`
	
	// Batch size for evaluation
	BatchSize int `json:"batch_size"`
	
	// Metric to optimize ("pr_auc", "roc_auc", "precision", "recall")
	OptimizationMetric string `json:"optimization_metric"`
	
	// Whether to log progress
	Verbose bool `json:"verbose"`
	
	// Log interval (epochs)
	LogInterval int `json:"log_interval"`
	
	// Calibration settings
	EnableCalibration bool   `json:"enable_calibration"`
	CalibrationMethod string `json:"calibration_method"` // "platt", "isotonic", "beta", "none"
	ThresholdMetric   string `json:"threshold_metric"` // "f1", "precision", "recall", "accuracy", "mcc", "pr_distance"
}

// OptimizerConfig holds optimizer-related configuration
type OptimizerConfig struct {
	// Optimizer type ("differential_evolution", "random_search", "grid_search")
	Type string `json:"type"`
	
	// Population size for evolutionary algorithms
	PopulationSize int `json:"population_size"`
	
	// Mutation factor for differential evolution
	MutationFactor float64 `json:"mutation_factor"`
	
	// Crossover probability
	CrossoverProb float64 `json:"crossover_prob"`
	
	// Weight bounds
	MinWeight float64 `json:"min_weight"`
	MaxWeight float64 `json:"max_weight"`
	
	// Enforce non-zero weights (prevents models from being completely ignored)
	EnforceNonZero bool `json:"enforce_non_zero"`
}

// EarlyStoppingConfig holds early stopping configuration
type EarlyStoppingConfig struct {
	// Number of epochs to wait before stopping
	Patience int `json:"patience"`
	
	// Minimum change to qualify as improvement
	MinDelta float64 `json:"min_delta"`
	
	// Metric to monitor
	Monitor string `json:"monitor"`
	
	// Whether higher is better for the monitored metric
	Mode string `json:"mode"` // "max" or "min"
}

// VisualizationConfig holds visualization configuration
type VisualizationConfig struct {
	// Whether to generate plots
	Enabled bool `json:"enabled"`
	
	// Output directory for plots (deprecated - always uses ./output)
	OutputDir string `json:"output_dir,omitempty"`
	
	// Plot formats ("png", "svg", "pdf")
	Formats []string `json:"formats"`
	
	// Whether to generate HTML report
	GenerateReport bool `json:"generate_report"`
	
	// DPI for raster formats
	DPI int `json:"dpi"`
}

// DefaultConfig returns a default configuration
func DefaultConfig() *Config {
	return &Config{
		DataConfig: DataConfig{
			ValidationSplit: 0.2,
			KFolds:          5,
			Stratified:      true,
			RandomSeed:      42,
		},
		TrainingConfig: TrainingConfig{
			MaxEpochs:          100,
			BatchSize:          32,
			OptimizationMetric: "pr_auc",
			Verbose:            true,
			LogInterval:        10,
			EnableCalibration:  true,
			CalibrationMethod:  "beta",
			ThresholdMetric:    "precision",
		},
		OptimizerConfig: OptimizerConfig{
			Type:           "differential_evolution",
			PopulationSize: 50,
			MutationFactor: 0.8,
			CrossoverProb:  0.9,
			MinWeight:      0.01,  // Avoid 0 to prevent models being ignored
			MaxWeight:      2.0,
		},
		EarlyStopping: &EarlyStoppingConfig{
			Patience: 10,
			MinDelta: 0.001,
			Monitor:  "val_pr_auc",
			Mode:     "max",
		},
		Visualization: VisualizationConfig{
			Enabled:        true,
			Formats:        []string{"png"},
			GenerateReport: true,
			DPI:            300,
		},
	}
}

// LoadConfig loads configuration from a JSON file
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}
	
	config := DefaultConfig()
	if err := json.Unmarshal(data, config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}
	
	return config, nil
}

// SaveConfig saves configuration to a JSON file
func (c *Config) SaveConfig(path string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}
	
	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config: %w", err)
	}
	
	return nil
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	if c.DataConfig.ValidationSplit < 0 || c.DataConfig.ValidationSplit > 1 {
		return fmt.Errorf("validation split must be between 0 and 1")
	}
	
	if c.DataConfig.KFolds < 1 {
		return fmt.Errorf("k_folds must be at least 1")
	}
	
	if c.TrainingConfig.MaxEpochs < 1 {
		return fmt.Errorf("max_epochs must be at least 1")
	}
	
	if c.OptimizerConfig.MinWeight >= c.OptimizerConfig.MaxWeight {
		return fmt.Errorf("min_weight must be less than max_weight")
	}
	
	validMetrics := map[string]bool{
		"pr_auc": true, "roc_auc": true, "precision": true, "recall": true,
	}
	if !validMetrics[c.TrainingConfig.OptimizationMetric] {
		return fmt.Errorf("invalid optimization metric: %s", c.TrainingConfig.OptimizationMetric)
	}
	
	return nil
}