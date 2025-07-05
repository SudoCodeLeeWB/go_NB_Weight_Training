package main

import (
	"flag"
	"fmt"
	"log"
	"os"

	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/visualization"
)

func main() {
	// Command line flags
	var (
		dataPath   = flag.String("data", "", "Path to training data (CSV or JSON)")
		configPath = flag.String("config", "", "Path to configuration file (optional)")
		verbose    = flag.Bool("verbose", true, "Enable verbose logging")
	)
	flag.Parse()

	// Validate required flags
	if *dataPath == "" {
		flag.Usage()
		log.Fatal("Error: -data flag is required")
	}

	// Load or create configuration
	var config *framework.Config
	var err error
	if *configPath != "" {
		config, err = framework.LoadConfig(*configPath)
		if err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}
	} else {
		config = framework.DefaultConfig()
	}

	// Override config with command line flags
	config.TrainingConfig.Verbose = *verbose
	
	// Output directory is always ./output
	if err := os.MkdirAll("./output", 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}

	// Load data
	fmt.Println("Loading data...")
	dataset, err := data.LoadData(*dataPath)
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}
	fmt.Printf("Loaded %d samples with %d features\n", dataset.NumSamples, dataset.NumFeatures)
	fmt.Printf("Class distribution: %.2f%% positive\n", dataset.ClassBalance()*100)

	// Create example models (in real usage, these would be provided by the user)
	models := createExampleModels(dataset.NumFeatures)

	// Create trainer
	fmt.Println("\nInitializing trainer...")
	trainer := framework.NewTrainer(config)

	// Train
	fmt.Println("Starting training...")
	result, err := trainer.Train(dataset, models)
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	// Print results
	fmt.Println("\n=== Training Results ===")
	fmt.Printf("Training completed in: %v\n", result.TrainingTime)
	fmt.Printf("Total epochs: %d\n", result.TotalEpochs)
	fmt.Printf("Converged: %v\n", result.Converged)

	fmt.Println("\nFinal Metrics:")
	for metric, value := range result.FinalMetrics {
		fmt.Printf("  %s: %.4f\n", metric, value)
	}

	fmt.Println("\nOptimized Weights:")
	for i, weight := range result.BestWeights {
		fmt.Printf("  Model %d: %.4f\n", i, weight)
	}

	// Generate visualizations if enabled
	if config.Visualization.Enabled {
		fmt.Println("\nGenerating visualizations...")
		reporter := visualization.NewReportGenerator(*outputDir)
		if err := reporter.GenerateReport(result, config); err != nil {
			log.Printf("Failed to generate report: %v", err)
		} else {
			fmt.Printf("Report saved to: %s/report.html\n", *outputDir)
		}
	}

	// Save weights
	weightsPath := *outputDir + "/weights.json"
	if err := saveWeights(result.BestWeights, weightsPath); err != nil {
		log.Printf("Failed to save weights: %v", err)
	} else {
		fmt.Printf("\nWeights saved to: %s\n", weightsPath)
	}

	// Save configuration
	configSavePath := *outputDir + "/config.json"
	if err := config.SaveConfig(configSavePath); err != nil {
		log.Printf("Failed to save config: %v", err)
	} else {
		fmt.Printf("Configuration saved to: %s\n", configSavePath)
	}
}

// createExampleModels creates dummy models for demonstration
// In real usage, users would provide their own models
func createExampleModels(numFeatures int) []framework.Model {
	// Create some example models
	models := []framework.Model{
		&ExampleModel{name: "Model1", weight: 0.8},
		&ExampleModel{name: "Model2", weight: 0.6},
		&ExampleModel{name: "Model3", weight: 0.7},
		&ExampleModel{name: "Model4", weight: 0.9},
		&ExampleModel{name: "Model5", weight: 0.5},
	}
	return models
}

// ExampleModel is a dummy model for demonstration
type ExampleModel struct {
	name   string
	weight float64
}

func (m *ExampleModel) Predict(samples [][]float64) ([]float64, error) {
	// Simple dummy prediction: average of features * weight
	predictions := make([]float64, len(samples))
	for i, sample := range samples {
		sum := 0.0
		for _, feature := range sample {
			sum += feature
		}
		avg := sum / float64(len(sample))
		predictions[i] = avg * m.weight
		
		// Ensure in [0, 1] range
		if predictions[i] > 1 {
			predictions[i] = 1
		} else if predictions[i] < 0 {
			predictions[i] = 0
		}
	}
	return predictions, nil
}

func (m *ExampleModel) GetName() string {
	return m.name
}

// saveWeights saves weights to a JSON file
func saveWeights(weights []float64, path string) error {
	file, err := os.Create(path)
	if err != nil {
		return err
	}
	defer file.Close()

	// Simple JSON format
	fmt.Fprintf(file, "{\n  \"weights\": [")
	for i, w := range weights {
		if i > 0 {
			fmt.Fprintf(file, ",")
		}
		fmt.Fprintf(file, "\n    %.6f", w)
	}
	fmt.Fprintf(file, "\n  ]\n}\n")

	return nil
}