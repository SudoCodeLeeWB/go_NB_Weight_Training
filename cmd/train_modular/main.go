package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"path/filepath"

	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/visualization"
	
	// Import model wrappers (in production, these would be plugins)
	_ "github.com/iwonbin/go-nb-weight-training/models/spam_ensemble"
)

func main() {
	// Command line flags
	var (
		modelDir   = flag.String("model", "", "Path to model directory")
		configPath = flag.String("config", "", "Path to configuration file")
		dataPath   = flag.String("data", "", "Path to dataset (CSV or JSON)")
		verbose    = flag.Bool("verbose", true, "Enable verbose logging")
		help       = flag.Bool("help", false, "Show help")
	)
	
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Weighted Naive Bayes Training Framework - Modular CLI\n\n")
		fmt.Fprintf(os.Stderr, "Usage:\n")
		fmt.Fprintf(os.Stderr, "  %s -model <model_dir> -data <dataset> [-config <config>]\n\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "Arguments:\n")
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExample:\n")
		fmt.Fprintf(os.Stderr, "  %s -model models/spam_ensemble -data datasets/spam_data.csv -config config/production_config.json\n", os.Args[0])
	}
	
	flag.Parse()
	
	if *help {
		flag.Usage()
		os.Exit(0)
	}
	
	// Validate required flags
	if *modelDir == "" || *dataPath == "" {
		flag.Usage()
		log.Fatal("\nError: -model and -data flags are required")
	}
	
	// Load model wrapper
	fmt.Printf("Loading model from: %s\n", *modelDir)
	wrapper, err := framework.LoadModelWrapper(*modelDir)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	
	// Get model info
	info := wrapper.GetInfo()
	fmt.Printf("Model: %s (v%s)\n", info.Name, info.Version)
	fmt.Printf("Description: %s\n", info.Description)
	fmt.Printf("Internal models: %v\n", info.Models)
	
	// Get the aggregated model
	model := wrapper.GetAggregatedModel()
	
	// Load dataset
	fmt.Printf("\nLoading dataset from: %s\n", *dataPath)
	dataset, err := loadDataset(*dataPath)
	if err != nil {
		log.Fatalf("Failed to load dataset: %v", err)
	}
	fmt.Printf("Dataset: %d samples, %d features\n", dataset.NumSamples, dataset.NumFeatures)
	
	// Load or create configuration
	var config *framework.Config
	if *configPath != "" {
		fmt.Printf("\nLoading config from: %s\n", *configPath)
		config, err = framework.LoadConfig(*configPath)
		if err != nil {
			log.Fatalf("Failed to load config: %v", err)
		}
	} else {
		fmt.Println("\nUsing default configuration")
		config = framework.DefaultConfig()
	}
	
	// Override verbose setting
	config.TrainingConfig.Verbose = *verbose
	
	// Ensure output directory exists
	if err := os.MkdirAll("./output", 0755); err != nil {
		log.Fatalf("Failed to create output directory: %v", err)
	}
	
	// Show initial weights
	fmt.Println("\nInitial weights:")
	modelNames := model.GetModelNames()
	weights := model.GetWeights()
	for i, name := range modelNames {
		fmt.Printf("  %s: %.4f\n", name, weights[i])
	}
	
	// Train
	fmt.Println("\n" + "============================================================")
	fmt.Println("Starting Weight Optimization...")
	fmt.Println("============================================================")
	
	result, err := framework.TrainAggregatedModel(dataset, model, config)
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}
	
	// Show results
	fmt.Println("\n" + "============================================================")
	fmt.Println("Optimization Complete!")
	fmt.Println("============================================================")
	
	fmt.Printf("\nTraining time: %v\n", result.TrainingTime)
	fmt.Printf("Best %s: %.4f\n", config.TrainingConfig.OptimizationMetric, 
		result.FinalMetrics[config.TrainingConfig.OptimizationMetric])
	
	// Show optimized weights
	fmt.Println("\nOptimized weights:")
	for i, name := range modelNames {
		change := result.BestWeights[i] - weights[i]
		fmt.Printf("  %s: %.4f (%+.4f)\n", name, result.BestWeights[i], change)
	}
	
	// Generate visualization
	if result.OutputDir != "" {
		reporter := visualization.NewReportGenerator(result.OutputDir)
		if err := reporter.GenerateReport(result, config); err != nil {
			fmt.Printf("\nWarning: Failed to generate report: %v\n", err)
		} else {
			fmt.Printf("\n✅ Results saved to: %s\n", result.OutputDir)
			fmt.Println("   - best_weights.json: Optimized weights")
			fmt.Println("   - report.html: Interactive visualization")
			fmt.Println("   - training_result.json: Complete results")
		}
	}
}

// loadDataset loads data from CSV or JSON
func loadDataset(path string) (*data.Dataset, error) {
	// LoadData now handles both CSV and JSON automatically
	return data.LoadData(path)
}