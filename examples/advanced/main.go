package main

import (
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/iwonbin/go-nb-weight-training/pkg/data"
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/iwonbin/go-nb-weight-training/pkg/visualization"
)

// UserModel represents a custom user-defined model
type UserModel struct {
	name      string
	modelPath string
	// In a real implementation, this would load and use an actual trained model
}

func (m *UserModel) Predict(samples [][]float64) ([]float64, error) {
	// This is where you would implement your actual model prediction
	// For example, calling a Python model via subprocess, or loading a serialized model
	
	// For demonstration, we'll simulate predictions
	predictions := make([]float64, len(samples))
	for i := range predictions {
		// Simulate model-specific behavior
		sum := 0.0
		for _, val := range samples[i] {
			sum += val
		}
		predictions[i] = sum / float64(len(samples[i]))
	}
	return predictions, nil
}

func (m *UserModel) GetName() string {
	return m.name
}

func main() {
	fmt.Println("=== Advanced Weighted Naive Bayes Training Example ===")

	// Step 1: Generate or load training data
	// This would typically be the predictions from your base classifiers
	if err := generateTrainingData("training_data.csv"); err != nil {
		log.Fatalf("Failed to generate training data: %v", err)
	}

	// Step 2: Load the data
	dataset, err := data.LoadData("training_data.csv")
	if err != nil {
		log.Fatalf("Failed to load data: %v", err)
	}

	fmt.Printf("Loaded %d samples with %d features\n", dataset.NumSamples, dataset.NumFeatures)
	fmt.Printf("Class balance: %.2f%% positive\n", dataset.ClassBalance()*100)

	// Step 3: Define your models
	// In practice, these would be your actual trained classifiers
	models := []framework.Model{
		&UserModel{name: "RandomForest_v1"},
		&UserModel{name: "XGBoost_v2"},
		&UserModel{name: "NeuralNet_v3"},
		&UserModel{name: "SVM_rbf"},
		&UserModel{name: "LogisticReg_l2"},
	}

	// Step 4: Create custom configuration
	config := &framework.Config{
		DataConfig: framework.DataConfig{
			ValidationSplit: 0.2,
			KFolds:          5,
			Stratified:      true,
			RandomSeed:      42,
		},
		TrainingConfig: framework.TrainingConfig{
			MaxEpochs:          100,
			BatchSize:          32,
			OptimizationMetric: "pr_auc",
			Verbose:            true,
			LogInterval:        10,
		},
		OptimizerConfig: framework.OptimizerConfig{
			Type:           "differential_evolution",
			PopulationSize: 100,
			MutationFactor: 0.8,
			CrossoverProb:  0.9,
			MinWeight:      0.0,
			MaxWeight:      3.0,
		},
		EarlyStopping: &framework.EarlyStoppingConfig{
			Patience: 15,
			MinDelta: 0.001,
			Monitor:  "val_pr_auc",
			Mode:     "max",
		},
		Visualization: framework.VisualizationConfig{
			Enabled:        true,
			OutputDir:      "./results",
			Formats:        []string{"png", "svg"},
			GenerateReport: true,
			DPI:            300,
		},
	}

	// Save configuration
	if err := config.SaveConfig("training_config.json"); err != nil {
		log.Printf("Failed to save config: %v", err)
	}

	// Step 5: Train with callbacks
	trainer := framework.NewTrainer(config)
	
	fmt.Println("\nStarting advanced training with cross-validation...")
	result, err := trainer.Train(dataset, models)
	if err != nil {
		log.Fatalf("Training failed: %v", err)
	}

	// Step 6: Analyze results
	fmt.Println("\n=== Detailed Results ===")
	fmt.Printf("Training completed in: %v\n", result.TrainingTime)
	fmt.Printf("Total epochs: %d\n", result.TotalEpochs)
	fmt.Printf("Converged: %v\n", result.Converged)

	// Display cross-validation results
	if len(result.CVResults) > 0 {
		fmt.Println("\nCross-Validation Results:")
		fmt.Println("Fold | Train PR-AUC | Val PR-AUC | Train ROC-AUC | Val ROC-AUC")
		fmt.Println("-----|---------------|------------|---------------|------------")
		
		for _, fold := range result.CVResults {
			fmt.Printf(" %2d  |    %.4f    |   %.4f   |    %.4f    |   %.4f\n",
				fold.FoldIndex+1,
				fold.TrainMetrics["pr_auc"],
				fold.ValMetrics["pr_auc"],
				fold.TrainMetrics["roc_auc"],
				fold.ValMetrics["roc_auc"])
		}
		
		// Calculate mean and std
		var prAucSum, rocAucSum float64
		for _, fold := range result.CVResults {
			prAucSum += fold.ValMetrics["pr_auc"]
			rocAucSum += fold.ValMetrics["roc_auc"]
		}
		meanPRAuc := prAucSum / float64(len(result.CVResults))
		meanROCAuc := rocAucSum / float64(len(result.CVResults))
		
		fmt.Printf("\nMean Val PR-AUC: %.4f\n", meanPRAuc)
		fmt.Printf("Mean Val ROC-AUC: %.4f\n", meanROCAuc)
	}

	// Display final weights with analysis
	fmt.Println("\nModel Weight Analysis:")
	fmt.Println("Model           | Weight  | Relative Importance")
	fmt.Println("----------------|---------|-------------------")
	
	totalWeight := 0.0
	for _, w := range result.BestWeights {
		totalWeight += w
	}
	
	for i, weight := range result.BestWeights {
		relativeImportance := (weight / totalWeight) * 100
		fmt.Printf("%-15s | %.4f  | %.1f%%\n", 
			models[i].GetName(), weight, relativeImportance)
	}

	// Step 7: Generate comprehensive report
	if config.Visualization.Enabled {
		fmt.Println("\nGenerating comprehensive visualization report...")
		
		// Create output directory
		os.MkdirAll(config.Visualization.OutputDir, 0755)
		
		reporter := visualization.NewReportGenerator(config.Visualization.OutputDir)
		if err := reporter.GenerateReport(result, config); err != nil {
			log.Printf("Failed to generate report: %v", err)
		} else {
			fmt.Printf("Report saved to: %s/report.html\n", config.Visualization.OutputDir)
		}
		
		// Additional custom plots
		plotter := visualization.NewPlotter(config.Visualization.OutputDir, float64(config.Visualization.DPI))
		
		// Plot weight distribution
		modelNames := make([]string, len(models))
		for i, m := range models {
			modelNames[i] = m.GetName()
		}
		
		if err := plotter.PlotWeightDistribution(result.BestWeights, modelNames, "weight_distribution.png"); err != nil {
			log.Printf("Failed to plot weight distribution: %v", err)
		}
		
		// Plot metric history if available
		if history, ok := result.MetricHistory["pr_auc"]; ok && len(history) > 0 {
			epochs := make([]float64, len(history))
			for i := range epochs {
				epochs[i] = float64(i)
			}
			
			if err := plotter.PlotMetricHistory(epochs, history, "PR-AUC", "pr_auc_history.png"); err != nil {
				log.Printf("Failed to plot PR-AUC history: %v", err)
			}
		}
	}

	// Step 8: Save the trained ensemble for production use
	if err := saveEnsemble(result.BestWeights, models, "ensemble_model.json"); err != nil {
		log.Printf("Failed to save ensemble: %v", err)
	} else {
		fmt.Println("\nEnsemble model saved to: ensemble_model.json")
	}

	// Clean up
	os.Remove("training_data.csv")
}

// generateTrainingData creates sample training data
// In practice, this would be your actual classifier predictions
func generateTrainingData(filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	writer := csv.NewWriter(file)
	defer writer.Flush()

	// Header
	header := []string{"rf_pred", "xgb_pred", "nn_pred", "svm_pred", "lr_pred", "label"}
	if err := writer.Write(header); err != nil {
		return err
	}

	// Generate 1000 samples
	for i := 0; i < 1000; i++ {
		var row []string
		
		// Simulate predictions from 5 models
		if i < 600 { // True positives and true negatives
			if i < 300 { // True positives
				row = []string{
					fmt.Sprintf("%.4f", 0.7+randFloat()*0.3),
					fmt.Sprintf("%.4f", 0.65+randFloat()*0.35),
					fmt.Sprintf("%.4f", 0.8+randFloat()*0.2),
					fmt.Sprintf("%.4f", 0.6+randFloat()*0.4),
					fmt.Sprintf("%.4f", 0.75+randFloat()*0.25),
					"1",
				}
			} else { // True negatives
				row = []string{
					fmt.Sprintf("%.4f", randFloat()*0.3),
					fmt.Sprintf("%.4f", randFloat()*0.35),
					fmt.Sprintf("%.4f", randFloat()*0.2),
					fmt.Sprintf("%.4f", randFloat()*0.4),
					fmt.Sprintf("%.4f", randFloat()*0.25),
					"0",
				}
			}
		} else { // Harder cases
			if i%2 == 0 {
				row = []string{
					fmt.Sprintf("%.4f", 0.4+randFloat()*0.3),
					fmt.Sprintf("%.4f", 0.5+randFloat()*0.2),
					fmt.Sprintf("%.4f", 0.45+randFloat()*0.3),
					fmt.Sprintf("%.4f", 0.3+randFloat()*0.4),
					fmt.Sprintf("%.4f", 0.5+randFloat()*0.3),
					"1",
				}
			} else {
				row = []string{
					fmt.Sprintf("%.4f", 0.3+randFloat()*0.4),
					fmt.Sprintf("%.4f", 0.4+randFloat()*0.3),
					fmt.Sprintf("%.4f", 0.35+randFloat()*0.3),
					fmt.Sprintf("%.4f", 0.4+randFloat()*0.3),
					fmt.Sprintf("%.4f", 0.45+randFloat()*0.2),
					"0",
				}
			}
		}
		
		if err := writer.Write(row); err != nil {
			return err
		}
	}

	return nil
}

// Simple random float generator
var randSeed int64 = 42

func randFloat() float64 {
	randSeed = (randSeed*1103515245 + 12345) & 0x7fffffff
	return float64(randSeed) / float64(0x7fffffff)
}

// saveEnsemble saves the ensemble model configuration
func saveEnsemble(weights []float64, models []framework.Model, filename string) error {
	file, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	fmt.Fprintf(file, "{\n")
	fmt.Fprintf(file, "  \"ensemble_type\": \"weighted_naive_bayes\",\n")
	fmt.Fprintf(file, "  \"models\": [\n")
	
	for i, model := range models {
		fmt.Fprintf(file, "    {\n")
		fmt.Fprintf(file, "      \"name\": \"%s\",\n", model.GetName())
		fmt.Fprintf(file, "      \"weight\": %.6f\n", weights[i])
		fmt.Fprintf(file, "    }")
		if i < len(models)-1 {
			fmt.Fprintf(file, ",")
		}
		fmt.Fprintf(file, "\n")
	}
	
	fmt.Fprintf(file, "  ]\n")
	fmt.Fprintf(file, "}\n")

	return nil
}