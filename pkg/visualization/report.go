package visualization

import (
	"fmt"
	"html/template"
	"os"
	"path/filepath"
	"time"

	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
)

// ReportGenerator generates HTML reports
type ReportGenerator struct {
	outputDir string
}

// NewReportGenerator creates a new report generator
func NewReportGenerator(outputDir string) *ReportGenerator {
	return &ReportGenerator{
		outputDir: outputDir,
	}
}

// GenerateReport creates an HTML report from training results
func (rg *ReportGenerator) GenerateReport(result *framework.TrainingResult, config *framework.Config) error {
	// Create output directory if it doesn't exist
	if err := os.MkdirAll(rg.outputDir, 0755); err != nil {
		return fmt.Errorf("failed to create output directory: %w", err)
	}
	
	// Generate plots
	plotter := NewPlotter(rg.outputDir, 300)
	
	// Plot PR curve
	if result.PRCurve != nil {
		if err := plotter.PlotPRCurve(result.PRCurve.X, result.PRCurve.Y, 
			result.PRCurve.AUC, "pr_curve.png"); err != nil {
			return fmt.Errorf("failed to plot PR curve: %w", err)
		}
	}
	
	// Plot ROC curve
	if result.ROCCurve != nil {
		if err := plotter.PlotROCCurve(result.ROCCurve.X, result.ROCCurve.Y, 
			result.ROCCurve.AUC, "roc_curve.png"); err != nil {
			return fmt.Errorf("failed to plot ROC curve: %w", err)
		}
	}
	
	// Plot calibration comparison if available
	if result.CalibrationComparison != nil {
		// Score distributions
		if err := plotter.PlotScoreDistributions(
			result.CalibrationComparison.CalibrationComparisons,
			result.CalibrationComparison.RawScoreDistribution,
			"calibration_score_distributions.png"); err != nil {
			fmt.Printf("Warning: failed to plot score distributions: %v\n", err)
		}
		
		// Calibration comparison
		if err := plotter.PlotCalibrationComparison(
			result.CalibrationComparison.CalibrationComparisons,
			result.CalibrationComparison.ModelProvidedCalibration,
			"calibration_comparison.png"); err != nil {
			fmt.Printf("Warning: failed to plot calibration comparison: %v\n", err)
		}
		
		// Calibration PR curves
		if err := plotter.PlotCalibrationCurves(
			result.CalibrationComparison.CalibrationComparisons,
			result.CalibrationComparison.ModelProvidedCalibration,
			"calibration_pr_curves.png"); err != nil {
			fmt.Printf("Warning: failed to plot calibration curves: %v\n", err)
		}
	}
	
	// Generate HTML report
	reportPath := filepath.Join(rg.outputDir, "report.html")
	return rg.generateHTML(result, config, reportPath)
}

// generateHTML creates the HTML report file
func (rg *ReportGenerator) generateHTML(result *framework.TrainingResult, config *framework.Config, path string) error {
	tmpl := template.Must(template.New("report").Parse(reportTemplate))
	
	file, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create report file: %w", err)
	}
	defer file.Close()
	
	data := struct {
		Result                    *framework.TrainingResult
		Config                    *framework.Config
		GeneratedAt               time.Time
		PRCurvePath               string
		ROCCurvePath              string
		CalibrationScoreDistPath  string
		CalibrationComparisonPath string
		CalibrationCurvesPath     string
		HasCalibration            bool
	}{
		Result:                    result,
		Config:                    config,
		GeneratedAt:               time.Now(),
		PRCurvePath:               "pr_curve.png",
		ROCCurvePath:              "roc_curve.png",
		CalibrationScoreDistPath:  "calibration_score_distributions.png",
		CalibrationComparisonPath: "calibration_comparison.png",
		CalibrationCurvesPath:     "calibration_pr_curves.png",
		HasCalibration:            result.CalibrationComparison != nil,
	}
	
	return tmpl.Execute(file, data)
}

// reportTemplate is the HTML template for the report
const reportTemplate = `
<!DOCTYPE html>
<html>
<head>
    <title>Weighted Naive Bayes Training Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f4f4f4;
        }
        .container {
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            border-left: 4px solid #3498db;
        }
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }
        .metric-label {
            color: #7f8c8d;
            font-size: 14px;
        }
        .plot-container {
            margin: 20px 0;
            text-align: center;
        }
        .plot-container img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        th, td {
            text-align: left;
            padding: 12px;
            border-bottom: 1px solid #ddd;
        }
        th {
            background-color: #3498db;
            color: white;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
        .weights-table {
            margin: 20px 0;
        }
        .info-section {
            background-color: #ecf0f1;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }
        .cv-results {
            margin-top: 20px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Weighted Naive Bayes Training Report</h1>
        <p>Generated at: {{.GeneratedAt.Format "2006-01-02 15:04:05"}}</p>
        
        <div class="info-section">
            <h3>Training Configuration</h3>
            <ul>
                <li>Optimization Metric: {{.Config.TrainingConfig.OptimizationMetric}}</li>
                <li>Max Epochs: {{.Config.TrainingConfig.MaxEpochs}}</li>
                <li>Optimizer: {{.Config.OptimizerConfig.Type}}</li>
                <li>K-Folds: {{.Config.DataConfig.KFolds}}</li>
                <li>Stratified: {{.Config.DataConfig.Stratified}}</li>
            </ul>
        </div>
        
        <h2>Performance Metrics</h2>
        <div class="metrics-grid">
            {{range $key, $value := .Result.FinalMetrics}}
            <div class="metric-card">
                <div class="metric-label">{{$key}}</div>
                <div class="metric-value">{{printf "%.4f" $value}}</div>
            </div>
            {{end}}
        </div>
        
        <h2>Model Weights</h2>
        <div class="weights-table">
            <table>
                <tr>
                    <th>Model Index</th>
                    <th>Weight</th>
                </tr>
                {{range $i, $weight := .Result.BestWeights}}
                <tr>
                    <td>Model {{$i}}</td>
                    <td>{{printf "%.4f" $weight}}</td>
                </tr>
                {{end}}
            </table>
        </div>
        
        <h2>Performance Curves</h2>
        <div class="plot-container">
            <h3>Precision-Recall Curve</h3>
            {{if .Result.PRCurve}}
            <img src="{{.PRCurvePath}}" alt="PR Curve">
            <p>PR-AUC: {{printf "%.4f" .Result.PRCurve.AUC}}</p>
            {{end}}
        </div>
        
        <div class="plot-container">
            <h3>ROC Curve</h3>
            {{if .Result.ROCCurve}}
            <img src="{{.ROCCurvePath}}" alt="ROC Curve">
            <p>ROC-AUC: {{printf "%.4f" .Result.ROCCurve.AUC}}</p>
            {{end}}
        </div>
        
        {{if .HasCalibration}}
        <h2>Calibration Comparison</h2>
        <div class="info-section">
            <p><strong>Best Calibration Method:</strong> {{.Result.CalibrationComparison.BestMethod}} ({{.Config.TrainingConfig.OptimizationMetric}} = {{printf "%.4f" .Result.CalibrationComparison.BestScore}})</p>
        </div>
        
        <div class="plot-container">
            <h3>Score Distributions by Calibration Method</h3>
            <img src="{{.CalibrationScoreDistPath}}" alt="Score Distributions">
            <p>Box plots showing score ranges for raw and calibrated scores</p>
        </div>
        
        <div class="plot-container">
            <h3>Calibration Method Performance</h3>
            <img src="{{.CalibrationComparisonPath}}" alt="Calibration Comparison">
            <p>Comparison of PR-AUC, Precision, Recall, and F1-Score across methods</p>
        </div>
        
        <div class="plot-container">
            <h3>PR Curves by Calibration Method</h3>
            <img src="{{.CalibrationCurvesPath}}" alt="Calibration PR Curves">
            <p>Precision-Recall curves for each calibration method</p>
        </div>
        
        <h3>Calibration Methods Comparison</h3>
        <table>
            <tr>
                <th>Method</th>
                <th>PR-AUC</th>
                <th>Optimal Threshold</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>Score Range</th>
            </tr>
            {{range .Result.CalibrationComparison.CalibrationComparisons}}
            <tr>
                <td>{{.Method}}</td>
                <td>{{printf "%.4f" .PRCurve.AUC}}</td>
                <td>{{printf "%.4f" .OptimalThreshold}}</td>
                <td>{{printf "%.4f" (index .MetricsAtThreshold "precision")}}</td>
                <td>{{printf "%.4f" (index .MetricsAtThreshold "recall")}}</td>
                <td>{{printf "%.4f" (index .MetricsAtThreshold "f1_score")}}</td>
                <td>[{{printf "%.6f" .ScoreDistribution.Min}}, {{printf "%.6f" .ScoreDistribution.Max}}]</td>
            </tr>
            {{end}}
            {{if .Result.CalibrationComparison.ModelProvidedCalibration}}
            <tr style="background-color: #e8f4f8;">
                <td><strong>Model's {{.Result.CalibrationComparison.ModelProvidedCalibration.Method}}</strong></td>
                <td><strong>{{printf "%.4f" .Result.CalibrationComparison.ModelProvidedCalibration.PRCurve.AUC}}</strong></td>
                <td><strong>{{printf "%.4f" .Result.CalibrationComparison.ModelProvidedCalibration.OptimalThreshold}}</strong></td>
                <td><strong>{{printf "%.4f" (index .Result.CalibrationComparison.ModelProvidedCalibration.MetricsAtThreshold "precision")}}</strong></td>
                <td><strong>{{printf "%.4f" (index .Result.CalibrationComparison.ModelProvidedCalibration.MetricsAtThreshold "recall")}}</strong></td>
                <td><strong>{{printf "%.4f" (index .Result.CalibrationComparison.ModelProvidedCalibration.MetricsAtThreshold "f1_score")}}</strong></td>
                <td><strong>[{{printf "%.6f" .Result.CalibrationComparison.ModelProvidedCalibration.ScoreDistribution.Min}}, {{printf "%.6f" .Result.CalibrationComparison.ModelProvidedCalibration.ScoreDistribution.Max}}]</strong></td>
            </tr>
            {{end}}
        </table>
        {{end}}
        
        {{if .Result.CVResults}}
        <h2>Cross-Validation Results</h2>
        <div class="cv-results">
            <table>
                <tr>
                    <th>Fold</th>
                    <th>Train PR-AUC</th>
                    <th>Val PR-AUC</th>
                    <th>Train ROC-AUC</th>
                    <th>Val ROC-AUC</th>
                </tr>
                {{range .Result.CVResults}}
                <tr>
                    <td>{{.FoldIndex}}</td>
                    <td>{{printf "%.4f" (index .TrainMetrics "pr_auc")}}</td>
                    <td>{{printf "%.4f" (index .ValMetrics "pr_auc")}}</td>
                    <td>{{printf "%.4f" (index .TrainMetrics "roc_auc")}}</td>
                    <td>{{printf "%.4f" (index .ValMetrics "roc_auc")}}</td>
                </tr>
                {{end}}
            </table>
        </div>
        {{end}}
        
        <div class="info-section">
            <h3>Training Summary</h3>
            <ul>
                <li>Total Epochs: {{.Result.TotalEpochs}}</li>
                <li>Training Time: {{.Result.TrainingTime}}</li>
                <li>Converged: {{.Result.Converged}}</li>
            </ul>
        </div>
    </div>
</body>
</html>
`