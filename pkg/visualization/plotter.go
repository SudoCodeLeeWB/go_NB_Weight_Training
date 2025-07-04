package visualization

import (
	"fmt"
	"image/color"
	"path/filepath"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// Plotter handles visualization of training results
type Plotter struct {
	outputDir string
	dpi       float64
}

// NewPlotter creates a new plotter
func NewPlotter(outputDir string, dpi float64) *Plotter {
	return &Plotter{
		outputDir: outputDir,
		dpi:       dpi,
	}
}

// PlotPRCurve plots a precision-recall curve
func (p *Plotter) PlotPRCurve(recalls, precisions []float64, auc float64, filename string) error {
	// Create plot
	plt := plot.New()
	plt.Title.Text = fmt.Sprintf("Precision-Recall Curve (AUC = %.3f)", auc)
	plt.X.Label.Text = "Recall"
	plt.Y.Label.Text = "Precision"
	
	// Set axis ranges
	plt.X.Min = 0
	plt.X.Max = 1
	plt.Y.Min = 0
	plt.Y.Max = 1
	
	// Create line plot
	pts := make(plotter.XYs, len(recalls))
	for i := range recalls {
		pts[i].X = recalls[i]
		pts[i].Y = precisions[i]
	}
	
	line, err := plotter.NewLine(pts)
	if err != nil {
		return fmt.Errorf("failed to create line plot: %w", err)
	}
	line.Color = color.RGBA{R: 31, G: 119, B: 180, A: 255} // Blue
	line.Width = vg.Points(2)
	
	// Add baseline (random classifier)
	baseLine := plotter.NewFunction(func(x float64) float64 {
		// For PR curve, baseline depends on positive class ratio
		// Here we use 0.5 as a simple baseline
		return 0.5
	})
	baseLine.Color = color.RGBA{R: 255, G: 127, B: 14, A: 255} // Orange
	baseLine.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	baseLine.Width = vg.Points(1)
	
	// Add plots
	plt.Add(line)
	plt.Add(baseLine)
	
	// Add legend
	plt.Legend.Add("PR Curve", line)
	plt.Legend.Add("Baseline", baseLine)
	plt.Legend.Top = true
	plt.Legend.Left = true
	
	// Add grid
	plt.Add(plotter.NewGrid())
	
	// Save plot
	outputPath := filepath.Join(p.outputDir, filename)
	if err := plt.Save(6*vg.Inch, 4*vg.Inch, outputPath); err != nil {
		return fmt.Errorf("failed to save plot: %w", err)
	}
	
	return nil
}

// PlotROCCurve plots a ROC curve
func (p *Plotter) PlotROCCurve(fprs, tprs []float64, auc float64, filename string) error {
	// Create plot
	plt := plot.New()
	plt.Title.Text = fmt.Sprintf("ROC Curve (AUC = %.3f)", auc)
	plt.X.Label.Text = "False Positive Rate"
	plt.Y.Label.Text = "True Positive Rate"
	
	// Set axis ranges
	plt.X.Min = 0
	plt.X.Max = 1
	plt.Y.Min = 0
	plt.Y.Max = 1
	
	// Create line plot
	pts := make(plotter.XYs, len(fprs))
	for i := range fprs {
		pts[i].X = fprs[i]
		pts[i].Y = tprs[i]
	}
	
	line, err := plotter.NewLine(pts)
	if err != nil {
		return fmt.Errorf("failed to create line plot: %w", err)
	}
	line.Color = color.RGBA{R: 31, G: 119, B: 180, A: 255} // Blue
	line.Width = vg.Points(2)
	
	// Add diagonal line (random classifier)
	diagonal := plotter.NewFunction(func(x float64) float64 { return x })
	diagonal.Color = color.RGBA{R: 255, G: 127, B: 14, A: 255} // Orange
	diagonal.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	diagonal.Width = vg.Points(1)
	
	// Add plots
	plt.Add(line)
	plt.Add(diagonal)
	
	// Add legend
	plt.Legend.Add("ROC Curve", line)
	plt.Legend.Add("Random Classifier", diagonal)
	plt.Legend.Top = false
	plt.Legend.Left = false
	
	// Add grid
	plt.Add(plotter.NewGrid())
	
	// Save plot
	outputPath := filepath.Join(p.outputDir, filename)
	if err := plt.Save(6*vg.Inch, 4*vg.Inch, outputPath); err != nil {
		return fmt.Errorf("failed to save plot: %w", err)
	}
	
	return nil
}

// PlotMetricHistory plots the history of a metric over epochs
func (p *Plotter) PlotMetricHistory(epochs []float64, values []float64, metricName, filename string) error {
	// Create plot
	plt := plot.New()
	plt.Title.Text = fmt.Sprintf("%s History", metricName)
	plt.X.Label.Text = "Epoch"
	plt.Y.Label.Text = metricName
	
	// Create line plot
	pts := make(plotter.XYs, len(epochs))
	for i := range epochs {
		pts[i].X = epochs[i]
		pts[i].Y = values[i]
	}
	
	line, err := plotter.NewLine(pts)
	if err != nil {
		return fmt.Errorf("failed to create line plot: %w", err)
	}
	line.Color = color.RGBA{R: 31, G: 119, B: 180, A: 255} // Blue
	line.Width = vg.Points(2)
	
	// Add plot
	plt.Add(line)
	
	// Add grid
	plt.Add(plotter.NewGrid())
	
	// Save plot
	outputPath := filepath.Join(p.outputDir, filename)
	if err := plt.Save(6*vg.Inch, 4*vg.Inch, outputPath); err != nil {
		return fmt.Errorf("failed to save plot: %w", err)
	}
	
	return nil
}

// PlotWeightDistribution plots the distribution of weights
func (p *Plotter) PlotWeightDistribution(weights []float64, modelNames []string, filename string) error {
	// Create plot
	plt := plot.New()
	plt.Title.Text = "Model Weight Distribution"
	plt.X.Label.Text = "Model"
	plt.Y.Label.Text = "Weight"
	
	// Create bar chart data
	bars := make(plotter.Values, len(weights))
	for i, w := range weights {
		bars[i] = w
	}
	
	// Create bar chart
	barChart, err := plotter.NewBarChart(bars, vg.Points(20))
	if err != nil {
		return fmt.Errorf("failed to create bar chart: %w", err)
	}
	barChart.Color = color.RGBA{R: 31, G: 119, B: 180, A: 255} // Blue
	
	// Add bar chart
	plt.Add(barChart)
	
	// Set X axis labels
	if len(modelNames) == len(weights) {
		plt.NominalX(modelNames...)
	}
	
	// Add grid
	plt.Add(plotter.NewGrid())
	
	// Save plot
	outputPath := filepath.Join(p.outputDir, filename)
	if err := plt.Save(8*vg.Inch, 4*vg.Inch, outputPath); err != nil {
		return fmt.Errorf("failed to save plot: %w", err)
	}
	
	return nil
}

// PlotConfusionMatrix visualizes a confusion matrix
func (p *Plotter) PlotConfusionMatrix(tp, tn, fp, fn int, filename string) error {
	// For confusion matrix, we'll create a simple text-based visualization
	// gonum/plot doesn't have built-in heatmap support
	
	// Create plot
	plt := plot.New()
	plt.Title.Text = "Confusion Matrix"
	
	// Create a scatter plot with text annotations
	// This is a workaround for heatmap
	pts := plotter.XYs{
		{X: 0, Y: 0}, // TN
		{X: 1, Y: 0}, // FP
		{X: 0, Y: 1}, // FN
		{X: 1, Y: 1}, // TP
	}
	
	scatter, err := plotter.NewScatter(pts)
	if err != nil {
		return fmt.Errorf("failed to create scatter plot: %w", err)
	}
	scatter.GlyphStyle.Shape = draw.BoxGlyph{}
	scatter.GlyphStyle.Color = color.RGBA{R: 200, G: 200, B: 200, A: 255}
	scatter.GlyphStyle.Radius = vg.Points(50)
	
	plt.Add(scatter)
	
	// Add labels
	labels := []string{
		fmt.Sprintf("TN: %d", tn),
		fmt.Sprintf("FP: %d", fp),
		fmt.Sprintf("FN: %d", fn),
		fmt.Sprintf("TP: %d", tp),
	}
	
	for i, pt := range pts {
		label, err := plotter.NewLabels(plotter.XYLabels{
			XYs:    []plotter.XY{pt},
			Labels: []string{labels[i]},
		})
		if err != nil {
			return fmt.Errorf("failed to create label: %w", err)
		}
		plt.Add(label)
	}
	
	// Set axis labels
	plt.X.Label.Text = "Predicted"
	plt.Y.Label.Text = "Actual"
	plt.NominalX("Negative", "Positive")
	plt.NominalY("Negative", "Positive")
	
	// Save plot
	outputPath := filepath.Join(p.outputDir, filename)
	if err := plt.Save(6*vg.Inch, 6*vg.Inch, outputPath); err != nil {
		return fmt.Errorf("failed to save plot: %w", err)
	}
	
	return nil
}