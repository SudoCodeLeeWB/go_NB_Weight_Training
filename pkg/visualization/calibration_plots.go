package visualization

import (
	"fmt"
	"image/color"
	"path/filepath"

	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
	"gonum.org/v1/plot/vg/draw"
)

// PlotScoreDistributions creates box plots for each calibration method
func (p *Plotter) PlotScoreDistributions(comparisons []framework.CalibrationComparison, 
	rawDist framework.ScoreDistribution, filename string) error {
	
	// Create plot
	plt := plot.New()
	plt.Title.Text = "Score Distributions by Calibration Method"
	plt.Y.Label.Text = "Score"
	
	// Prepare data for box plots
	groupNames := []string{"Raw"}
	values := make([]plotter.Values, 1)
	
	// Add raw score distribution (using percentiles for box plot)
	values[0] = plotter.Values{
		rawDist.Percentiles[0],   // min
		rawDist.Percentiles[25],  // Q1
		rawDist.Percentiles[50],  // median
		rawDist.Percentiles[75],  // Q3
		rawDist.Percentiles[100], // max
	}
	
	// Add calibration methods
	for _, comp := range comparisons {
		groupNames = append(groupNames, comp.Method)
		dist := comp.ScoreDistribution
		vals := plotter.Values{
			dist.Percentiles[0],
			dist.Percentiles[25],
			dist.Percentiles[50],
			dist.Percentiles[75],
			dist.Percentiles[100],
		}
		values = append(values, vals)
	}
	
	// Create box plots
	width := vg.Points(20)
	for i, vals := range values {
		// Create a box plot for each method
		box, err := plotter.NewBoxPlot(width, float64(i), vals)
		if err != nil {
			return fmt.Errorf("failed to create box plot: %w", err)
		}
		
		// Set colors
		if i == 0 {
			// Raw scores in red
			box.FillColor = color.RGBA{R: 255, G: 200, B: 200, A: 255}
			box.BoxStyle.Color = color.RGBA{R: 200, G: 0, B: 0, A: 255}
		} else {
			// Calibrated scores in blue shades
			shade := uint8(180 + i*10)
			box.FillColor = color.RGBA{R: shade, G: shade, B: 255, A: 255}
			box.BoxStyle.Color = color.RGBA{R: 0, G: 0, B: 200, A: 255}
		}
		
		plt.Add(box)
	}
	
	// Set X axis labels
	plt.NominalX(groupNames...)
	
	// Set Y axis range
	plt.Y.Min = 0
	plt.Y.Max = 1
	
	// Add grid
	plt.Add(plotter.NewGrid())
	
	// Add reference line at 0.5
	refLine := plotter.NewFunction(func(x float64) float64 { return 0.5 })
	refLine.Color = color.RGBA{R: 128, G: 128, B: 128, A: 255}
	refLine.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
	plt.Add(refLine)
	
	// Save plot
	outputPath := filepath.Join(p.outputDir, filename)
	if err := plt.Save(10*vg.Inch, 6*vg.Inch, outputPath); err != nil {
		return fmt.Errorf("failed to save plot: %w", err)
	}
	
	return nil
}

// PlotCalibrationComparison creates a comparison chart of metrics
func (p *Plotter) PlotCalibrationComparison(comparisons []framework.CalibrationComparison,
	modelCalibration *framework.CalibrationComparison, filename string) error {
	
	// Create plot
	plt := plot.New()
	plt.Title.Text = "Calibration Method Comparison"
	plt.X.Label.Text = "Calibration Method"
	plt.Y.Label.Text = "Metric Value"
	
	// Prepare data
	methods := []string{}
	prAUCs := plotter.Values{}
	precisions := plotter.Values{}
	recalls := plotter.Values{}
	f1Scores := plotter.Values{}
	
	// Add framework methods
	for _, comp := range comparisons {
		methods = append(methods, comp.Method)
		prAUCs = append(prAUCs, comp.PRCurve.AUC)
		precisions = append(precisions, comp.MetricsAtThreshold["precision"])
		recalls = append(recalls, comp.MetricsAtThreshold["recall"])
		f1Scores = append(f1Scores, comp.MetricsAtThreshold["f1_score"])
	}
	
	// Add model's calibration if provided
	if modelCalibration != nil {
		methods = append(methods, "Model's\n"+modelCalibration.Method)
		prAUCs = append(prAUCs, modelCalibration.PRCurve.AUC)
		precisions = append(precisions, modelCalibration.MetricsAtThreshold["precision"])
		recalls = append(recalls, modelCalibration.MetricsAtThreshold["recall"])
		f1Scores = append(f1Scores, modelCalibration.MetricsAtThreshold["f1_score"])
	}
	
	// Create grouped bar chart
	w := vg.Points(15)
	
	// PR-AUC bars
	prAUCBars, _ := plotter.NewBarChart(prAUCs, w)
	prAUCBars.Color = color.RGBA{R: 31, G: 119, B: 180, A: 255}
	prAUCBars.Offset = -1.5 * w
	
	// Precision bars
	precBars, _ := plotter.NewBarChart(precisions, w)
	precBars.Color = color.RGBA{R: 255, G: 127, B: 14, A: 255}
	precBars.Offset = -0.5 * w
	
	// Recall bars
	recallBars, _ := plotter.NewBarChart(recalls, w)
	recallBars.Color = color.RGBA{R: 44, G: 160, B: 44, A: 255}
	recallBars.Offset = 0.5 * w
	
	// F1 Score bars
	f1Bars, _ := plotter.NewBarChart(f1Scores, w)
	f1Bars.Color = color.RGBA{R: 214, G: 39, B: 40, A: 255}
	f1Bars.Offset = 1.5 * w
	
	// Add bars to plot
	plt.Add(prAUCBars, precBars, recallBars, f1Bars)
	
	// Add legend
	plt.Legend.Add("PR-AUC", prAUCBars)
	plt.Legend.Add("Precision", precBars)
	plt.Legend.Add("Recall", recallBars)
	plt.Legend.Add("F1-Score", f1Bars)
	plt.Legend.Top = true
	plt.Legend.Left = false
	
	// Set X axis labels
	plt.NominalX(methods...)
	
	// Set Y axis range
	plt.Y.Min = 0
	plt.Y.Max = 1
	
	// Add grid
	plt.Add(plotter.NewGrid())
	
	// Save plot
	outputPath := filepath.Join(p.outputDir, filename)
	if err := plt.Save(10*vg.Inch, 6*vg.Inch, outputPath); err != nil {
		return fmt.Errorf("failed to save plot: %w", err)
	}
	
	return nil
}

// PlotScoreHistograms creates histograms for score distributions
func (p *Plotter) PlotScoreHistograms(comparisons []framework.CalibrationComparison,
	rawDist framework.ScoreDistribution, filename string) error {
	
	// Create subplots (2x2 grid for up to 4 methods + raw)
	rows, cols := 2, 3
	plots := make([][]*plot.Plot, rows)
	for i := range plots {
		plots[i] = make([]*plot.Plot, cols)
	}
	
	// Plot raw scores
	plots[0][0] = p.createHistogramPlot(rawDist, "Raw Scores", color.RGBA{R: 200, G: 0, B: 0, A: 255})
	
	// Plot calibrated scores
	plotIdx := 1
	for _, comp := range comparisons {
		row := plotIdx / cols
		col := plotIdx % cols
		if row < rows {
			plots[row][col] = p.createHistogramPlot(comp.ScoreDistribution, 
				fmt.Sprintf("%s Calibration", comp.Method),
				color.RGBA{R: 31, G: 119, B: 180, A: 255})
		}
		plotIdx++
	}
	
	// Create table from plots
	img := vg.NewTableCanvas(plots, 6*vg.Inch, 4*vg.Inch)
	
	// Save plot
	outputPath := filepath.Join(p.outputDir, filename)
	if err := img.Save(outputPath); err != nil {
		return fmt.Errorf("failed to save histogram grid: %w", err)
	}
	
	return nil
}

// createHistogramPlot creates a single histogram plot
func (p *Plotter) createHistogramPlot(dist framework.ScoreDistribution, title string, clr color.Color) *plot.Plot {
	plt := plot.New()
	plt.Title.Text = title
	plt.X.Label.Text = "Score"
	plt.Y.Label.Text = "Frequency"
	
	// Create histogram bars
	if len(dist.Histogram) > 0 {
		bars := make(plotter.Values, len(dist.Histogram))
		for i, bin := range dist.Histogram {
			bars[i] = bin.Ratio
		}
		
		h, err := plotter.NewBarChart(bars, vg.Points(20))
		if err == nil {
			h.Color = clr
			plt.Add(h)
			
			// Set X labels to show bin ranges
			labels := make([]string, len(dist.Histogram))
			for i, bin := range dist.Histogram {
				labels[i] = fmt.Sprintf("%.2f", (bin.Start+bin.End)/2)
			}
			plt.NominalX(labels...)
		}
	}
	
	// Set axis ranges
	plt.X.Min = -0.5
	plt.X.Max = float64(len(dist.Histogram)) - 0.5
	plt.Y.Min = 0
	
	// Add grid
	plt.Add(plotter.NewGrid())
	
	// Add mean line
	if dist.Mean > 0 {
		meanLine := plotter.NewFunction(func(x float64) float64 { return 0 })
		meanLine.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255}
		meanLine.Width = vg.Points(2)
		
		// Convert mean to bin index
		meanBinIdx := dist.Mean * float64(len(dist.Histogram))
		if meanBinIdx >= 0 && meanBinIdx < float64(len(dist.Histogram)) {
			// Add vertical line at mean
			plt.Add(&verticalLine{
				X:     meanBinIdx,
				Color: color.RGBA{R: 255, G: 0, B: 0, A: 255},
				Width: vg.Points(2),
			})
		}
	}
	
	return plt
}

// verticalLine is a custom plotter for drawing vertical lines
type verticalLine struct {
	X     float64
	Color color.Color
	Width vg.Length
}

func (v *verticalLine) Plot(c draw.Canvas, plt *plot.Plot) {
	trX, _ := plt.Transforms(&c)
	x := trX(v.X)
	c.StrokeLine2(draw.LineStyle{Color: v.Color, Width: v.Width},
		x, c.Min.Y, x, c.Max.Y)
}

// PlotCalibrationCurves plots multiple PR curves for comparison
func (p *Plotter) PlotCalibrationCurves(comparisons []framework.CalibrationComparison,
	modelCalibration *framework.CalibrationComparison, filename string) error {
	
	// Create plot
	plt := plot.New()
	plt.Title.Text = "PR Curves by Calibration Method"
	plt.X.Label.Text = "Recall"
	plt.Y.Label.Text = "Precision"
	
	// Colors for different methods
	colors := []color.Color{
		color.RGBA{R: 31, G: 119, B: 180, A: 255},   // blue
		color.RGBA{R: 255, G: 127, B: 14, A: 255},   // orange
		color.RGBA{R: 44, G: 160, B: 44, A: 255},    // green
		color.RGBA{R: 214, G: 39, B: 40, A: 255},    // red
		color.RGBA{R: 148, G: 103, B: 189, A: 255},  // purple
	}
	
	// Plot each calibration method
	for i, comp := range comparisons {
		if comp.PRCurve == nil {
			continue
		}
		
		pts := make(plotter.XYs, len(comp.PRCurve.X))
		for j := range comp.PRCurve.X {
			pts[j].X = comp.PRCurve.X[j]
			pts[j].Y = comp.PRCurve.Y[j]
		}
		
		line, err := plotter.NewLine(pts)
		if err != nil {
			continue
		}
		
		line.Color = colors[i%len(colors)]
		line.Width = vg.Points(2)
		plt.Add(line)
		
		// Add to legend
		plt.Legend.Add(fmt.Sprintf("%s (AUC=%.3f)", comp.Method, comp.PRCurve.AUC), line)
	}
	
	// Add model's calibration if provided
	if modelCalibration != nil && modelCalibration.PRCurve != nil {
		pts := make(plotter.XYs, len(modelCalibration.PRCurve.X))
		for j := range modelCalibration.PRCurve.X {
			pts[j].X = modelCalibration.PRCurve.X[j]
			pts[j].Y = modelCalibration.PRCurve.Y[j]
		}
		
		line, err := plotter.NewLine(pts)
		if err == nil {
			line.Color = color.RGBA{R: 0, G: 0, B: 0, A: 255} // black
			line.Width = vg.Points(2)
			line.Dashes = []vg.Length{vg.Points(5), vg.Points(5)}
			plt.Add(line)
			plt.Legend.Add(fmt.Sprintf("Model's %s (AUC=%.3f)", 
				modelCalibration.Method, modelCalibration.PRCurve.AUC), line)
		}
	}
	
	// Set axis ranges
	plt.X.Min = 0
	plt.X.Max = 1
	plt.Y.Min = 0
	plt.Y.Max = 1
	
	// Add grid
	plt.Add(plotter.NewGrid())
	
	// Position legend
	plt.Legend.Top = false
	plt.Legend.Left = false
	
	// Save plot
	outputPath := filepath.Join(p.outputDir, filename)
	if err := plt.Save(8*vg.Inch, 6*vg.Inch, outputPath); err != nil {
		return fmt.Errorf("failed to save plot: %w", err)
	}
	
	return nil
}