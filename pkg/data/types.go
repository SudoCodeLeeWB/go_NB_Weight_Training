package data

// Sample represents a single data sample
type Sample struct {
	Features []float64 // Feature vector (predictions from base classifiers)
	Label    float64   // Binary label (0 or 1)
	ID       string    // Optional sample identifier
}

// Dataset represents a collection of samples
type Dataset struct {
	Samples []Sample
	// Metadata
	NumFeatures int
	NumSamples  int
	ClassCounts map[float64]int // Count of samples per class
}

// Split represents a train-test split
type Split struct {
	Train *Dataset
	Test  *Dataset
}

// Fold represents a cross-validation fold
type Fold struct {
	TrainIndices []int
	TestIndices  []int
}

// NewDataset creates a new dataset from samples
func NewDataset(samples []Sample) *Dataset {
	if len(samples) == 0 {
		return &Dataset{
			Samples:     samples,
			NumFeatures: 0,
			NumSamples:  0,
			ClassCounts: make(map[float64]int),
		}
	}
	
	// Calculate metadata
	numFeatures := len(samples[0].Features)
	classCounts := make(map[float64]int)
	
	for _, sample := range samples {
		classCounts[sample.Label]++
	}
	
	return &Dataset{
		Samples:     samples,
		NumFeatures: numFeatures,
		NumSamples:  len(samples),
		ClassCounts: classCounts,
	}
}

// GetFeatures returns feature matrix
func (d *Dataset) GetFeatures() [][]float64 {
	features := make([][]float64, d.NumSamples)
	for i, sample := range d.Samples {
		features[i] = sample.Features
	}
	return features
}

// GetLabels returns label vector
func (d *Dataset) GetLabels() []float64 {
	labels := make([]float64, d.NumSamples)
	for i, sample := range d.Samples {
		labels[i] = sample.Label
	}
	return labels
}

// Subset creates a new dataset from indices
func (d *Dataset) Subset(indices []int) *Dataset {
	samples := make([]Sample, len(indices))
	for i, idx := range indices {
		samples[i] = d.Samples[idx]
	}
	return NewDataset(samples)
}

// ClassBalance returns the proportion of positive samples
func (d *Dataset) ClassBalance() float64 {
	if d.NumSamples == 0 {
		return 0.0
	}
	positives := d.ClassCounts[1.0]
	return float64(positives) / float64(d.NumSamples)
}