package generators

import (
	"math"
	"math/rand"
	
	"github.com/iwonbin/go-nb-weight-training/pkg/data"
)

// ComplexDatasetConfig configures the complex dataset generation
type ComplexDatasetConfig struct {
	NumSamples          int
	NumFeatures         int
	NoiseLevel         float64 // 0-1, subtle noise level
	ClassImbalance     float64 // ratio of positive to negative (e.g., 0.3 = 30% positive)
	FeatureCorrelation float64 // correlation between features
	Nonlinearity       float64 // degree of nonlinear patterns
	TemporalDrift      bool    // whether to include temporal patterns
	HiddenGroups       int     // number of hidden subgroups
	RandomSeed         int64
}

// GenerateComplexDataset creates a realistic, challenging dataset
func GenerateComplexDataset(config ComplexDatasetConfig) *data.Dataset {
	rng := rand.New(rand.NewSource(config.RandomSeed))
	samples := make([]data.Sample, config.NumSamples)
	
	// Calculate class distribution
	numPositive := int(float64(config.NumSamples) * config.ClassImbalance)
	numNegative := config.NumSamples - numPositive
	
	// Generate base patterns for different hidden groups
	patterns := generateHiddenPatterns(config.HiddenGroups, config.NumFeatures, rng)
	
	// Generate positive samples
	for i := 0; i < numPositive; i++ {
		features := generateComplexFeatures(
			1.0, // positive class
			i,
			config,
			patterns,
			rng,
		)
		samples[i] = data.Sample{
			Features: features,
			Label:    1,
		}
	}
	
	// Generate negative samples
	for i := 0; i < numNegative; i++ {
		features := generateComplexFeatures(
			0.0, // negative class
			i + numPositive,
			config,
			patterns,
			rng,
		)
		samples[i+numPositive] = data.Sample{
			Features: features,
			Label:    0,
		}
	}
	
	// Shuffle to avoid ordering bias
	rng.Shuffle(len(samples), func(i, j int) {
		samples[i], samples[j] = samples[j], samples[i]
	})
	
	return data.NewDataset(samples)
}

// generateComplexFeatures creates features with realistic complexity
func generateComplexFeatures(
	label float64,
	sampleIndex int,
	config ComplexDatasetConfig,
	patterns [][]float64,
	rng *rand.Rand,
) []float64 {
	features := make([]float64, config.NumFeatures)
	
	// Select hidden group
	groupIdx := rng.Intn(config.HiddenGroups)
	basePattern := patterns[groupIdx]
	
	// Base signal strength varies by class
	signalStrength := 0.3 + 0.4*label + rng.NormFloat64()*0.1
	
	// Generate correlated features
	prev := rng.NormFloat64()
	for j := 0; j < config.NumFeatures; j++ {
		// Base value from pattern
		base := basePattern[j] * signalStrength
		
		// Add correlation with previous features
		if j > 0 {
			base += config.FeatureCorrelation * prev * (0.5 + 0.5*label)
		}
		
		// Add nonlinear interactions
		if j > 1 {
			interaction := math.Sin(features[j-1]*features[j-2]) * config.Nonlinearity
			base += interaction * (0.3 + 0.4*label)
		}
		
		// Add temporal drift if enabled
		if config.TemporalDrift {
			drift := float64(sampleIndex) / float64(config.NumSamples)
			base += drift * 0.2 * math.Sin(float64(j))
		}
		
		// Add subtle noise
		noise := rng.NormFloat64() * config.NoiseLevel
		
		// Some features are more discriminative than others
		discriminativeness := math.Exp(-float64(j) / 10.0)
		classSignal := (label - 0.5) * discriminativeness * 0.5
		
		features[j] = base + classSignal + noise
		prev = features[j]
		
		// Apply sigmoid squashing to keep values reasonable
		features[j] = sigmoid(features[j])
	}
	
	// Add some outlier features occasionally
	if rng.Float64() < 0.05 { // 5% outliers
		outlierIdx := rng.Intn(config.NumFeatures)
		features[outlierIdx] = rng.Float64()
	}
	
	return features
}

// generateHiddenPatterns creates base patterns for hidden groups
func generateHiddenPatterns(numGroups, numFeatures int, rng *rand.Rand) [][]float64 {
	patterns := make([][]float64, numGroups)
	
	for i := 0; i < numGroups; i++ {
		pattern := make([]float64, numFeatures)
		
		// Create sinusoidal base pattern with random phase
		phase := rng.Float64() * 2 * math.Pi
		frequency := 1.0 + rng.Float64()*3.0
		
		for j := 0; j < numFeatures; j++ {
			// Combine multiple frequencies for complexity
			pattern[j] = 0.5 * math.Sin(frequency*float64(j)/float64(numFeatures)+phase)
			pattern[j] += 0.3 * math.Cos(2*frequency*float64(j)/float64(numFeatures))
			pattern[j] += rng.NormFloat64() * 0.2
		}
		
		patterns[i] = pattern
	}
	
	return patterns
}

// sigmoid applies sigmoid function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// GenerateDefaultComplexDataset creates a default challenging dataset
func GenerateDefaultComplexDataset() *data.Dataset {
	return GenerateComplexDataset(ComplexDatasetConfig{
		NumSamples:         5000,
		NumFeatures:        20,
		NoiseLevel:         0.15,
		ClassImbalance:     0.35, // 35% positive class
		FeatureCorrelation: 0.4,
		Nonlinearity:       0.3,
		TemporalDrift:      true,
		HiddenGroups:       5,
		RandomSeed:         42,
	})
}

// DatasetCharacteristics analyzes dataset properties
type DatasetCharacteristics struct {
	NumSamples       int
	NumFeatures      int
	ClassBalance     float64
	FeatureRanges    [][]float64 // min, max for each feature
	CorrelationMatrix [][]float64
	Separability     float64 // estimated class separability
}

// AnalyzeDataset computes dataset characteristics
func AnalyzeDataset(dataset *data.Dataset) DatasetCharacteristics {
	features := dataset.GetFeatures()
	labels := dataset.GetLabels()
	
	numSamples := len(labels)
	numFeatures := len(features[0])
	
	// Calculate class balance
	positiveCount := 0.0
	for _, label := range labels {
		positiveCount += label
	}
	classBalance := positiveCount / float64(numSamples)
	
	// Calculate feature ranges
	featureRanges := make([][]float64, numFeatures)
	for j := 0; j < numFeatures; j++ {
		min, max := features[0][j], features[0][j]
		for i := 1; i < numSamples; i++ {
			if features[i][j] < min {
				min = features[i][j]
			}
			if features[i][j] > max {
				max = features[i][j]
			}
		}
		featureRanges[j] = []float64{min, max}
	}
	
	// Estimate separability using simple distance metric
	var posMean, negMean []float64
	posCount, negCount := 0, 0
	
	posMean = make([]float64, numFeatures)
	negMean = make([]float64, numFeatures)
	
	for i, label := range labels {
		if label > 0.5 {
			posCount++
			for j := 0; j < numFeatures; j++ {
				posMean[j] += features[i][j]
			}
		} else {
			negCount++
			for j := 0; j < numFeatures; j++ {
				negMean[j] += features[i][j]
			}
		}
	}
	
	// Normalize means
	for j := 0; j < numFeatures; j++ {
		if posCount > 0 {
			posMean[j] /= float64(posCount)
		}
		if negCount > 0 {
			negMean[j] /= float64(negCount)
		}
	}
	
	// Calculate separability as distance between class means
	separability := 0.0
	for j := 0; j < numFeatures; j++ {
		diff := posMean[j] - negMean[j]
		separability += diff * diff
	}
	separability = math.Sqrt(separability)
	
	return DatasetCharacteristics{
		NumSamples:    numSamples,
		NumFeatures:   numFeatures,
		ClassBalance:  classBalance,
		FeatureRanges: featureRanges,
		Separability:  separability,
	}
}