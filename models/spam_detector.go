package models

import (
	"math"
	"math/rand"
	"strings"
)

// SpamDetectorBase is a base spam detector with common functionality
type SpamDetectorBase struct {
	name           string
	spamWords      []string
	hamWords       []string
	threshold      float64
	modelBias      float64 // Each model has different bias/behavior
}

// Common spam and ham indicators
var (
	commonSpamWords = []string{
		"free", "win", "winner", "cash", "prize", "money", "click",
		"buy", "viagra", "pills", "cheap", "discount", "offer",
		"congratulations", "selected", "urgent", "act now", "limited",
		"guarantee", "100%", "risk-free", "cancel", "unsubscribe",
	}
	
	commonHamWords = []string{
		"meeting", "report", "project", "team", "schedule", "please",
		"thanks", "regard", "attach", "document", "review", "discuss",
		"tomorrow", "today", "week", "update", "question", "help",
		"colleague", "work", "office", "lunch", "coffee",
	}
)

// BayesianSpamDetector implements a mock Bayesian spam filter
type BayesianSpamDetector struct {
	SpamDetectorBase
}

func NewBayesianSpamDetector() *BayesianSpamDetector {
	return &BayesianSpamDetector{
		SpamDetectorBase{
			name:      "BayesianFilter",
			spamWords: commonSpamWords,
			hamWords:  commonHamWords,
			threshold: 0.5,
			modelBias: 0.1, // Slightly conservative
		},
	}
}

func (b *BayesianSpamDetector) Predict(samples [][]float64) ([]float64, error) {
	predictions := make([]float64, len(samples))
	
	for i, features := range samples {
		// Features represent email characteristics
		// [spam_word_count, ham_word_count, exclamation_count, caps_ratio, length_score]
		if len(features) < 5 {
			predictions[i] = 0.5 // Default if features are missing
			continue
		}
		
		spamScore := features[0] / 10.0  // Normalized spam word count
		hamScore := features[1] / 10.0    // Normalized ham word count
		exclamations := features[2] / 5.0 // Normalized exclamation count
		capsRatio := features[3]          // Already 0-1
		lengthScore := features[4]        // Already normalized
		
		// Bayesian-style calculation
		spamProbability := (spamScore*0.4 + exclamations*0.2 + capsRatio*0.2 + lengthScore*0.2)
		hamProbability := hamScore * 0.8
		
		// Combine probabilities
		score := spamProbability / (spamProbability + hamProbability + 0.1)
		
		// Apply model bias
		score = score*(1-b.modelBias) + 0.5*b.modelBias
		
		// Ensure in [0,1] range
		predictions[i] = math.Max(0, math.Min(1, score))
	}
	
	return predictions, nil
}

func (b *BayesianSpamDetector) GetName() string {
	return b.name
}

// NeuralNetSpamDetector implements a mock neural network spam detector
type NeuralNetSpamDetector struct {
	SpamDetectorBase
}

func NewNeuralNetSpamDetector() *NeuralNetSpamDetector {
	return &NeuralNetSpamDetector{
		SpamDetectorBase{
			name:      "NeuralNetwork",
			spamWords: commonSpamWords,
			hamWords:  commonHamWords,
			threshold: 0.5,
			modelBias: -0.1, // Slightly aggressive
		},
	}
}

func (n *NeuralNetSpamDetector) Predict(samples [][]float64) ([]float64, error) {
	predictions := make([]float64, len(samples))
	
	for i, features := range samples {
		if len(features) < 5 {
			predictions[i] = 0.5
			continue
		}
		
		// Simulate neural network with non-linear activation
		x1 := features[0]/10.0*0.5 + features[2]/5.0*0.3 + features[3]*0.2
		x2 := features[1]/10.0*(-0.4) + features[4]*0.2
		
		// Hidden layer (tanh activation)
		h1 := math.Tanh(x1*2 - 0.5)
		h2 := math.Tanh(x2*2 + 0.3)
		
		// Output layer (sigmoid)
		output := 1.0 / (1.0 + math.Exp(-(h1*0.7 + h2*0.3 + n.modelBias)))
		
		predictions[i] = output
	}
	
	return predictions, nil
}

func (n *NeuralNetSpamDetector) GetName() string {
	return n.name
}

// SVMSpamDetector implements a mock SVM spam detector
type SVMSpamDetector struct {
	SpamDetectorBase
}

func NewSVMSpamDetector() *SVMSpamDetector {
	return &SVMSpamDetector{
		SpamDetectorBase{
			name:      "SVM_RBF",
			spamWords: commonSpamWords,
			hamWords:  commonHamWords,
			threshold: 0.5,
			modelBias: 0.0, // Balanced
		},
	}
}

func (s *SVMSpamDetector) Predict(samples [][]float64) ([]float64, error) {
	predictions := make([]float64, len(samples))
	
	for i, features := range samples {
		if len(features) < 5 {
			predictions[i] = 0.5
			continue
		}
		
		// Simulate SVM with RBF kernel
		// Create support vector scores
		sv1 := math.Exp(-0.5 * math.Pow(features[0]/10.0-0.8, 2))
		sv2 := math.Exp(-0.5 * math.Pow(features[1]/10.0-0.2, 2))
		sv3 := math.Exp(-0.3 * math.Pow(features[3]-0.7, 2))
		
		// Decision function
		decision := sv1*0.6 - sv2*0.4 + sv3*0.3 + s.modelBias
		
		// Convert to probability
		predictions[i] = 1.0 / (1.0 + math.Exp(-decision*2))
	}
	
	return predictions, nil
}

func (s *SVMSpamDetector) GetName() string {
	return s.name
}

// RandomForestSpamDetector implements a mock Random Forest spam detector
type RandomForestSpamDetector struct {
	SpamDetectorBase
}

func NewRandomForestSpamDetector() *RandomForestSpamDetector {
	return &RandomForestSpamDetector{
		SpamDetectorBase{
			name:      "RandomForest",
			spamWords: commonSpamWords,
			hamWords:  commonHamWords,
			threshold: 0.5,
			modelBias: 0.05, // Slightly conservative
		},
	}
}

func (r *RandomForestSpamDetector) Predict(samples [][]float64) ([]float64, error) {
	predictions := make([]float64, len(samples))
	rng := rand.New(rand.NewSource(42))
	
	for i, features := range samples {
		if len(features) < 5 {
			predictions[i] = 0.5
			continue
		}
		
		// Simulate multiple decision trees
		votes := 0.0
		numTrees := 10
		
		for t := 0; t < numTrees; t++ {
			// Each tree has different feature importance
			w1 := 0.3 + rng.Float64()*0.4
			w2 := 0.1 + rng.Float64()*0.2
			w3 := 0.1 + rng.Float64()*0.3
			w4 := 0.2 + rng.Float64()*0.2
			w5 := 0.1 + rng.Float64()*0.2
			
			// Normalize weights
			sum := w1 + w2 + w3 + w4 + w5
			w1, w2, w3, w4, w5 = w1/sum, w2/sum, w3/sum, w4/sum, w5/sum
			
			treeScore := features[0]/10.0*w1 - features[1]/10.0*w2 + 
						features[2]/5.0*w3 + features[3]*w4 + features[4]*w5
			
			if treeScore > 0.5 {
				votes += 1.0
			}
		}
		
		predictions[i] = (votes/float64(numTrees))*(1-r.modelBias) + 0.5*r.modelBias
	}
	
	return predictions, nil
}

func (r *RandomForestSpamDetector) GetName() string {
	return r.name
}

// LogisticRegressionSpamDetector implements a mock Logistic Regression spam detector
type LogisticRegressionSpamDetector struct {
	SpamDetectorBase
	weights []float64
	bias    float64
}

func NewLogisticRegressionSpamDetector() *LogisticRegressionSpamDetector {
	return &LogisticRegressionSpamDetector{
		SpamDetectorBase: SpamDetectorBase{
			name:      "LogisticRegression",
			spamWords: commonSpamWords,
			hamWords:  commonHamWords,
			threshold: 0.5,
			modelBias: 0.0,
		},
		weights: []float64{0.8, -0.6, 0.4, 0.5, 0.3}, // Feature weights
		bias:    -0.1,
	}
}

func (l *LogisticRegressionSpamDetector) Predict(samples [][]float64) ([]float64, error) {
	predictions := make([]float64, len(samples))
	
	for i, features := range samples {
		if len(features) < 5 {
			predictions[i] = 0.5
			continue
		}
		
		// Linear combination
		z := l.bias
		z += features[0] / 10.0 * l.weights[0]  // spam words
		z += features[1] / 10.0 * l.weights[1]  // ham words
		z += features[2] / 5.0 * l.weights[2]   // exclamations
		z += features[3] * l.weights[3]         // caps ratio
		z += features[4] * l.weights[4]         // length score
		
		// Sigmoid
		predictions[i] = 1.0 / (1.0 + math.Exp(-z))
	}
	
	return predictions, nil
}

func (l *LogisticRegressionSpamDetector) GetName() string {
	return l.name
}

// ExtractEmailFeatures extracts features from email text
// This is a helper function for testing
func ExtractEmailFeatures(emailText string) []float64 {
	text := strings.ToLower(emailText)
	words := strings.Fields(text)
	
	// Count spam words
	spamCount := 0.0
	for _, word := range words {
		for _, spamWord := range commonSpamWords {
			if strings.Contains(word, spamWord) {
				spamCount++
			}
		}
	}
	
	// Count ham words
	hamCount := 0.0
	for _, word := range words {
		for _, hamWord := range commonHamWords {
			if strings.Contains(word, hamWord) {
				hamCount++
			}
		}
	}
	
	// Count exclamations
	exclamationCount := float64(strings.Count(emailText, "!"))
	
	// Calculate caps ratio
	upperCount := 0
	letterCount := 0
	for _, ch := range emailText {
		if ch >= 'A' && ch <= 'Z' {
			upperCount++
			letterCount++
		} else if ch >= 'a' && ch <= 'z' {
			letterCount++
		}
	}
	capsRatio := 0.0
	if letterCount > 0 {
		capsRatio = float64(upperCount) / float64(letterCount)
	}
	
	// Length score (normalized)
	lengthScore := math.Min(float64(len(emailText))/1000.0, 1.0)
	
	return []float64{spamCount, hamCount, exclamationCount, capsRatio, lengthScore}
}