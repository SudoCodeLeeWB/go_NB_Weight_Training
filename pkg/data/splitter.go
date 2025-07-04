package data

import (
	"fmt"
	"math/rand"
	"sort"
)

// Splitter interface for different splitting strategies
type Splitter interface {
	Split(dataset *Dataset) (*Split, error)
}

// ThreeWaySplitter interface for train-calibration-test splits
type ThreeWaySplitter interface {
	SplitThreeWay(dataset *Dataset) (*ThreeWaySplit, error)
}

// CrossValidator interface for cross-validation strategies
type CrossValidator interface {
	GetFolds(dataset *Dataset) ([]Fold, error)
}

// RandomSplitter performs random train-test split
type RandomSplitter struct {
	TestSize   float64
	RandomSeed int64
}

// NewRandomSplitter creates a new random splitter
func NewRandomSplitter(testSize float64, seed int64) *RandomSplitter {
	return &RandomSplitter{
		TestSize:   testSize,
		RandomSeed: seed,
	}
}

// Split implements the Splitter interface
func (rs *RandomSplitter) Split(dataset *Dataset) (*Split, error) {
	if rs.TestSize <= 0 || rs.TestSize >= 1 {
		return nil, fmt.Errorf("test_size must be between 0 and 1")
	}
	
	rng := rand.New(rand.NewSource(rs.RandomSeed))
	
	// Create shuffled indices
	indices := make([]int, dataset.NumSamples)
	for i := range indices {
		indices[i] = i
	}
	
	// Shuffle
	rng.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})
	
	// Split indices
	testSize := int(float64(dataset.NumSamples) * rs.TestSize)
	trainIndices := indices[testSize:]
	testIndices := indices[:testSize]
	
	return &Split{
		Train: dataset.Subset(trainIndices),
		Test:  dataset.Subset(testIndices),
	}, nil
}

// StratifiedSplitter performs stratified train-test split
type StratifiedSplitter struct {
	TestSize   float64
	RandomSeed int64
}

// NewStratifiedSplitter creates a new stratified splitter
func NewStratifiedSplitter(testSize float64, seed int64) *StratifiedSplitter {
	return &StratifiedSplitter{
		TestSize:   testSize,
		RandomSeed: seed,
	}
}

// Split implements the Splitter interface with stratification
func (ss *StratifiedSplitter) Split(dataset *Dataset) (*Split, error) {
	if ss.TestSize <= 0 || ss.TestSize >= 1 {
		return nil, fmt.Errorf("test_size must be between 0 and 1")
	}
	
	rng := rand.New(rand.NewSource(ss.RandomSeed))
	
	// Group indices by class
	classIndices := make(map[float64][]int)
	for i, sample := range dataset.Samples {
		classIndices[sample.Label] = append(classIndices[sample.Label], i)
	}
	
	trainIndices := []int{}
	testIndices := []int{}
	
	// Split each class proportionally
	for class, indices := range classIndices {
		// Shuffle indices for this class
		shuffled := make([]int, len(indices))
		copy(shuffled, indices)
		rng.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})
		
		// Calculate split point
		testSize := int(float64(len(shuffled)) * ss.TestSize)
		if testSize == 0 && len(shuffled) > 0 {
			testSize = 1 // Ensure at least one sample in test if class has samples
		}
		
		// Split
		testIndices = append(testIndices, shuffled[:testSize]...)
		trainIndices = append(trainIndices, shuffled[testSize:]...)
		
		_ = class // Use class variable
	}
	
	// Sort indices for consistent ordering
	sort.Ints(trainIndices)
	sort.Ints(testIndices)
	
	return &Split{
		Train: dataset.Subset(trainIndices),
		Test:  dataset.Subset(testIndices),
	}, nil
}

// KFoldCV performs k-fold cross-validation
type KFoldCV struct {
	K          int
	Shuffle    bool
	RandomSeed int64
}

// NewKFoldCV creates a new k-fold cross-validator
func NewKFoldCV(k int, shuffle bool, seed int64) *KFoldCV {
	return &KFoldCV{
		K:          k,
		Shuffle:    shuffle,
		RandomSeed: seed,
	}
}

// GetFolds implements the CrossValidator interface
func (kf *KFoldCV) GetFolds(dataset *Dataset) ([]Fold, error) {
	if kf.K < 2 {
		return nil, fmt.Errorf("k must be at least 2")
	}
	
	if kf.K > dataset.NumSamples {
		return nil, fmt.Errorf("k cannot be greater than number of samples")
	}
	
	// Create indices
	indices := make([]int, dataset.NumSamples)
	for i := range indices {
		indices[i] = i
	}
	
	// Shuffle if requested
	if kf.Shuffle {
		rng := rand.New(rand.NewSource(kf.RandomSeed))
		rng.Shuffle(len(indices), func(i, j int) {
			indices[i], indices[j] = indices[j], indices[i]
		})
	}
	
	// Create folds
	folds := make([]Fold, kf.K)
	foldSize := dataset.NumSamples / kf.K
	remainder := dataset.NumSamples % kf.K
	
	start := 0
	for i := 0; i < kf.K; i++ {
		// Calculate fold size (distribute remainder)
		size := foldSize
		if i < remainder {
			size++
		}
		
		// Test indices for this fold
		testIndices := indices[start : start+size]
		
		// Train indices (all except test)
		trainIndices := make([]int, 0, dataset.NumSamples-size)
		trainIndices = append(trainIndices, indices[:start]...)
		trainIndices = append(trainIndices, indices[start+size:]...)
		
		folds[i] = Fold{
			TrainIndices: trainIndices,
			TestIndices:  testIndices,
		}
		
		start += size
	}
	
	return folds, nil
}

// StratifiedKFoldCV performs stratified k-fold cross-validation
type StratifiedKFoldCV struct {
	K          int
	Shuffle    bool
	RandomSeed int64
}

// NewStratifiedKFoldCV creates a new stratified k-fold cross-validator
func NewStratifiedKFoldCV(k int, shuffle bool, seed int64) *StratifiedKFoldCV {
	return &StratifiedKFoldCV{
		K:          k,
		Shuffle:    shuffle,
		RandomSeed: seed,
	}
}

// GetFolds implements the CrossValidator interface with stratification
func (skf *StratifiedKFoldCV) GetFolds(dataset *Dataset) ([]Fold, error) {
	if skf.K < 2 {
		return nil, fmt.Errorf("k must be at least 2")
	}
	
	// Group indices by class
	classIndices := make(map[float64][]int)
	for i, sample := range dataset.Samples {
		classIndices[sample.Label] = append(classIndices[sample.Label], i)
	}
	
	// Check if we have enough samples per class
	for class, indices := range classIndices {
		if len(indices) < skf.K {
			return nil, fmt.Errorf("class %.0f has only %d samples, need at least %d for %d-fold CV", 
				class, len(indices), skf.K, skf.K)
		}
	}
	
	// Shuffle indices within each class if requested
	if skf.Shuffle {
		rng := rand.New(rand.NewSource(skf.RandomSeed))
		for class, indices := range classIndices {
			shuffled := make([]int, len(indices))
			copy(shuffled, indices)
			rng.Shuffle(len(shuffled), func(i, j int) {
				shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
			})
			classIndices[class] = shuffled
		}
	}
	
	// Create folds
	folds := make([]Fold, skf.K)
	for i := 0; i < skf.K; i++ {
		folds[i] = Fold{
			TrainIndices: []int{},
			TestIndices:  []int{},
		}
	}
	
	// Distribute each class across folds
	for _, indices := range classIndices {
		// Distribute this class's samples across folds
		for i, idx := range indices {
			foldIdx := i % skf.K
			folds[foldIdx].TestIndices = append(folds[foldIdx].TestIndices, idx)
			
			// Add to train indices of other folds
			for j := 0; j < skf.K; j++ {
				if j != foldIdx {
					folds[j].TrainIndices = append(folds[j].TrainIndices, idx)
				}
			}
		}
	}
	
	// Sort indices for consistent ordering
	for i := range folds {
		sort.Ints(folds[i].TrainIndices)
		sort.Ints(folds[i].TestIndices)
	}
	
	return folds, nil
}

// ThreeWayRandomSplitter performs random train-calibration-test split
type ThreeWayRandomSplitter struct {
	CalibrationSize float64 // Size for calibration set
	TestSize        float64 // Size for test set
	RandomSeed      int64
}

// NewThreeWayRandomSplitter creates a new three-way splitter
// The training size will be 1 - calibrationSize - testSize
func NewThreeWayRandomSplitter(calibrationSize, testSize float64, seed int64) *ThreeWayRandomSplitter {
	return &ThreeWayRandomSplitter{
		CalibrationSize: calibrationSize,
		TestSize:        testSize,
		RandomSeed:      seed,
	}
}

// SplitThreeWay implements the ThreeWaySplitter interface
func (trs *ThreeWayRandomSplitter) SplitThreeWay(dataset *Dataset) (*ThreeWaySplit, error) {
	totalTestSize := trs.CalibrationSize + trs.TestSize
	if totalTestSize <= 0 || totalTestSize >= 1 {
		return nil, fmt.Errorf("calibration_size + test_size must be between 0 and 1")
	}
	
	rng := rand.New(rand.NewSource(trs.RandomSeed))
	
	// Create shuffled indices
	indices := make([]int, dataset.NumSamples)
	for i := range indices {
		indices[i] = i
	}
	
	// Shuffle
	rng.Shuffle(len(indices), func(i, j int) {
		indices[i], indices[j] = indices[j], indices[i]
	})
	
	// Calculate split points
	calibrationSize := int(float64(dataset.NumSamples) * trs.CalibrationSize)
	testSize := int(float64(dataset.NumSamples) * trs.TestSize)
	
	// Split indices
	calibrationIndices := indices[:calibrationSize]
	testIndices := indices[calibrationSize : calibrationSize+testSize]
	trainIndices := indices[calibrationSize+testSize:]
	
	return &ThreeWaySplit{
		Train:       dataset.Subset(trainIndices),
		Calibration: dataset.Subset(calibrationIndices),
		Test:        dataset.Subset(testIndices),
	}, nil
}

// ThreeWayStratifiedSplitter performs stratified train-calibration-test split
type ThreeWayStratifiedSplitter struct {
	CalibrationSize float64
	TestSize        float64
	RandomSeed      int64
}

// NewThreeWayStratifiedSplitter creates a new stratified three-way splitter
func NewThreeWayStratifiedSplitter(calibrationSize, testSize float64, seed int64) *ThreeWayStratifiedSplitter {
	return &ThreeWayStratifiedSplitter{
		CalibrationSize: calibrationSize,
		TestSize:        testSize,
		RandomSeed:      seed,
	}
}

// SplitThreeWay implements the ThreeWaySplitter interface with stratification
func (tss *ThreeWayStratifiedSplitter) SplitThreeWay(dataset *Dataset) (*ThreeWaySplit, error) {
	totalTestSize := tss.CalibrationSize + tss.TestSize
	if totalTestSize <= 0 || totalTestSize >= 1 {
		return nil, fmt.Errorf("calibration_size + test_size must be between 0 and 1")
	}
	
	rng := rand.New(rand.NewSource(tss.RandomSeed))
	
	// Group indices by class
	classIndices := make(map[float64][]int)
	for i, sample := range dataset.Samples {
		classIndices[sample.Label] = append(classIndices[sample.Label], i)
	}
	
	trainIndices := []int{}
	calibrationIndices := []int{}
	testIndices := []int{}
	
	// Split each class proportionally
	for _, indices := range classIndices {
		// Shuffle indices for this class
		shuffled := make([]int, len(indices))
		copy(shuffled, indices)
		rng.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})
		
		// Calculate split points
		calibSize := int(float64(len(shuffled)) * tss.CalibrationSize)
		testSize := int(float64(len(shuffled)) * tss.TestSize)
		
		// Ensure at least one sample in each set if possible
		if calibSize == 0 && len(shuffled) >= 3 {
			calibSize = 1
		}
		if testSize == 0 && len(shuffled) >= 3 {
			testSize = 1
		}
		
		// Split
		calibrationIndices = append(calibrationIndices, shuffled[:calibSize]...)
		testIndices = append(testIndices, shuffled[calibSize:calibSize+testSize]...)
		trainIndices = append(trainIndices, shuffled[calibSize+testSize:]...)
	}
	
	// Sort indices for consistent ordering
	sort.Ints(trainIndices)
	sort.Ints(calibrationIndices)
	sort.Ints(testIndices)
	
	return &ThreeWaySplit{
		Train:       dataset.Subset(trainIndices),
		Calibration: dataset.Subset(calibrationIndices),
		Test:        dataset.Subset(testIndices),
	}, nil
}