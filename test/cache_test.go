package test

import (
	"testing"
	"time"
	
	"github.com/iwonbin/go-nb-weight-training/pkg/framework"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// SlowModel simulates a model with expensive predictions
type SlowModel struct {
	name      string
	delay     time.Duration
	callCount int
}

func (sm *SlowModel) Predict(samples [][]float64) ([]float64, error) {
	sm.callCount++
	time.Sleep(sm.delay)
	
	predictions := make([]float64, len(samples))
	for i := range predictions {
		predictions[i] = 0.5 // Simple constant prediction
	}
	return predictions, nil
}

func (sm *SlowModel) GetName() string {
	return sm.name
}

func TestPredictionCache(t *testing.T) {
	cache := framework.NewPredictionCache(100, 1*time.Minute)
	
	modelName := "test_model"
	samples := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
	predictions := []float64{0.5, 0.6}
	
	// Test cache miss
	cached, found := cache.Get(modelName, samples)
	assert.False(t, found)
	assert.Nil(t, cached)
	
	// Store in cache
	cache.Set(modelName, samples, predictions)
	
	// Test cache hit
	cached, found = cache.Get(modelName, samples)
	assert.True(t, found)
	assert.Equal(t, predictions, cached)
	
	// Test different samples (should be cache miss)
	differentSamples := [][]float64{{0.5, 0.6}, {0.7, 0.8}}
	cached, found = cache.Get(modelName, differentSamples)
	assert.False(t, found)
	assert.Nil(t, cached)
	
	// Check stats
	stats := cache.Stats()
	assert.Equal(t, int64(1), stats.Hits)
	assert.Equal(t, int64(2), stats.Misses)
	assert.Equal(t, 1, stats.Size)
}

func TestCacheExpiration(t *testing.T) {
	cache := framework.NewPredictionCache(100, 50*time.Millisecond)
	
	modelName := "test_model"
	samples := [][]float64{{0.1, 0.2}}
	predictions := []float64{0.5}
	
	// Store in cache
	cache.Set(modelName, samples, predictions)
	
	// Should be in cache
	cached, found := cache.Get(modelName, samples)
	assert.True(t, found)
	assert.Equal(t, predictions, cached)
	
	// Wait for expiration
	time.Sleep(60 * time.Millisecond)
	
	// Should be expired
	cached, found = cache.Get(modelName, samples)
	assert.False(t, found)
	assert.Nil(t, cached)
}

func TestCacheEviction(t *testing.T) {
	cache := framework.NewPredictionCache(2, 1*time.Minute)
	
	// Fill cache to capacity
	cache.Set("model1", [][]float64{{0.1}}, []float64{0.1})
	cache.Set("model2", [][]float64{{0.2}}, []float64{0.2})
	
	stats := cache.Stats()
	assert.Equal(t, 2, stats.Size)
	assert.Equal(t, int64(0), stats.Evictions)
	
	// Add one more (should trigger eviction)
	cache.Set("model3", [][]float64{{0.3}}, []float64{0.3})
	
	stats = cache.Stats()
	assert.Equal(t, 2, stats.Size)
	assert.Equal(t, int64(1), stats.Evictions)
}

func TestCachedModel(t *testing.T) {
	slowModel := &SlowModel{
		name:  "slow_model",
		delay: 10 * time.Millisecond,
	}
	
	cache := framework.NewPredictionCache(100, 1*time.Minute)
	cachedModel := framework.NewCachedModel(slowModel, cache)
	
	samples := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
	
	// First call - should hit the model
	start := time.Now()
	predictions1, err := cachedModel.Predict(samples)
	duration1 := time.Since(start)
	require.NoError(t, err)
	assert.Len(t, predictions1, 2)
	assert.Equal(t, 1, slowModel.callCount)
	
	// Second call - should hit cache
	start = time.Now()
	predictions2, err := cachedModel.Predict(samples)
	duration2 := time.Since(start)
	require.NoError(t, err)
	assert.Equal(t, predictions1, predictions2)
	assert.Equal(t, 1, slowModel.callCount) // No additional calls
	
	// Cache should be much faster
	assert.Less(t, duration2, duration1/2)
	
	// Check cache stats
	stats := cache.Stats()
	assert.Equal(t, int64(1), stats.Hits)
	assert.Equal(t, int64(1), stats.Misses)
	assert.Equal(t, 0.5, stats.HitRate)
}

func TestCachedEnsemble(t *testing.T) {
	// Create slow models
	models := []framework.Model{
		&SlowModel{name: "model1", delay: 10 * time.Millisecond},
		&SlowModel{name: "model2", delay: 10 * time.Millisecond},
	}
	weights := []float64{0.6, 0.4}
	
	// Create cached ensemble
	cachedEnsemble := framework.NewCachedEnsemble(models, weights, 100, 1*time.Minute)
	
	samples := [][]float64{{0.1, 0.2}, {0.3, 0.4}}
	
	// First prediction
	start := time.Now()
	predictions1, err := cachedEnsemble.Predict(samples)
	duration1 := time.Since(start)
	require.NoError(t, err)
	assert.Len(t, predictions1, 2)
	
	// Second prediction (should use cache)
	start = time.Now()
	predictions2, err := cachedEnsemble.Predict(samples)
	duration2 := time.Since(start)
	require.NoError(t, err)
	assert.Equal(t, predictions1, predictions2)
	
	// Should be much faster
	assert.Less(t, duration2, duration1/2)
	
	// Check cache stats
	stats := cachedEnsemble.GetCacheStats()
	assert.Equal(t, int64(2), stats.Hits) // Each model gets a cache hit
	assert.Equal(t, int64(2), stats.Misses) // Each model had initial miss
}

func TestCacheClear(t *testing.T) {
	cache := framework.NewPredictionCache(100, 1*time.Minute)
	
	// Add some entries
	cache.Set("model1", [][]float64{{0.1}}, []float64{0.1})
	cache.Set("model2", [][]float64{{0.2}}, []float64{0.2})
	
	stats := cache.Stats()
	assert.Equal(t, 2, stats.Size)
	
	// Clear cache
	cache.Clear()
	
	stats = cache.Stats()
	assert.Equal(t, 0, stats.Size)
	
	// Previous entries should be gone
	_, found := cache.Get("model1", [][]float64{{0.1}})
	assert.False(t, found)
}

func BenchmarkCachedVsUncachedPrediction(b *testing.B) {
	// Create a model with some computation
	model := &SlowModel{
		name:  "bench_model",
		delay: 100 * time.Microsecond,
	}
	
	cache := framework.NewPredictionCache(1000, 1*time.Minute)
	cachedModel := framework.NewCachedModel(model, cache)
	
	samples := [][]float64{
		{0.1, 0.2, 0.3, 0.4},
		{0.5, 0.6, 0.7, 0.8},
		{0.9, 1.0, 1.1, 1.2},
	}
	
	b.Run("Uncached", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = model.Predict(samples)
		}
	})
	
	b.Run("Cached", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_, _ = cachedModel.Predict(samples)
		}
	})
}