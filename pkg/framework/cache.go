package framework

import (
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"sync"
	"time"
)

// PredictionCache caches model predictions to avoid redundant computations
type PredictionCache struct {
	cache      map[string]cacheEntry
	mu         sync.RWMutex
	maxSize    int
	ttl        time.Duration
	hits       int64
	misses     int64
	evictions  int64
}

type cacheEntry struct {
	predictions []float64
	timestamp   time.Time
}

// NewPredictionCache creates a new prediction cache
func NewPredictionCache(maxSize int, ttl time.Duration) *PredictionCache {
	return &PredictionCache{
		cache:   make(map[string]cacheEntry),
		maxSize: maxSize,
		ttl:     ttl,
	}
}

// generateKey creates a cache key from model name and samples
func (pc *PredictionCache) generateKey(modelName string, samples [][]float64) string {
	h := md5.New()
	h.Write([]byte(modelName))
	
	// Hash the samples
	for _, sample := range samples {
		for _, val := range sample {
			h.Write([]byte(fmt.Sprintf("%.6f", val)))
		}
	}
	
	return hex.EncodeToString(h.Sum(nil))
}

// Get retrieves predictions from cache if available
func (pc *PredictionCache) Get(modelName string, samples [][]float64) ([]float64, bool) {
	key := pc.generateKey(modelName, samples)
	
	pc.mu.RLock()
	entry, exists := pc.cache[key]
	pc.mu.RUnlock()
	
	if !exists {
		pc.mu.Lock()
		pc.misses++
		pc.mu.Unlock()
		return nil, false
	}
	
	// Check if entry is expired
	if time.Since(entry.timestamp) > pc.ttl {
		pc.mu.Lock()
		delete(pc.cache, key)
		pc.misses++
		pc.mu.Unlock()
		return nil, false
	}
	
	pc.mu.Lock()
	pc.hits++
	pc.mu.Unlock()
	
	// Return a copy to avoid data races
	predictions := make([]float64, len(entry.predictions))
	copy(predictions, entry.predictions)
	
	return predictions, true
}

// Set stores predictions in cache
func (pc *PredictionCache) Set(modelName string, samples [][]float64, predictions []float64) {
	key := pc.generateKey(modelName, samples)
	
	// Create a copy of predictions to store
	predCopy := make([]float64, len(predictions))
	copy(predCopy, predictions)
	
	pc.mu.Lock()
	defer pc.mu.Unlock()
	
	// Check if we need to evict entries
	if len(pc.cache) >= pc.maxSize {
		pc.evictOldest()
	}
	
	pc.cache[key] = cacheEntry{
		predictions: predCopy,
		timestamp:   time.Now(),
	}
}

// evictOldest removes the oldest cache entry
func (pc *PredictionCache) evictOldest() {
	var oldestKey string
	var oldestTime time.Time
	
	for key, entry := range pc.cache {
		if oldestKey == "" || entry.timestamp.Before(oldestTime) {
			oldestKey = key
			oldestTime = entry.timestamp
		}
	}
	
	if oldestKey != "" {
		delete(pc.cache, oldestKey)
		pc.evictions++
	}
}

// Clear removes all entries from cache
func (pc *PredictionCache) Clear() {
	pc.mu.Lock()
	defer pc.mu.Unlock()
	
	pc.cache = make(map[string]cacheEntry)
}

// Stats returns cache statistics
func (pc *PredictionCache) Stats() CacheStats {
	pc.mu.RLock()
	defer pc.mu.RUnlock()
	
	total := pc.hits + pc.misses
	hitRate := 0.0
	if total > 0 {
		hitRate = float64(pc.hits) / float64(total)
	}
	
	return CacheStats{
		Hits:      pc.hits,
		Misses:    pc.misses,
		Evictions: pc.evictions,
		Size:      len(pc.cache),
		HitRate:   hitRate,
	}
}

// CacheStats holds cache performance statistics
type CacheStats struct {
	Hits      int64
	Misses    int64
	Evictions int64
	Size      int
	HitRate   float64
}

// CachedModel wraps a model with caching functionality
type CachedModel struct {
	Model
	cache *PredictionCache
}

// NewCachedModel creates a cached wrapper for a model
func NewCachedModel(model Model, cache *PredictionCache) *CachedModel {
	return &CachedModel{
		Model: model,
		cache: cache,
	}
}

// Predict implements Model interface with caching
func (cm *CachedModel) Predict(samples [][]float64) ([]float64, error) {
	// Check cache first
	if predictions, found := cm.cache.Get(cm.GetName(), samples); found {
		return predictions, nil
	}
	
	// Cache miss - compute predictions
	predictions, err := cm.Model.Predict(samples)
	if err != nil {
		return nil, err
	}
	
	// Store in cache
	cm.cache.Set(cm.GetName(), samples, predictions)
	
	return predictions, nil
}

// CachedEnsemble wraps an ensemble with caching for base models
type CachedEnsemble struct {
	*EnsembleModel
	cache *PredictionCache
}

// NewCachedEnsemble creates a cached ensemble
func NewCachedEnsemble(models []Model, weights []float64, cacheSize int, ttl time.Duration) *CachedEnsemble {
	cache := NewPredictionCache(cacheSize, ttl)
	
	// Wrap each model with caching
	cachedModels := make([]Model, len(models))
	for i, model := range models {
		cachedModels[i] = NewCachedModel(model, cache)
	}
	
	return &CachedEnsemble{
		EnsembleModel: &EnsembleModel{
			Models:  cachedModels,
			Weights: weights,
		},
		cache: cache,
	}
}

// GetCacheStats returns cache statistics
func (ce *CachedEnsemble) GetCacheStats() CacheStats {
	return ce.cache.Stats()
}

// ClearCache clears the prediction cache
func (ce *CachedEnsemble) ClearCache() {
	ce.cache.Clear()
}