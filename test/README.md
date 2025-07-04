# Test Suite

This directory contains comprehensive tests for the Weighted Naive Bayes Training Framework.

## Test Organization

### Unit Tests
- `cache_test.go` - Tests for prediction caching functionality
- `early_stopping_test.go` - Tests for early stopping mechanism
- `model_extended_test.go` - Tests for extended model interface
- `result_writer_test.go` - Tests for result writing with timestamps
- `validation_test.go` - Tests for input validation and error handling

### Integration Tests
- `integration/spam_test.go` - End-to-end spam detection test

### Benchmarks
- `benchmarks/` - Performance benchmarks (if any)

### Test Data
- `fixtures/test_data.csv` - Sample data for testing

## Running Tests

```bash
# Run all tests
go test ./...

# Run specific test file
go test ./test -v -run TestCache

# Run with coverage
go test -cover ./...

# Run benchmarks
go test -bench=. ./...

# Run integration tests only
go test ./test/integration -v
```

## Test Utilities

The test files include several mock implementations:
- `TestModel` - Basic model for testing
- `SlowModel` - Model with artificial delay for cache testing
- `BadModel` - Model that returns invalid values for validation testing
- `ConstantModel` - Model that returns constant predictions