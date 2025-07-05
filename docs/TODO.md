# TODO List

## 🔴 High Priority

### Framework Architecture
- [ ] Consider removing old Model interface and trainer once migration is complete
- [ ] Remove deprecated calibration fields from TrainingConfig in next major version
- [ ] Create new test suite using AggregatedModel interface

### Documentation
- [ ] Add API documentation (godoc)
- [ ] Create deployment guide
- [ ] Add troubleshooting guide
- [ ] Create performance benchmarks documentation

## 🟡 Medium Priority

### Performance Optimization
- [ ] Add caching for expensive operations in cross-validation
- [ ] Implement parallel optimization for Differential Evolution
- [ ] Add GPU support for large-scale predictions
- [ ] Optimize memory usage for very large datasets

### Features
- [ ] Add confidence intervals for predictions
- [ ] Implement SHAP values for model explainability
- [ ] Support for multi-class classification (not just binary)
- [ ] Add model versioning support
- [ ] Implement A/B testing framework

### Data Handling
- [ ] Add automated missing value handling strategies
- [ ] Implement feature preprocessing pipeline
- [ ] Add data drift detection for production monitoring
- [ ] Support for sparse data formats

## 🟢 Low Priority

### Developer Experience
- [ ] Create more example models (XGBoost, LightGBM wrappers)
- [ ] Add Jupyter notebook examples
- [ ] Create model comparison visualization tools
- [ ] Add cross-validation progress visualization
- [ ] Create a web UI for model training

### Testing & Quality
- [ ] Add property-based testing with quick/check
- [ ] Create performance regression test suite
- [ ] Add mutation testing
- [ ] Increase test coverage to >90%

### Integrations
- [ ] Add MLflow integration for experiment tracking
- [ ] Support for ONNX model format
- [ ] Integration with popular Go ML libraries
- [ ] Docker containerization

## ✅ Recently Completed

### Framework Improvements
- [x] Implement CalibratedAggregatedModel interface for calibration comparison
- [x] Fix early stopping to properly restore best weights
- [x] Add modular model loading system
- [x] Clean up outdated code and documentation
- [x] Add timestamp-based output directories
- [x] Create comprehensive configuration examples

### Production Readiness
- [x] Add model persistence (save/load)
- [x] Implement probability calibration methods
- [x] Add input validation with clear error messages
- [x] Implement memory-efficient batch processing
- [x] Add prediction caching layer
- [x] Handle division by zero in metrics

### Documentation
- [x] Create clear model integration guide
- [x] Document calibration comparison feature
- [x] Update all examples to use new architecture
- [x] Consolidate documentation into README, THEORY, and TODO

## 💡 Ideas for Future

### Research & Experimentation
- [ ] Experiment with other gradient-free optimizers (PSO, Genetic Algorithms)
- [ ] Research adaptive weight bounds during optimization
- [ ] Investigate online learning capabilities
- [ ] Explore federated learning support

### Enterprise Features
- [ ] Add audit logging for compliance
- [ ] Implement role-based access control
- [ ] Add encryption for sensitive model data
- [ ] Create enterprise deployment templates

## 📝 Notes

- This framework is designed for local use and research
- API servers and cloud deployment were intentionally excluded
- Focus is on providing a solid, maintainable core for ensemble weight optimization
- Community contributions are welcome for additional features