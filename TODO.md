# TODO List

## High Priority Tasks 🔴

### Input Data Clarification
- [ ] Clarify input data format and requirements
  - Document what the framework is actually looking for in input files
  - Explain difference between raw features vs classifier predictions
  - Add clear examples of expected CSV format
  - Update examples to show realistic input data structure
  - Add input data validation with clear error messages

### Output Location Configuration
- [ ] Add configurable output location specification
  - Allow users to specify custom output directories via CLI flags
  - Support environment variables for default output paths
  - Add output path configuration in JSON config files
  - Ensure all examples respect custom output locations
  - Update documentation with output path configuration examples

## Completed ✅
- [x] Fix early stopping to restore best weights when training stops early
- [x] Add dedicated PredictSingle() method for efficient single predictions
- [x] Handle division by zero in metrics calculations
- [x] Add comprehensive input validation and error handling
- [x] Add caching layer for repeated predictions
- [x] Create clear model integration guide in README
- [x] Add timestamp-based output directories
- [x] Create config directory with example configurations

## Future Enhancements (Optional)

### Performance
- [ ] Add GPU support for large-scale training
- [ ] Implement multi-threaded optimization
- [ ] Add sparse data support for memory efficiency

### Features
- [ ] Add more calibration methods (isotonic regression)
- [ ] Implement confidence intervals for predictions
- [ ] Add feature importance/explainability (SHAP values)
- [ ] Support for multi-class classification (not just binary)

### Data Handling
- [ ] Add automated missing value handling
- [ ] Implement feature preprocessing pipeline
- [ ] Add data drift detection for model monitoring

### Developer Experience
- [ ] Create more example models (SVM, Random Forest wrappers)
- [ ] Add Jupyter notebook examples
- [ ] Create model comparison tools
- [ ] Add cross-validation visualization

### Testing
- [ ] Add more integration test scenarios
- [ ] Create performance regression tests
- [ ] Add property-based testing

## Notes
This framework is designed for local use and research. Features like API servers, monitoring, and cloud deployment have been intentionally excluded to keep the codebase focused and maintainable.