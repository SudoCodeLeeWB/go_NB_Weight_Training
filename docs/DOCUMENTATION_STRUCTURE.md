# Documentation Structure

The documentation has been reorganized into a clean, three-part structure with supporting guides.

## Main Documentation Files

### 1. README.md
**Purpose**: Primary entry point and user guide
- Quick start guide
- Installation instructions
- Key features overview
- Example implementations
- Configuration guide
- Testing instructions

### 2. THEORY.md
**Purpose**: Theoretical foundations and algorithm explanations
- Differential Evolution algorithm
- PR-AUC optimization rationale
- Weighted Naive Bayes multiplication
- Probability calibration methods
- Three-way data split explanation
- Gradient-free optimization concepts

### 3. TODO.md
**Purpose**: Development roadmap and task tracking
- High priority tasks
- Medium priority enhancements
- Low priority nice-to-haves
- Recently completed features
- Future research ideas
- Known issues and limitations

## Supporting Documentation

### CALIBRATION_COMPARISON.md
**Purpose**: Detailed guide for the calibration comparison feature
- How calibration works
- CalibratedAggregatedModel interface
- Comparison methodology
- Interpretation of results

### MODULAR_USAGE.md
**Purpose**: Guide for using the modular model system
- Directory structure
- Creating custom models
- Using the wrapper system
- Dataset formats

### CLAUDE.md
**Purpose**: AI assistant instructions (for Claude)
- Project overview
- Key commands
- Architecture notes
- Current state information

## Removed Files
- `PRODUCTION_ISSUES.md` - Content merged into TODO.md
- `PROJECT_STRUCTURE.md` - Outdated, structure changed
- `CLEANUP_SUMMARY.md` - Temporary file, no longer needed
- `theory.md` - Renamed to THEORY.md for consistency

## Benefits of New Structure
1. **Clear hierarchy**: Three main files cover usage, theory, and future work
2. **No redundancy**: Each topic has one authoritative location
3. **Easy navigation**: Users know where to find information
4. **Maintainable**: Fewer files to keep updated
5. **Focused content**: Each file has a specific purpose