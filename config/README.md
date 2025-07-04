# Configuration Files

This directory contains example configuration files for the Weighted Naive Bayes Training Framework.

## Available Configurations

### 1. `default_config.json`
- Standard configuration with balanced settings
- 5-fold cross-validation
- 100 epochs with early stopping (patience=10)
- Suitable for most use cases

### 2. `quick_training.json`
- Fast training for rapid prototyping
- Simple train/validation split (no cross-validation)
- 50 epochs with aggressive early stopping (patience=5)
- Smaller population size for faster optimization
- Use this for quick experiments

### 3. `production_config.json`
- Robust configuration for production models
- 10-fold cross-validation for reliable evaluation
- 200 epochs with conservative early stopping (patience=20)
- Larger population size for thorough optimization
- High-quality visualizations

## How to Use

1. **From command line:**
   ```bash
   go run cmd/train/main.go -data your_data.csv -config config/quick_training.json
   ```

2. **In your code:**
   ```go
   config, err := framework.LoadConfig("config/production_config.json")
   if err != nil {
       log.Fatal(err)
   }
   ```

3. **Customize for your needs:**
   - Copy one of these files
   - Modify the parameters
   - Save with a descriptive name

## Key Parameters to Adjust

- **`patience`**: Number of epochs without improvement before stopping (5-20)
- **`k_folds`**: Set to 1 for simple split, or 5-10 for cross-validation
- **`population_size`**: Larger = better optimization but slower (20-100)
- **`max_epochs`**: Maximum training iterations (50-200)
- **`validation_split`**: Fraction of data for validation (0.2-0.3)

## Early Stopping

The `patience` parameter controls early stopping:
- `patience: 5` = Stop after 5 epochs with no improvement
- `patience: 10` = Stop after 10 epochs with no improvement
- Set higher for more conservative stopping, lower for faster training