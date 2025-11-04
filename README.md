# CropMOSAIKS Crop Modeling

This repository contains the complete workflow for predicting crop yields in Zambia using Random Convolutional Features (RCFs) extracted from satellite imagery. It is part of the [CropMOSAIKS project](https://github.com/cropmosaiks), which extends the [MOSAIKS approach](https://www.nature.com/articles/s41467-021-24638-z) to agricultural applications.

## Overview

The CropMOSAIKS crop modeling workflow consists of three main phases:

1. **Grid Creation**: Generate uniform grids of points over regions of interest
2. **Feature Extraction**: Extract Random Convolutional Features from satellite imagery at grid points
3. **Task Modeling**: Train predictive models using features and crop yield data

This repository integrates all phases of the workflow, from creating spatial grids to training models and generating predictions. The approach uses machine learning with satellite-derived features to predict crop yields at the district level in Zambia.

## Key Features

- **Satellite Data Processing**: Works with Landsat 8 and Sentinel-2 imagery from Microsoft's Planetary Computer
- **Random Convolutional Features**: Encodes geospatial locations with information from satellite imagery (colors, textures, edges)
- **Flexible Modeling**: Task-agnostic features can predict various outcomes beyond crop yields (forest cover, population, income, etc.)
- **Temporal Analysis**: Processes imagery across multiple time periods for longitudinal modeling
- **Statistical Validation**: Comprehensive model evaluation with multiple R² metrics and residual analysis

## Repository Structure

```
crop-modeling/
├── code/
│   ├── 1_grid_creation/       # Create spatial grids over regions of interest
│   ├── 2_feature_extraction/  # Extract RCFs from satellite imagery
│   ├── 3_task_modeling/        # Train and evaluate predictive models
│   ├── 4_results/              # Results analysis and visualization
│   └── 5_figures/              # Figure generation scripts
├── data/                       # Data folders (see data/README.md)
│   ├── crop_yield/             # District-level crop yield data
│   ├── geo_boundaries/         # Administrative boundaries
│   ├── land_cover/             # Cropland coverage rasters
│   └── random_features/        # Extracted RCF features by satellite
├── environment/                # Environment configuration files
├── figures/                    # Generated figures and visualizations
├── models/                     # Trained RCF models
└── slurm/                      # SLURM batch scripts for HPC

```

## Workflow

### 1. Grid Creation (`code/1_grid_creation/`)

Create a uniform grid of points at 0.01-degree resolution (~1 km²) over the region of interest:

- **`dense_grid.ipynb`**: Generate grids from country boundaries or custom geometries
- **`land_cover_9_class.ipynb`**: Extract land cover percentages at grid points
- **`district_ndvi.ipynb`**: Calculate NDVI at district level

Options:
- **Equal angle grids** (EPSG 4326): 0.01° × 0.01° cells
- **Equal area grids**: Local coordinate reference system (e.g., Zambia EPSG)

### 2. Feature Extraction (`code/2_feature_extraction/`)

Extract Random Convolutional Features from satellite imagery at each grid point:

**Primary Notebooks:**
- **`s-2_ls-c2-l2_multiband.ipynb`**: Extract features from Landsat 8 or Sentinel-2 (multiple bands)
- **`s-2_rgb-only.ipynb`**: Extract features from Sentinel-2 (RGB only, faster processing)

**Configuration Options:**
- Select satellite collection (Landsat 8 or Sentinel-2)
- Number of features (default: 1000)
- Spectral bands to include
- Time period for featurization
- Cloud cover threshold (default: 10%)

**Supporting Notebooks:**
- **`cloud_cover.ipynb`**: Analyze cloud cover in imagery
- **`concatenate_files.ipynb`**: Merge feature files
- **`impute_features.ipynb`**: Handle missing values in features

### 3. Task Modeling (`code/3_task_modeling/`)

Train predictive models by joining features with crop yield data:

**Primary Notebooks:**
- **`model_1_sensor.ipynb`**: Train models with single-sensor features
- **`model_2_sensor.ipynb`**: Train models combining Landsat 8 and Sentinel-2
- **`climate_model.ipynb`**: Incorporate climate data into models

**Modeling Process:**
1. Import and process administrative boundaries, features, and crop yield data
2. Handle NA values (imputation, dropping)
3. Summarize features to match crop data spatial resolution
4. Split data into train/test sets
5. Train linear regression model
6. Generate prediction maps
7. Evaluate model performance (R², residual analysis)

**Python Scripts** (for HPC/batch processing):
- `model_1_sensor.py`
- `model_2_sensor_10_splits.py`
- `model_2_sensor_10_splits_oos_preds.py`

### 4. Results & Visualization (`code/4_results/` and `code/5_figures/`)

Generate comprehensive results analysis and publication-quality figures:

- **`results.qmd`/`results.ipynb`**: Main results document (see [rendered HTML](https://htmlpreview.github.io/?https://github.com/cropmosaiks/crop-modeling/blob/main/code/4_results/results.html))
- **`final_figures.qmd`**: Generate publication figures
- **R scripts**: `figure_01.R` through `figure_07.R` for individual figures

## Requirements

### Software Dependencies

- **Python** ≥ 3.9
- **Key Libraries**:
  - `pytorch`, `torchvision` (GPU-enabled)
  - `planetary-computer` ≥ 0.4.6
  - `stackstac` ≥ 0.2.2
  - `geopandas`, `dask-geopandas`
  - `scikit-learn`, `statsmodels`
  - See `environment/environment.yml` for complete list

### Hardware

- **For Feature Extraction**:
  - GPU (NVIDIA with CUDA support)
  - Adequate RAM (processing can be memory-intensive)
  - Access to Microsoft Planetary Computer or similar STAC catalog
  
- **For Modeling**:
  - Most modern personal computers can run the modeling notebooks
  - HPC access useful for large-scale experiments

### Data Access

- **Planetary Computer**: Free account required for satellite data access ([sign up](https://planetarycomputer.microsoft.com/))
- **Crop Data**: Zambia district-level data used in this project is not publicly available due to restrictions. Contact the CropMOSAIKS team for access inquiries

## Installation

### Option 1: Local Installation

```bash
# Clone the repository
git clone https://github.com/cropmosaiks/crop-modeling.git
cd crop-modeling

# Create conda environment
conda env create -f environment/environment.yml

# Activate environment
conda activate mosaiks-env

# Launch Jupyter Lab
jupyter lab
```

### Option 2: Microsoft Planetary Computer Hub

1. Sign up for [MPC Hub](https://planetarycomputer.microsoft.com/docs/overview/environment/) (approval may take 24+ hours)
2. Select the **GPU - PyTorch** environment option
3. Clone this repository into your MPC Hub workspace
4. Open and run the notebooks

### Option 3: HPC/SLURM Systems

For large-scale processing on HPC systems:

```bash
# Example SLURM submission
sbatch slurm/model_1.sh
sbatch slurm/model_2_10_splits.sh
```

See `slurm/slurm_notes.txt` for details on configuring SLURM scripts.

## Getting Started

### Quick Start

1. **Create a spatial grid** over your region of interest using `code/1_grid_creation/dense_grid.ipynb`
2. **Extract features** from satellite imagery using `code/2_feature_extraction/s-2_ls-c2-l2_multiband.ipynb`
3. **Train a model** by joining features with your task data using `code/3_task_modeling/model_1_sensor.ipynb`
4. **Analyze results** and generate visualizations using scripts in `code/4_results/` and `code/5_figures/`

### Using Your Own Data

The workflow is designed to be task-agnostic. To apply it to your own prediction task:

1. Prepare your labeled data as a tabular dataframe with latitude/longitude columns
2. Follow the grid creation and feature extraction steps for your region
3. Spatially join your labels with the extracted features
4. Adapt the modeling notebooks to your specific task

## Model Performance

The best-performing model achieves:
- **Demeaned R (correlation coefficient)** = 0.61
- **Demeaned R²** = 0.27 (explains ~27% of variance in crop yield predictions)

These metrics indicate strong predictive performance for temporal crop yield prediction, though there is room for improvement through:
- Expanding training data with more years
- Dense sampling within crop-masked regions
- Testing different cloud cover thresholds
- Incorporating additional spectral bands

## Datasets

### Input Data

1. **Satellite Imagery**:
   - [Landsat 8 Collection 2 Level-2](https://planetarycomputer.microsoft.com/dataset/landsat-c2-l2) (February 2013 - present)
   - [Sentinel-2 Level-2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a) (June 2015 - present)

2. **Administrative Boundaries**: District-level boundaries for Zambia

3. **Crop Area Masks**: Used for weighted averages and spatial filtering

4. **Crop Yield Data**: District-level maize yields for Zambia (not included due to restrictions)
   - Coarse province-level data available from [Zambia Data Portal](https://zambia.opendataforafrica.org/)

### Output Data

- **Feature Files**: Compressed feather format containing RCF values for each grid point
- **Model Predictions**: Crop yield predictions at district level
- **Visualizations**: Maps of predicted vs. actual yields, residuals, and temporal trends

## Computing Constraints

### Microsoft Planetary Computer

- **Storage**: 15 GB persistent + ~200 GB temporary per session
- **Memory**: Can be limiting for large-scale processing
- **GPU Access**: Shared nodes (first-come, first-served)

### Recommendations

- Download and delete output files regularly to avoid storage limits
- Implement aggressive memory management for large datasets
- Consider alternative compute options for extensive processing (see [MPC compute docs](https://planetarycomputer.microsoft.com/docs/concepts/computing/))

## Contributing

This project was completed in June 2022 by the CropMOSAIKS team. Suggestions for improvements are welcome through:

- **Issues**: Report bugs or suggest enhancements
- **Pull Requests**: Contribute code improvements or new features
- **Contact**: Reach out to team members (see [organization README](https://github.com/cropmosaiks))

### Areas for Contribution

- Expand to other regions (Tanzania, Nigeria, other countries)
- Improve cloud cover filtering methods
- Test additional satellite bands and sensors
- Correlate predictions with climate anomalies
- Apply to other prediction tasks (forest cover, population, etc.)

For contributing features to the [MOSAIKS API](https://nadar.gspp.berkeley.edu/home/index/?next=/portal/index/), see the [mosaiks-api repository](https://github.com/calebrob6/mosaiks-api).

## Related Repositories

This repository is part of a larger project ecosystem:

- **[Featurization](https://github.com/cropmosaiks/Featurization)**: Detailed featurization workflow (legacy/reference)
- **[Modeling](https://github.com/cropmosaiks/Modeling)**: Additional modeling documentation (legacy/reference)

## Citation

If you use this code or approach in your research, please cite:

Rolf, E., Proctor, J., Carleton, T. et al. A generalizable and accessible approach to machine learning with global satellite imagery. *Nat Commun* **12**, 4392 (2021). https://doi.org/10.1038/s41467-021-24638-z

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

Copyright 2022 CropMOSAIKS

## Acknowledgments

- **Kathy Baylis Lab** (UC Santa Barbara) for providing Zambia crop yield data
- **Protensia Hudunka** for contextual information on crop data
- **Microsoft Planetary Computer Team** (Caleb Robinson, Tom Augspurger) for base code and platform support
- **MOSAIKS Team** for the foundational methodology

## Code of Conduct

Please see the [Code of Conduct](https://github.com/cropmosaiks/.github/blob/main/CODE_OF_CONDUCT.md) for participation guidelines.