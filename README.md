# Occluder Study: Visualizing Network Representations of Occluded Objects

## Overview
This repository contains code and data for investigating how discriminative neural networks represent occluded objects. We explore different computational approaches to visualize what networks "think" lies behind occluders.

## Repository Structure
```
occluder-study/
├── src/                    # Shared utilities and functions
├── data/                   # Raw and processed datasets shared by all approaches
├── approaches/             # Different research approaches
│   ├── heatmap/           # Current: Probabilistic heatmap visualization
│   ├── cluster_analysis/  # Abandoned: Clustering-based completion
│   ├── procrustes/        # Abandoned: Shape alignment methods
│   └── bspline/           # Abandoned: Fitting a bspline on a distance Matrix
├── results/               # General results shared by all approaches
└── docs/                  # Documentation
```

## Quick Start

### Prerequisites
- MATLAB R2024b or later
- Deep Learning Toolbox
- Computer Vision Toolbox
- Statistics and Machine Learning Toolbox

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/05d762de69/occluder-study.git
   cd occluder-study
   ```

2. Set up MATLAB paths:
   ```matlab
   addpath('src/stimulus_generation');
   addpath('src/network_utils');
   ```

3. Download required data:
   - Trained network will be available in `data/models/`

### Running the Analysis
```matlab
% Current best approach
cd approaches/heatmap/scripts
run('main_heatmap.m')
```

## Research Approaches

### Current: Heatmap Visualization
**Status:** Active development  
**Method:** Generate probabilistic heatmaps from network activations and fits a spline
**Location:** `approaches/heatmap/`

### Previous Approaches
- **B-spline Completion** (Abandoned): Direct curve fitting approach for stimuli generation
- **Clustering Analysis** (Abandoned): Prototype-based completion / best cluster based on random sampling and comparing distances
- **Procrustes Methods** (Abandoned): Shape alignment techniques for stimuli generation

See `EXPERIMENT_LOG.md` for detailed research timeline.

## Data
- **Raw stimuli:** File containing different classes of shape contours
- **Network responses:** AlexNet final layer activations
- **Generated outputs:** Stimuli shared by all approaches


## License
This project is licensed under the MIT License - see LICENSE file for details.

## Contact
- **Author:** Hannes Schätzle
- **Email:** 679693hs@eur.nl
- **Institution:** Erasmus University Rotterdam
