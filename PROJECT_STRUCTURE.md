# Project Structure

## Overview
This project is organized into the following directories:

## Directory Structure

```
Paraphrase/
├── data/                    # All data files
│   ├── raw/                # Raw input data files
│   │   ├── book1_fixed.json
│   │   ├── Book1.xlsx
│   │   ├── template_data_collection.csv
│   │   └── ...
│   ├── processed/          # Processed training data
│   │   ├── mpc_cleaned_combined_train_with_book1.json
│   │   ├── mpc_cleaned_combined_val.json
│   │   └── ...
│   └── cache/              # Tokenized data cache
│
├── models/                  # Model files
│   ├── final/              # Final trained model (created after training)
│   └── checkpoints/        # Training checkpoints (created during training)
│
├── scripts/                 # Executable scripts
│   ├── train_with_book1.bat          # Main training script (MPC + Book1)
│   ├── train_full_data.bat           # Train on full MPC data
│   ├── START_TRAINING.bat            # Wrapper to start training
│   ├── reset_training.bat            # Reset training environment
│   ├── start_api.bat                 # Start API server
│   ├── install_pytorch_cuda.bat      # Install PyTorch with CUDA
│   └── utils/                        # Utility Python scripts
│       ├── check_gpu.py
│       ├── merge_training_files.py
│       ├── convert_csv_to_json.py
│       └── ...
│
├── src/                     # Source code
│   ├── api/                # API server code
│   ├── training/           # Training code
│   ├── data/               # Data processing code
│   └── evaluation/         # Evaluation code
│
├── tests/                   # Test files
│   ├── test_model_part1.py
│   └── ...
│
├── docs/                    # Documentation
│   ├── HOW_TO_RUN_TRAINING.md
│   ├── INSTALL_GPU.md
│   ├── README_DATA_COLLECTION.md
│   └── ...
│
├── config.yaml              # Configuration file
├── requirements.txt         # Python dependencies (CPU)
├── requirements-cuda.txt    # Python dependencies (CUDA)
├── setup.py                 # Setup script
├── README.md                # Main README
└── docker-compose.yml       # Docker configuration
```

## Quick Start

### Training
```bash
# From scripts directory
scripts\START_TRAINING.bat

# Or directly
scripts\train_with_book1.bat
```

### Testing GPU
```bash
python scripts\utils\check_gpu.py
```

### Starting API
```bash
scripts\start_api.bat
```

## File Organization Rules

### Batch Scripts (.bat)
- **Location**: `scripts/`
- **Purpose**: Executable scripts for training, testing, setup
- **Examples**: `train_with_book1.bat`, `reset_training.bat`

### Python Utilities (.py)
- **Location**: `scripts/utils/`
- **Purpose**: Helper scripts for data processing, checking, fixing
- **Examples**: `check_gpu.py`, `merge_training_files.py`

### Documentation (.md)
- **Location**: `docs/`
- **Purpose**: Project documentation and guides
- **Examples**: `HOW_TO_RUN_TRAINING.md`, `INSTALL_GPU.md`

### Data Files
- **Raw data**: `data/raw/`
- **Processed data**: `data/processed/`
- **Cache**: `data/cache/` (auto-generated)

### Model Files
- **Final model**: `models/final/` (created after training)
- **Checkpoints**: `models/checkpoints/` (created during training)

### Test Files
- **Location**: `tests/`
- **Purpose**: Unit tests and model testing scripts

## Notes

- The `models/` directory is empty initially and will be populated during training
- The `data/cache/` directory is auto-generated during training
- All batch scripts in `scripts/` can be run directly or via wrapper scripts
- Utility Python scripts should be run with: `python scripts/utils/script_name.py`
