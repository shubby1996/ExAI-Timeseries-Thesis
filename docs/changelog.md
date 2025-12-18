# Changelog

## [Unreleased] - 2025-11-29

### Added
- **NHiTS Model Support**: Implemented a full pipeline for N-HiTS model.
    - Created `NHiTS/` directory.
    - Added `train_nhits_nordbyen.py`, `evaluate_nhits_nordbyen.py`, `predict_nhits_nordbyen.py`, `visualize_nhits_predictions.py`.
- **Documentation**: Added `docs/` folder with project structure and pipeline guides.

### Changed
- **Project Structure**:
    - Moved TFT-specific scripts to `TFT/` directory.
    - Moved generic scripts (`align_data.py`, `feature_engineering.py`) to root.
- **Preprocessing**:
    - Renamed `tft_preprocessing.py` to `model_preprocessing.py` to reflect its generic nature.
    - Renamed `TFTFeatureConfig` to `ModelFeatureConfig`.
    - Renamed `prepare_tft_data` to `prepare_model_data`.
    - Added backward compatibility aliases to allow loading old TFT models.
- **NHiTS Training Logic**:
    - Modified training and evaluation scripts to merge "future covariates" into "past covariates" because Darts' NHiTS implementation does not support future covariates.

### Fixed
- **Path Issues**: Updated all scripts to use relative paths (`os.path.join(os.path.dirname(__file__), "..")`) to correctly locate data and modules regardless of execution directory.
