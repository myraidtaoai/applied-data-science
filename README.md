# Seoul Bike Rental — TFT Prediction

A small project demonstrating Temporal Fusion Transformer (TFT) predictions for hourly bike rental demand in Seoul. The repository contains a prediction script (`tft_prediction.py`), example notebooks, training artifacts (pretrained models), and the dataset used for inference.

## Repository structure
- `tft_prediction.py` — Main prediction pipeline. Performs data loading, preprocessing and feature engineering, converts data to Darts `TimeSeries`, scales data, loads pretrained TFT models (multiple horizons), runs inference and prints evaluation metrics.
- `run_python_script.ipynb` — Colab-oriented notebook that installs dependencies, mounts Google Drive, copies `tft_prediction.py` into the Colab environment and runs it. Useful for running the pipeline on Colab with mounted Drive.
- `Seoul_bike_prediction.ipynb` — Notebook with prediction examples, EDA and model comparisons (visualizations and evaluation metrics).
- `notebooks/` — Folder containing analysis notebooks:
  - `notebooks/Prophet_Prediction.ipynb` — Prophet experiment notebook (hourly, daily, 3-day forecasts using Darts' Prophet wrapper).
  - `notebooks/SeoulBikeMidReport.ipynb` — Mid-report notebook with EDA, summary statistics and key visualizations.
- `data/` — Data folder. Contains `data/SeoulBikeData.csv` (raw CSV dataset used across notebooks and the prediction script).
- `models/` — Saved model artifacts (used for inference):
  - `tft_1h.pt`, `tft_1h.pt.ckpt`  — 1-hour horizon TFT model
  - `tft_24h.pt`, `tft_24h.pt.ckpt` — 24-hour horizon TFT model
  - `tft_3d.pt`, `tft_3d.pt.ckpt`  — 3-day horizon TFT model
- `requirements.txt` — Pinned Python dependencies for reproducible installs.
- `README.md` — Project overview, usage and notes (this file).


## What this code does
- Loads raw Seoul bike data from `data/SeoulBikeData.csv`.
- Performs preprocessing and feature engineering (datetime index, one-hot encoding for categorical variables, numerical feature handling).
- Converts data into Darts `TimeSeries` objects and rescales them (float32 for MPS compatibility).
- Loads pretrained TFT models for multiple horizons (1-hour, 24-hour, 3-day) and runs predictions.
- Evaluates predictions (RMSE, RMSLE, R2) and outputs arrays / `TimeSeries` of predicted values.

## Requirements
- Python 3.8+ recommended
- pytorch (compatible with your platform; CPU/MPS/CUDA supported by the script)
- darts (a forecasting library) — the script and notebooks assume `darts[torch]`
- pandas, numpy

A minimal `pip` install line used in the included notebook is:

```bash
pip install darts "darts[torch]" pandas numpy --quiet
```

Note: On macOS with Apple Silicon, installing PyTorch with MPS support requires following PyTorch installation instructions on https://pytorch.org/.


## Colab / Notebook usage
- The included `run_python_script.ipynb` demonstrates mounting Google Drive, installing `darts`, and running `tft_prediction.py` on Colab.
- If running on Colab, copy or upload the `tft_prediction.py` to the Colab environment and update path variables to point to mounted Drive locations.

### Key configuration points in `tft_prediction.py`
- `TFTPredictorConfig` contains paths for the three models and `data_path`. Update values to your local paths before running.
- Prediction lengths are set for `'1h'` (1), `'24h'` (24), and `'3d'` (72).
- The script attempts to choose an appropriate device (CUDA / CPU). Make sure the installed PyTorch supports your hardware.

### Outputs
- The script prints progress and evaluation metrics for each horizon.
- Predicted values are returned as NumPy arrays and Darts `TimeSeries` objects inside the script; adapt the script to save predictions to CSV if needed.

## Troubleshooting
- FileNotFoundError: Verify the `data_path` points to `data/SeoulBikeData.csv` and model paths exist.
- Darts or PyTorch import errors: Check your installed package versions and the Python interpreter. Use the Colab notebook for an easy environment that already installs `darts`.
- MPS issues on Apple Silicon: ensure PyTorch build supports MPS and avoid float64 tensors; the script converts to `float32` for MPS compatibility.

## Next steps / Recommendations
- Add a small example script/notebook that demonstrates saving predictions to `outputs/` as CSV.
- Optionally include unit tests for preprocessing functions and a small sample dataset for CI.
- The `tft_prediction.py` script still has some issues with MPS and float64 tensors. The issues need to be fixed in future version.
- Save example predicted CSV outputs under an `outputs/` folder.

