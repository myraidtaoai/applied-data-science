# Seoul Bike Rental — TFT Prediction

A small project demonstrating Temporal Fusion Transformer (TFT) predictions for hourly bike rental demand in Seoul. The repository contains a prediction script (`tft_prediction.py`), example notebooks, training artifacts (pretrained models), and the dataset used for inference.

**Repository structure**
- `tft_prediction.py` — TFT prediction pipeline (data preprocessing, model loading, scaling, inference, evaluation)
- `run_python_script.ipynb` — Notebook that shows how to run `tft_prediction.py` (Colab-oriented) due to the need to use GPU acceleration and the compatibility considerations of GPU (There are some issues with MPS support on Apple Silicon).
- `Seoul_bike_prediction.ipynb` — Notebook with prediction examples using varisous models and evaluation metrics.
- `data/SeoulBikeData.csv` — Raw dataset used for preprocessing and predictions
- `models/` — Saved model checkpoints and `.pt` files used for inference:
  - `tft_1h.pt`, `tft_1h.pt.ckpt`
  - `tft_24h.pt`, `tft_24h.pt.ckpt`
  - `tft_3d.pt`, `tft_3d.pt.ckpt`

**What this code does**
- Loads raw Seoul bike data from `data/SeoulBikeData.csv`.
- Performs preprocessing and feature engineering (datetime index, one-hot encoding for categorical variables, numerical feature handling).
- Converts data into Darts `TimeSeries` objects and rescales them (float32 for MPS compatibility).
- Loads pretrained TFT models for multiple horizons (1-hour, 24-hour, 3-day) and runs predictions.
- Evaluates predictions (RMSE, RMSLE, R2) and outputs arrays / `TimeSeries` of predicted values.

**Requirements**
- Python 3.8+ recommended
- pytorch (compatible with your platform; CPU/MPS/CUDA supported by the script)
- darts (a forecasting library) — the script and notebooks assume `darts[torch]`
- pandas, numpy

A minimal `pip` install line used in the included notebook is:

```bash
pip install darts "darts[torch]" pandas numpy --quiet
```

Note: On macOS with Apple Silicon, installing PyTorch with MPS support requires following PyTorch installation instructions on https://pytorch.org/.

**Quick usage (local)**
1. Clone or open this repository in your machine.
2. Ensure `data/SeoulBikeData.csv` and the model files under `models/` are present.
3. (Optional) Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install darts "darts[torch]" pandas numpy
```

4. Run the prediction script directly:

```bash
python tft_prediction.py
```

The script includes a `main()` function. By default the script may reference a Google Drive path — edit the `folder_path` and model/data paths near the top of `main()` to point to local paths (or place the data/models where the script expects them).

**Colab / Notebook usage**
- The included `run_python_script.ipynb` demonstrates mounting Google Drive, installing `darts`, and running `tft_prediction.py` on Colab.
- If running on Colab, copy the `tft_prediction.py` to the Colab environment and update path variables to point to mounted Drive locations.

**Key configuration points in `tft_prediction.py`**
- `TFTPredictorConfig` contains paths for the three models and `data_path`. Update values to your local paths before running.
- Prediction lengths are set for `'1h'` (1), `'24h'` (24), and `'3d'` (72).
- The script attempts to choose an appropriate device (CUDA / CPU). Make sure the installed PyTorch supports your hardware.

**Outputs**
- The script prints progress and evaluation metrics for each horizon.
- Predicted values are returned as NumPy arrays and Darts `TimeSeries` objects inside the script; adapt the script to save predictions to CSV if needed.

**Troubleshooting**
- FileNotFoundError: Verify the `data_path` points to `data/SeoulBikeData.csv` and model paths exist.
- Darts or PyTorch import errors: Check your installed package versions and the Python interpreter. Use the Colab notebook for an easy environment that already installs `darts`.
- MPS issues on Apple Silicon: ensure PyTorch build supports MPS and avoid float64 tensors; the script converts to `float32` for MPS compatibility.

**Next steps / Recommendations**
- Add a small example script/notebook that demonstrates saving predictions to `outputs/` as CSV.
- Optionally include unit tests for preprocessing functions and a small sample dataset for CI.
- The script has some issues with MPS and float64 tensors. The issues need to be fixed in future Darts/PyTorch releases.

If you'd like, I can:
- Add a `requirements.txt` with pinned versions.
- Update `tft_prediction.py` to accept CLI arguments for paths and device.
- Save example predicted CSV outputs under an `outputs/` folder.

---
Generated from project files. If you'd like changes or a more detailed README (installation with exact pinned versions, CLI usage, or examples), tell me which parts to expand.

**Prophet Experiment**
- **Notebook**: `Prophet_Prediction.ipynb` — an end-to-end exploration using Darts' `Prophet` wrapper for hourly, daily and 3-day aggregated forecasts.
- **What it does**: Preprocesses the same `SeoulBikeData.csv` dataset, creates time series and covariates, and trains/evaluates Prophet models for multiple horizons (hourly, daily, 3-day).
- **Modeling details**: The notebook demonstrates how to configure Prophet seasonality (hourly/daily/weekly/yearly), add covariates (monthly cyclic features), and use `historical_forecasts()` for rolling/expanding-window evaluation.
- **Evaluation metrics**: The notebook computes `MAE`, `RMSE`, `MAPE` (where applicable), `sMAPE`, and `R2` to assess forecast performance.
- **How to run**: Open `Prophet_Prediction.ipynb` in Colab or locally. In Colab the notebook installs required packages and mounts Google Drive; locally, ensure you have Darts and Prophet installed as in the requirements above.

