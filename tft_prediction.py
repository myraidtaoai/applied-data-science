"""
TFT Model Prediction Script
This script loads a trained Temporal Fusion Transformer (TFT) model and makes predictions
on bike rental demand using Seoul bike data.
"""

import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from typing import Tuple, List
import torch

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import required libraries
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import TFTModel
from darts.metrics import rmse, r2_score as darts_r2_score, rmsle


class TFTPredictorConfig:
    """Configuration class for TFT predictions with multiple horizons"""
    
    def __init__(
        self,
        model_path_1h: str = None,
        model_path_24h: str = None,
        model_path_3d: str = None,
        data_path: str = None,
    ):
        """
        Initialize configuration for TFT predictor with multiple prediction horizons
        
        Args:
            model_path_1h: Path to saved TFT model for 1-hour ahead prediction (.pt file)
            model_path_24h: Path to saved TFT model for 24-hour ahead prediction (.pt file)
            model_path_3d: Path to saved TFT model for 3-day ahead prediction (.pt file)
            data_path: Path to input data CSV
        """
        self.model_path_1h = model_path_1h
        self.model_path_24h = model_path_24h
        self.model_path_3d = model_path_3d
        self.data_path = data_path
        
        # Prediction lengths for each horizon
        self.prediction_lengths = {
            '1h': 1,
            '24h': 24,
            '3d': 72
        }


class TFTPredictor:
    """
    Temporal Fusion Transformer predictor for Seoul bike rental demand
    Supports multiple prediction horizons: 1-hour, 24-hour, and 3-day ahead
    """
    
    def __init__(self, config: TFTPredictorConfig):
        """
        Initialize the TFT predictor
        
        Args:
            config: TFTPredictorConfig object containing model and data paths
        """
        self.config = config
        self.models = {
            '1h': None,
            '24h': None,
            '3d': None
        }
        self.scaler = None
        self.ts_data = None
        self.features_data = None
        
    def load_model(self) -> bool:
        """
        Load all pre-trained TFT models for different prediction horizons
        Maps models to CPU if CUDA is not available. Uses CPU for MPS compatibility.
        
        Returns:
            bool: True if at least one model loaded successfully, False otherwise
        """
        models_loaded = 0
        
        # Determine device mapping for loading
        # Use CPU for MPS (float64 not supported) and when CUDA unavailable
        if torch.cuda.is_available():
            device_map = None  # Uses default CUDA device
        else:
            device_map = torch.device('cpu')  # Force CPU for MPS compatibility
        
        # Load 1-hour ahead model
        if self.config.model_path_1h is not None:
            try:
                self.models['1h'] = TFTModel.load(
                    self.config.model_path_1h,
                    map_location=device_map
                )
                self.models['1h'].model.float()
                # Force float32 for MPS compatibility and move to CPU
                self.models['1h'].model = self.models['1h'].model.to(dtype=torch.float32, device='cpu')
                print(f"✓ 1-hour ahead model loaded from: {self.config.model_path_1h}")
                models_loaded += 1
            except FileNotFoundError:
                print(f"✗ 1-hour ahead model not found at {self.config.model_path_1h}")
            except Exception as e:
                print(f"✗ Error loading 1-hour ahead model: {str(e)}")
        
        # Load 24-hour ahead model
        if self.config.model_path_24h is not None:
            try:
                self.models['24h'] = TFTModel.load(
                    self.config.model_path_24h,
                    map_location=device_map
                )
                self.models['24h'].model.float()
                # Force float32 for MPS compatibility and move to CPU
                self.models['24h'].model = self.models['24h'].model.to(dtype=torch.float32, device='cpu')
                print(f"✓ 24-hour ahead model loaded from: {self.config.model_path_24h}")
                models_loaded += 1
            except FileNotFoundError:
                print(f"✗ 24-hour ahead model not found at {self.config.model_path_24h}")
            except Exception as e:
                print(f"✗ Error loading 24-hour ahead model: {str(e)}")
        
        # Load 3-day ahead model
        if self.config.model_path_3d is not None:
            try:
                self.models['3d'] = TFTModel.load(
                    self.config.model_path_3d,
                    map_location=device_map
                )
                self.models['3d'].model.float()
                # Force float32 for MPS compatibility and move to CPU
                self.models['3d'].model = self.models['3d'].model.to(dtype=torch.float32, device='cpu')
                print(f"✓ 3-day ahead model loaded from: {self.config.model_path_3d}")
                models_loaded += 1
            except FileNotFoundError:
                print(f"✗ 3-day ahead model not found at {self.config.model_path_3d}")
            except Exception as e:
                print(f"✗ Error loading 3-day ahead model: {str(e)}")
        
        if models_loaded == 0:
            print("Error: No models were loaded successfully")
            return False
        
        print(f"\n✓ {models_loaded}/3 models loaded successfully")
        return True
    
    def preprocess_data(self, df: pd.DataFrame) -> Tuple[TimeSeries, TimeSeries]:
        """
        Preprocess Seoul bike data and perform feature engineering.
        
        Args:
            df: Raw dataframe from SeoulBikeData.csv
            
        Returns:
            Tuple containing:
                - ts_series: Target time series (hourly bike rental count)
                - features_series: Feature time series (all covariates)
        """
        # Rename columns
        new_columns = [
            "date", "hourly_count", "hour", "temperature", "humidity",
            "wind_speed", "visibility", "dew_point_temperature",
            "solar_radiation", "rainfall", "snowfall", "seasons",
            "holiday", "functioning_day"
        ]
        df.columns = new_columns
        
        # Convert date to datetime format
        df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
        
        # Standardize holiday column values
        df = df.replace({"No Holiday": "non_holiday", "Holiday": "holiday"})
        
        # Convert hour to string for categorical encoding
        df['hour'] = df['hour'].astype(str)
        
        # Create datetime index
        df['datetime'] = pd.to_datetime(
            df['date'].astype(str) + ' ' + df['hour'].astype(str) + ':00:00'
        )
        df.set_index('datetime', inplace=True)
        
        print("✓ Basic data preprocessing completed")
        
        # Feature engineering: separate numerical and categorical features
        numerical_features = df.select_dtypes(include=np.number).columns.tolist()
        categorical_features = df.select_dtypes(include='object').columns.tolist()
        
        # Remove 'hour' from categorical features since it will be one-hot encoded
        if 'hour' in categorical_features:
            categorical_features.remove('hour')
        
        # One-hot encode categorical variables
        df_categorical = df[categorical_features]
        df_dummies = pd.get_dummies(df_categorical)
        df_dummies = df_dummies.replace({False: 0, True: 1})
        df_dummies['date'] = df['date']
        df_dummies['hour'] = df['hour']
        
        # Merge numerical and encoded categorical features
        df_merged = pd.merge(df[numerical_features], df_dummies, 
                             left_index=True, right_index=True)
        
        print("✓ Feature engineering completed")
        print(f"  Total features: {df_merged.shape[1] - 1}")  # -1 for hourly_count
        
        # Convert to Darts TimeSeries format
        ts_series = TimeSeries.from_dataframe(df_merged[['hourly_count']])
        features_series = TimeSeries.from_dataframe(
            df_merged.drop(['hourly_count', 'date', 'hour'], axis=1)
        )
        
        # Resample to hourly frequency (handles any missing values)
        ts_series = ts_series.resample(freq="h")
        features_series = features_series.resample(freq="h")
        
        # Convert to float32 for MPS compatibility (MPS doesn't support float64)
        ts_series = ts_series.astype(np.float32)
        features_series = features_series.astype(np.float32)
        
        print("✓ Converted to TimeSeries format")
        print(f"  Target shape: {ts_series.shape}")
        print(f"  Features shape: {features_series.shape}")
        
        return ts_series, features_series
    
    
    def load_data(self) -> bool:
        """
        Load and preprocess data from CSV file
        
        Returns:
            bool: True if data loaded and preprocessed successfully, False otherwise
        """
        try:
            if self.config.data_path is None:
                print("Error: Data path not specified")
                return False
            
            # Read CSV file
            df = pd.read_csv(self.config.data_path, encoding='cp949')
            print(f"✓ Data loaded from: {self.config.data_path}")
            print(f"  Shape: {df.shape}")
            
            # Preprocess and engineer features
            self.ts_data, self.features_data = self.preprocess_data(df)
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.config.data_path}")
            return False
        except Exception as e:
            print(f"Error loading or preprocessing data: {str(e)}")
            return False
    
    def prepare_time_series(
        self, 
        df: pd.DataFrame, 
        freq: str = 'H'
    ) -> TimeSeries:
        """
        Prepare time series object from values
        
        Args:
            values: Array of values
            start_time: Start timestamp for the series
            
        Returns:
            TimeSeries: Darts TimeSeries object
        """    
        # Create hourly frequency index
        ts = TimeSeries.from_dataframe(df)
        ts = ts.resample(freq=freq)
        
        return ts
    
    def scale_data(self, ts: TimeSeries) -> TimeSeries:
        """
        Scale time series using a fitted Scaler
        
        Args:
            ts: TimeSeries to scale
            
        Returns:
            TimeSeries: Scaled time series (float32 for MPS compatibility)
        """
        if self.scaler is None:
            self.scaler = Scaler()
        scaled_ts = self.scaler.fit_transform(ts)
        # Ensure float32 for MPS compatibility
        scaled_ts = scaled_ts.astype(np.float32)
        return scaled_ts
    
    def inverse_scale(self, ts: TimeSeries) -> TimeSeries:
        """
        Inverse transform scaled time series back to original scale
        
        Args:
            ts: Scaled TimeSeries
            
        Returns:
            TimeSeries: Original scale time series (float32 for MPS compatibility)
        """
        if self.scaler is None:
            print("Warning: Scaler not loaded. Returning unscaled data.")
            return ts
            
        original_ts = self.scaler.inverse_transform(ts)
        # Ensure float32 for MPS compatibility
        original_ts = original_ts.astype(np.float32)
        return original_ts
    
    def predict(
        self,
        historical_data: TimeSeries,
        future_covariates: TimeSeries = None,
        horizon: str = '24h'
    ) -> TimeSeries:
        """
        Make predictions using the TFT model for specified horizon
        
        Args:
            historical_data: Historical time series data (should be scaled)
            future_covariates: Future covariate features (should be scaled)
            horizon: Prediction horizon - '1h', '24h', or '3d' (default: '24h')
            
        Returns:
            TimeSeries: Predictions in scaled space
        """
        if horizon not in self.models:
            print(f"Error: Unknown horizon '{horizon}'. Use '1h', '24h', or '3d'")
            return None
        
        model = self.models[horizon]
        if model is None:
            print(f"Error: Model for {horizon} prediction not loaded")
            return None
        
        n_steps = self.config.prediction_lengths[horizon]
        
        try:
            predictions = model.predict(
                n=n_steps,
                series=historical_data,
                future_covariates=future_covariates,
                verbose=False
            )
            return predictions
            
        except Exception as e:
            print(f"Error during {horizon} prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def predict_with_history(
        self,
        historical_data: TimeSeries,
        future_covariates: TimeSeries,
        validation_data: TimeSeries = None,
        horizon: str = '24h'
    ) -> Tuple[np.ndarray, List[TimeSeries]]:
        """
        Make recursive predictions for specified horizon, updating history with each iteration
        
        Args:
            historical_data: Initial historical time series data (scaled)
            future_covariates: Future covariate features (scaled)
            validation_data: Ground truth validation data for iterative updates
            horizon: Prediction horizon - '1h', '24h', or '3d' (default: '24h')
            n_iterations: Number of prediction iterations
            
        Returns:
            Tuple containing:
                - np.ndarray: Concatenated predictions
                - List[TimeSeries]: List of individual predictions
        """
        if horizon not in self.models:
            print(f"Error: Unknown horizon '{horizon}'. Use '1h', '24h', or '3d'")
            return None, []
        
        model = self.models[horizon]
        if model is None:
            print(f"Error: Model for {horizon} prediction not loaded")
            return None, []
        
        predictions_list = []
        current_history = historical_data
        current_covariates = future_covariates
        prediction_length = self.config.prediction_lengths[horizon]
        n_iterations = (len(validation_data) // prediction_length) if validation_data is not None else 1
        try:
            for i in range(n_iterations):
                pred = self.predict(current_history, current_covariates, horizon)
                # Check if prediction was successful
                if pred is None:
                    break
                    
                predictions_list.append(pred)
                
                # Update history and covariates with actual values if available
                if validation_data is not None:
                    start_idx = i * prediction_length
                    end_idx = start_idx + prediction_length
                    if end_idx <= len(validation_data):
                        actual_period = validation_data[start_idx:end_idx]
                        current_history = current_history.concatenate(actual_period)
                        
                        # Also advance the future_covariates slice
                        if end_idx < len(future_covariates):
                            current_covariates = future_covariates[end_idx:]
            
            # Concatenate all predictions only if we have any
            if len(predictions_list) > 0:
                predictions_array = np.concatenate([p.values().flatten() for p in predictions_list])
                return predictions_array, predictions_list
            else:
                print(f"Warning: No successful predictions generated for {horizon} horizon")
                return np.array([]), []
            
        except Exception as e:
            print(f"Error during iterative {horizon} prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            return None, []
    
    def evaluate_predictions(
        self,
        true_values: TimeSeries,
        predicted_values: TimeSeries
    ) -> dict:
        """
        Evaluate predictions using multiple metrics
        
        Args:
            true_values: Ground truth time series
            predicted_values: Predicted time series
            
        Returns:
            dict: Dictionary containing RMSE, R2, and RMSLE scores
        """
        try:
            metrics = {
                "rmse": rmse(true_values, predicted_values),
                "r2": darts_r2_score(true_values, predicted_values),
                "rmsle": rmsle(true_values, predicted_values),
            }
            return metrics
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return {}
    
    def clip_predictions(
        self,
        predictions: np.ndarray,
        min_val: float = 0,
        max_val: float = None
    ) -> np.ndarray:
        """
        Clip predictions to valid range (e.g., no negative bike rentals)
        
        Args:
            predictions: Prediction values to clip
            min_val: Minimum allowed value (default: 0)
            max_val: Maximum allowed value (default: None, no upper limit)
            
        Returns:
            np.ndarray: Clipped predictions
        """
        clipped = np.clip(predictions, a_min=min_val, a_max=max_val)
        return clipped


def main():
    """
    Main function: Load data, preprocess, initialize predictor, and make predictions
    Supports three prediction horizons: 1-hour, 24-hour, and 3-day ahead
    """
    # Configure paths - UPDATE THESE TO YOUR LOCAL PATHS
    folder_path = "/content/drive/My Drive/Appled-Data-Science/Data"
    config = TFTPredictorConfig(
        model_path_1h=f"{folder_path}/tft_1h.pt",     # Update this path - 1-hour ahead model
        model_path_24h=f"{folder_path}/tft_24h.pt",   # Update this path - 24-hour ahead model
        model_path_3d=f"{folder_path}/tft_3d.pt",     # Update this path - 3-day ahead model
        data_path=f"{folder_path}/SeoulBikeData.csv"    # Update this path
    )
    
    print("\n" + "="*60)
    print("Seoul Bike Rental - TFT Prediction Pipeline")
    print("Multiple Prediction Horizons: 1h, 24h, 3d")
    print("="*60)
    
    # Step 1: Load raw data
    print("\n[Step 1] Loading raw data...")
    df_raw = None
    
    try:
        df_raw = pd.read_csv(config.data_path, encoding='cp949')
        print(f"✓ Data loaded. Shape: {df_raw.shape}")
    except FileNotFoundError:
        print(f"✗ Error: Data file not found at {config.data_path}")
        print("Cannot proceed without data. Exiting.")
        return
    except Exception as e:
        print(f"✗ Error loading data: {str(e)}")
        print("Cannot proceed without data. Exiting.")
        return
    
    # Step 2: Preprocess and engineer features
    print("\n[Step 2] Preprocessing data and engineering features...")
    ts_series = None
    features_series = None
    historical_ts = None
    val_ts = None
    
    try:
        # Initialize a temporary predictor instance to use preprocess_data method
        temp_predictor = TFTPredictor(config)
        ts_series, features_series = temp_predictor.preprocess_data(df_raw)
        validation_cutoff = pd.Timestamp("2018-11-01")
        historical_ts, val_ts = ts_series.split_before(validation_cutoff) # Use all but last 500 for history
    except Exception as e:
        print(f"✗ Error during preprocessing: {str(e)}")
        print("Cannot proceed without preprocessed data. Exiting.")
        return
    
    # Step 3: Initialize predictor with all models
    print("\n[Step 3] Initializing predictor with multiple horizon models...")
    predictor = TFTPredictor(config)
    
    # Load models for all horizons
    if not predictor.load_model():
        print("✗ Warning: At least one model should be loaded to proceed")
        # Continue anyway to show the interface
    
    # Step 3.5: Scale the data for predictions (float32 for MPS compatibility)
    print("\n[Step 3.5] Scaling data for predictions...")
    historical_ts_scaled = None
    features_series_scaled = None
    val_ts_scaled = None
    
    try:
        historical_ts_scaled = predictor.scale_data(historical_ts)
        features_series_scaled = predictor.scale_data(features_series)
        val_ts_scaled = predictor.scale_data(val_ts)
        print("✓ Data scaled successfully (float32 format for MPS compatibility)")
    except Exception as e:
        print(f"✗ Error during scaling: {str(e)}")
        print("Cannot proceed without scaled data. Exiting.")
        return
    
    print("\n" + "="*60)
    print("TFT Predictor initialized successfully!")
    print("="*60)
    
    print("\n[Step 4] Ready to make predictions")
    print("\nMaking predictions for all horizons...")
    print("\n1. 1-hour ahead prediction...")
    predictions_1h_array, pred_list_1h = predictor.predict_with_history(historical_data = historical_ts_scaled, 
                                                                  future_covariates = features_series_scaled, 
                                                                  validation_data = val_ts_scaled,
                                                                  horizon='1h')
    print("✓ 1-hour predictions completed")
    
    print("\n2. 24-hour ahead prediction...")
    predictions_24h_array, pred_list_24h = predictor.predict_with_history(historical_data = historical_ts_scaled, 
                                                                  future_covariates = features_series_scaled, 
                                                                  validation_data = val_ts_scaled,
                                                                  horizon='24h')
    print("✓ 24-hour predictions completed")
    
    print("\n3. 3-day ahead prediction...")
    predictions_3d_array, pred_list_3d = predictor.predict_with_history(historical_data = historical_ts_scaled, 
                                                                  future_covariates = features_series_scaled, 
                                                                  validation_data = val_ts_scaled,
                                                                  horizon='3d')
    print("✓ 3-day predictions completed")
    
    # Convert and evaluate 1-hour predictions
    print("\n" + "="*60)
    print("[Step 5] Evaluating Predictions")
    print("="*60)
    
    print("\n1-Hour Ahead Predictions:")
    print("-" * 40)
    predicted_values_1h = None
    try:
        if predictions_1h_array is not None and len(predictions_1h_array) > 0:
            pred_time_index = val_ts.time_index[:len(predictions_1h_array)]
            predictions_1h_ts = TimeSeries.from_times_and_values(pred_time_index, predictions_1h_array)
            predicted_values_1h = predictor.inverse_scale(predictions_1h_ts)
            
            val_ts_trimmed = val_ts[:len(predicted_values_1h)]
            metrics_1h = predictor.evaluate_predictions(val_ts_trimmed, predicted_values_1h)
            
            print(f"  RMSE:  {metrics_1h.get('rmse', 'N/A'):.4f}")
            print(f"  R²:    {metrics_1h.get('r2', 'N/A'):.4f}")
            print(f"  RMSLE: {metrics_1h.get('rmsle', 'N/A'):.4f}")
        else:
            print("  ✗ No predictions generated")
    except Exception as e:
        print(f"  ✗ Error evaluating 1-hour predictions: {str(e)}")
    
    # Convert and evaluate 24-hour predictions
    print("\n24-Hour Ahead Predictions:")
    print("-" * 40)
    predicted_values_24h = None
    try:
        if predictions_24h_array is not None and len(predictions_24h_array) > 0:
            pred_time_index = val_ts.time_index[:len(predictions_24h_array)]
            predictions_24h_ts = TimeSeries.from_times_and_values(pred_time_index, predictions_24h_array)
            predicted_values_24h = predictor.inverse_scale(predictions_24h_ts)
            
            val_ts_trimmed = val_ts[:len(predicted_values_24h)]
            metrics_24h = predictor.evaluate_predictions(val_ts_trimmed, predicted_values_24h)
            
            print(f"  RMSE:  {metrics_24h.get('rmse', 'N/A'):.4f}")
            print(f"  R²:    {metrics_24h.get('r2', 'N/A'):.4f}")
            print(f"  RMSLE: {metrics_24h.get('rmsle', 'N/A'):.4f}")
        else:
            print("  ✗ No predictions generated")
    except Exception as e:
        print(f"  ✗ Error evaluating 24-hour predictions: {str(e)}")
    
    # Convert and evaluate 3-day predictions
    print("\n3-Day Ahead Predictions:")
    print("-" * 40)
    predicted_values_3d = None
    try:
        if predictions_3d_array is not None and len(predictions_3d_array) > 0:
            # Create TimeSeries from predictions using the correct time index
            pred_time_index = val_ts.time_index[:len(predictions_3d_array)]
            predictions_3d_ts = TimeSeries.from_times_and_values(pred_time_index, predictions_3d_array)
            
            # Inverse scale predictions back to original scale
            predicted_values_3d = predictor.inverse_scale(predictions_3d_ts)
            
            # Also trim validation data to match predictions length
            val_ts_trimmed = val_ts[:len(predicted_values_3d)]
            
            metrics_3d = predictor.evaluate_predictions(val_ts_trimmed, predicted_values_3d)
            
            print(f"  RMSE:  {metrics_3d.get('rmse', 'N/A'):.4f}")
            print(f"  R²:    {metrics_3d.get('r2', 'N/A'):.4f}")
            print(f"  RMSLE: {metrics_3d.get('rmsle', 'N/A'):.4f}")
        else:
            print("  ✗ No predictions generated")
    except Exception as e:
        print(f"  ✗ Error evaluating 3-day predictions: {str(e)}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
