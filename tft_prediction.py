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
import pickle

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Import required libraries
from darts import TimeSeries
from darts.models import TFTModel
from darts.metrics import rmse, r2_score as darts_r2_score, rmsle


class TFTPredictorConfig:
    """Configuration class for TFT predictions"""
    
    def __init__(
        self,
        model_path: str = None,
        scaler_path: str = None,
        data_path: str = None,
        prediction_length: int = 72,  # Default 3 days (72 hours)
    ):
        """
        Initialize configuration for TFT predictor
        
        Args:
            model_path: Path to saved TFT model (.pt file)
            scaler_path: Path to saved scaler (pickle file)
            data_path: Path to input data CSV
            prediction_length: Number of hours to predict (default: 72 for 3 days)
        """
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.data_path = data_path
        self.prediction_length = prediction_length


class TFTPredictor:
    """
    Temporal Fusion Transformer predictor for Seoul bike rental demand
    """
    
    def __init__(self, config: TFTPredictorConfig):
        """
        Initialize the TFT predictor
        
        Args:
            config: TFTPredictorConfig object containing model and data paths
        """
        self.config = config
        self.model = None
        self.scaler = None
        self.ts_data = None
        self.features_data = None
        
    def load_model(self) -> bool:
        """
        Load the pre-trained TFT model
        
        Returns:
            bool: True if model loaded successfully, False otherwise
        """
        try:
            if self.config.model_path is None:
                print("Error: Model path not specified")
                return False
                
            self.model = TFTModel.load(self.config.model_path)
            print(f"✓ Model loaded successfully from: {self.config.model_path}")
            return True
            
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.config.model_path}")
            return False
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
    
    def load_scaler(self) -> bool:
        """
        Load the pre-fitted scaler for data normalization
        
        Returns:
            bool: True if scaler loaded successfully, False otherwise
        """
        try:
            if self.config.scaler_path is None:
                print("Error: Scaler path not specified")
                return False
                
            with open(self.config.scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"✓ Scaler loaded successfully from: {self.config.scaler_path}")
            return True
            
        except FileNotFoundError:
            print(f"Error: Scaler file not found at {self.config.scaler_path}")
            return False
        except Exception as e:
            print(f"Error loading scaler: {str(e)}")
            return False
    
    def load_data(self) -> bool:
        """
        Load and preprocess data from CSV file
        
        Returns:
            bool: True if data loaded successfully, False otherwise
        """
        try:
            if self.config.data_path is None:
                print("Error: Data path not specified")
                return False
                
            # Read CSV file
            df = pd.read_csv(self.config.data_path, encoding='cp949')
            print(f"✓ Data loaded from: {self.config.data_path}")
            print(f"  Shape: {df.shape}")
            
            return True
            
        except FileNotFoundError:
            print(f"Error: Data file not found at {self.config.data_path}")
            return False
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def prepare_time_series(
        self, 
        values: np.ndarray, 
        start_time: pd.Timestamp = None
    ) -> TimeSeries:
        """
        Prepare time series object from values
        
        Args:
            values: Array of values
            start_time: Start timestamp for the series
            
        Returns:
            TimeSeries: Darts TimeSeries object
        """
        if start_time is None:
            start_time = pd.Timestamp('2017-12-01')
            
        # Create hourly frequency index
        time_index = pd.date_range(start=start_time, periods=len(values), freq='H')
        ts = TimeSeries.from_times_and_values(time_index, values)
        
        return ts
    
    def scale_data(self, ts: TimeSeries) -> TimeSeries:
        """
        Scale time series using loaded scaler
        
        Args:
            ts: TimeSeries to scale
            
        Returns:
            TimeSeries: Scaled time series
        """
        if self.scaler is None:
            print("Warning: Scaler not loaded. Returning unscaled data.")
            return ts
            
        scaled_ts = self.scaler.transform(ts)
        return scaled_ts
    
    def inverse_scale(self, ts: TimeSeries) -> TimeSeries:
        """
        Inverse transform scaled time series back to original scale
        
        Args:
            ts: Scaled TimeSeries
            
        Returns:
            TimeSeries: Original scale time series
        """
        if self.scaler is None:
            print("Warning: Scaler not loaded. Returning unscaled data.")
            return ts
            
        original_ts = self.scaler.inverse_transform(ts)
        return original_ts
    
    def predict(
        self,
        historical_data: TimeSeries,
        future_covariates: TimeSeries = None,
        n_steps: int = None
    ) -> TimeSeries:
        """
        Make predictions using the TFT model
        
        Args:
            historical_data: Historical time series data (should be scaled)
            future_covariates: Future covariate features (should be scaled)
            n_steps: Number of steps to predict (uses config if not specified)
            
        Returns:
            TimeSeries: Predictions in scaled space
        """
        if self.model is None:
            print("Error: Model not loaded")
            return None
        
        n_steps = n_steps or self.config.prediction_length
        
        try:
            predictions = self.model.predict(
                n=n_steps,
                series=historical_data,
                future_covariates=future_covariates
            )
            return predictions
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None
    
    def predict_with_history(
        self,
        historical_data: TimeSeries,
        future_covariates: TimeSeries,
        validation_data: TimeSeries = None,
        n_iterations: int = 1
    ) -> Tuple[np.ndarray, List[TimeSeries]]:
        """
        Make recursive predictions, updating history with each iteration
        
        Args:
            historical_data: Initial historical time series data (scaled)
            future_covariates: Future covariate features (scaled)
            validation_data: Ground truth validation data for iterative updates
            n_iterations: Number of prediction iterations
            
        Returns:
            Tuple containing:
                - np.ndarray: Concatenated predictions
                - List[TimeSeries]: List of individual predictions
        """
        if self.model is None:
            print("Error: Model not loaded")
            return None, []
        
        predictions_list = []
        current_history = historical_data
        
        try:
            for i in range(n_iterations):
                pred = self.predict(current_history, future_covariates)
                predictions_list.append(pred)
                
                # Update history with actual values if available
                if validation_data is not None:
                    start_idx = i * self.config.prediction_length
                    end_idx = start_idx + self.config.prediction_length
                    if end_idx <= len(validation_data):
                        actual_period = validation_data[start_idx:end_idx]
                        current_history = current_history.concatenate(actual_period)
                
                print(f"✓ Iteration {i+1}/{n_iterations} completed")
            
            # Concatenate all predictions
            predictions_array = np.concatenate([p.values().flatten() for p in predictions_list])
            
            return predictions_array, predictions_list
            
        except Exception as e:
            print(f"Error during iterative prediction: {str(e)}")
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
    Example usage of TFTPredictor
    """
    # Configure paths - UPDATE THESE TO YOUR LOCAL PATHS
    config = TFTPredictorConfig(
        model_path="/path/to/tft_3d.pt",  # Update this path
        scaler_path="/path/to/scaler.pkl",  # Update this path
        data_path="/path/to/SeoulBikeData.csv",  # Update this path
        prediction_length=72  # Predict 3 days (72 hours)
    )
    
    # Initialize predictor
    predictor = TFTPredictor(config)
    
    # Load model and scaler
    if not predictor.load_model():
        return
    
    if not predictor.load_scaler():
        return
    
    if not predictor.load_data():
        return
    
    print("\n" + "="*60)
    print("TFT Predictor initialized successfully!")
    print("="*60)
    
    # Example: Make predictions on new data
    # You would prepare your scaled historical data and covariates here
    # and call predictor.predict() or predictor.predict_with_history()
    
    print("\nTo make predictions, use:")
    print("  predictions = predictor.predict(historical_data, future_covariates)")
    print("\nOr for recursive predictions:")
    print("  predictions, pred_list = predictor.predict_with_history(")
    print("      historical_data, future_covariates, validation_data, n_iterations=5")
    print("  )")


if __name__ == "__main__":
    main()
