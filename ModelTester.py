from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class ModelTester:
    def __init__(self, scaler: MinMaxScaler):
        """
        Initialize ModelTester with the scaler used for data preprocessing.
        """
        self.scaler = scaler
        self.metrics = {}
        
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate various performance metrics.
        """
        # Ensure data is in the correct shape
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        
        # Calculate metrics
        metrics = {
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred),
            'R2': r2_score(y_true, y_pred),
            'MAPE': np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        }
        
        self.metrics = metrics
        return metrics
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Stock Price Predictions", 
                        window: int = None, ax=None) -> None:
        """
        Plot actual vs predicted values with optional moving average smoothing.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(15, 8))
        
        if window:
            y_true_smooth = pd.Series(y_true).rolling(window=window).mean()
            y_pred_smooth = pd.Series(y_pred).rolling(window=window).mean()
            ax.plot(y_true_smooth, label='Actual (MA)', color='blue', alpha=0.6)
            ax.plot(y_pred_smooth, label='Predicted (MA)', color='red', alpha=0.6)
        else:
            ax.plot(y_true, label='Actual', color='blue', alpha=0.6)
            ax.plot(y_pred, label='Predicted', color='red', alpha=0.6)
            
        ax.set_title(title)
        ax.set_xlabel('Time')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        
        
    def plot_error_distribution(self, y_true: np.ndarray, y_pred: np.ndarray, ax=None) -> None:
        """
        Plot the distribution of prediction errors.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 6))
            
        errors = y_true - y_pred
        sns.histplot(errors, kde=True, ax=ax)
        ax.set_title('Distribution of Prediction Errors')
        ax.set_xlabel('Error')
        ax.set_ylabel('Frequency')
        ax.grid(True)
        
        
    def plot_scatter(self, y_true: np.ndarray, y_pred: np.ndarray, ax=None) -> None:
        """
        Create a scatter plot of predicted vs actual values.
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 10))
            
        ax.scatter(y_true, y_pred, alpha=0.5)
        ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        ax.set_title('Predicted vs Actual Values')
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.grid(True)
        
        
        
    def evaluate_directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculate the directional accuracy of predictions.
        """
        actual_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        return directional_accuracy
    
    def plot_all_in_one(self, y_true: np.ndarray, y_pred: np.ndarray, 
                        title: str = "Model Performance Analysis", 
                        window: int = None) -> None:
        """
        Create a combined figure with all plots.
        """
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 15))
        
        # Create a 2x2 grid of subplots
        gs = plt.GridSpec(2, 2, figure=fig)
        
        # Plot predictions (top-left)
        ax1 = fig.add_subplot(gs[0, 0])
        self.plot_predictions(y_true, y_pred, window=window, ax=ax1)
        
        # Plot error distribution (top-right)
        ax2 = fig.add_subplot(gs[0, 1])
        self.plot_error_distribution(y_true, y_pred, ax=ax2)
        
        # Plot scatter (bottom-left)
        ax3 = fig.add_subplot(gs[1, 0])
        self.plot_scatter(y_true, y_pred, ax=ax3)
        
        # Add metrics text box (bottom-right)
        ax4 = fig.add_subplot(gs[1, 1])
        metrics = self.calculate_metrics(y_true, y_pred)
        directional_accuracy = self.evaluate_directional_accuracy(y_true, y_pred)
        
        metrics_text = (
            f"Model Performance Metrics:\n\n"
            f"MSE: {metrics['MSE']:.4f}\n"
            f"RMSE: {metrics['RMSE']:.4f}\n"
            f"MAE: {metrics['MAE']:.4f}\n"
            f"R²: {metrics['R2']:.4f}\n"
            f"MAPE: {metrics['MAPE']:.2f}%\n"
            f"Directional Accuracy: {directional_accuracy:.2f}%"
        )
        
        ax4.text(0.5, 0.5, metrics_text,
                ha='center', va='center',
                bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                fontsize=12)
        ax4.set_axis_off()
        
        # Add main title
        plt.suptitle(title, fontsize=16, y=0.95)
        plt.tight_layout()
        plt.show()