from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from config import Config
from utils import Utils
from typing import Tuple, Optional, Dict, Any, Union
import os
import numpy as np
from tensorflow.keras.optimizers import Adam

config = Config()
utils = Utils()

class Model:
    def __init__(self, ticker, dropout=0.2):
        self.dropout = dropout
        self.model = Sequential()
        self.history = None
        self.ticker = ticker
        
    def build_model(self):
        """
        Build the model architecture from config.
        """
        for layer in config.get_property("layers"):
            neurons = layer.get('neurons')
            dropout_rate = layer.get('rate')
            activation = layer.get('activation')
            return_seq = layer.get('return_seq')
            input_timesteps = layer.get('input_timesteps')
            input_dim = layer.get('input_dim')
            
                        
            if layer['type'] == 'dense':
                self.model.add(Dense(neurons, activation=activation))
            elif layer['type'] == 'lstm':
                self.model.add(LSTM(
                    neurons, 
                    input_shape=(input_timesteps, input_dim), 
                    return_sequences=return_seq
                ))
            elif layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))
        
        optimizer = Adam(learning_rate=config.get_property("learning_rate"))
        self.model.compile(loss=config.get_property("loss"), optimizer=optimizer)
        
    def train(self, 
              x: np.ndarray, 
              y: np.ndarray, 
              epochs: int, 
              batch_size: int,
              validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
              validation_split: float = 0.0) -> None:
        """
        Train the model with validation data and callbacks.
        
        Args:
            x: Training features
            y: Training targets
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Tuple of (X_val, y_val) for validation
            validation_split: Alternative to validation_data, fraction of data to use for validation
        """
        print("Training started")
        
        # Create model directory if it doesn't exist
        model_dir = config.get_property("model_dir", "models") + f"/{self.ticker}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Generate model checkpoint path
        checkpoint_path = os.path.join(
            model_dir,
            self.ticker,
            f"checkpoint-{utils.get_datetime()}-{self.ticker}.keras"
        )
        
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=config.get_property("early_stopping_patience", 2),
                restore_best_weights=True
            ),
            ModelCheckpoint(
                filepath=checkpoint_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            ),
                ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.000001
            )
        ]
        
        # Train the model
        self.history = self.model.fit(
            x,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=validation_data,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save the final model
        model_name = os.path.join(
            model_dir,
            self.ticker,
            f"{config.get_property('model_name_template')}-{utils.get_datetime()}-{self.ticker}.h5"
        )
        self.save_model(model_name)
        
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions using the trained model.
        
        Args:
            X_test: Test features
            
        Returns:
            Model predictions
        """
        if self.model is None:
            raise ValueError("Model is not built")
        
        return self.model.predict(X_test, verbose=0)
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model:
            self.model.save(filepath)
            print(f"Model saved to: {filepath}")
        else:
            raise ValueError("Model not built or trained")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
            
        self.model = load_model(filepath)
        print(f"Model loaded from: {filepath}")
        
    def get_training_history(self) -> Optional[Dict[str, Any]]:
        """
        Get the training history if available.
        
        Returns:
            Dictionary containing training metrics history
        """
        return self.history.history if self.history else None