from alphavantage import AlphaVantage
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from LSTM import Model
import pandas as pd
from typing import List, Tuple
from ModelTester import ModelTester
import os
from datetime import datetime
from config import Config

config = Config()

def load_and_preprocess_data(ticker: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split OHLCV data maintaining temporal order.
    Returns train, validation, and test sets.
    """
    alphavantage = AlphaVantage()
    df = alphavantage.get_ticker_data(ticker)
    
    # Extract OHLCV features
    features = np.column_stack([
        df['Open'].to_numpy(),
        df['High'].to_numpy(),
        df['Low'].to_numpy(),
        df['Close'].to_numpy(),
        df['Volume'].to_numpy()
    ])
    
    # Calculate split points
    test_size = int(len(features) * 0.2)
    val_size = int(len(features) * 0.1)
    
    # Split the data maintaining temporal order
    train_data = features[:-test_size-val_size]  # First 70%
    val_data = features[-test_size-val_size:-test_size]  # Next 10%
    test_data = features[-test_size:]  # Last 20%
    
    print(f"Data split sizes:")
    print(f"Training data: {len(train_data)} points ({len(train_data)/len(features)*100:.1f}%)")
    print(f"Validation data: {len(val_data)} points ({len(val_data)/len(features)*100:.1f}%)")
    print(f"Test data: {len(test_data)} points ({len(test_data)/len(features)*100:.1f}%)")
    print(f"Feature dimensionality (D): {features.shape[1]}")
    
    return train_data, val_data, test_data



def scale_data(train_data: np.ndarray, val_data: np.ndarray, test_data: np.ndarray, 
               window_size: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[MinMaxScaler]]:
    """
    Scale each OHLCV feature independently using separate scalers.
    """
    scalers = []
    train_scaled = np.zeros_like(train_data)
    val_scaled = np.zeros_like(val_data)
    test_scaled = np.zeros_like(test_data)
    
    # Scale each feature independently
    for i in range(train_data.shape[1]):
        scaler = MinMaxScaler()
        
        # Fit on training data
        scaler.fit(train_data[:, i].reshape(-1, 1))
        
        # Transform all datasets
        train_scaled[:, i] = scaler.transform(train_data[:, i].reshape(-1, 1)).flatten()
        val_scaled[:, i] = scaler.transform(val_data[:, i].reshape(-1, 1)).flatten()
        test_scaled[:, i] = scaler.transform(test_data[:, i].reshape(-1, 1)).flatten()
        
        scalers.append(scaler)
    
    return train_scaled, val_scaled, test_scaled, scalers
    


def prepare_sequences(data: np.ndarray, batch_size: int, 
                     sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences for LSTM with multiple features.
    """
    sequences = []
    targets = []
    
    for i in range(len(data) - sequence_length):
        sequences.append(data[i:(i + sequence_length)])
        targets.append(data[i + sequence_length, 3])  # Using Close price as target
    
    # Convert to numpy arrays
    X = np.array(sequences)
    y = np.array(targets)
    
    # X shape will be (samples, sequence_length, features)
    # y shape will be (samples, 1) containing only Close prices
    return X, y.reshape(-1, 1)


def test_model(predictions: np.ndarray, test_data: np.ndarray, scaler: MinMaxScaler, 
               sequence_length: int) -> None:
    """
    Comprehensive model testing function.
    """
    # Initialize tester
    tester = ModelTester(scaler)
    
    # Get actual values (excluding the first sequence_length points used for initial prediction)
    y_true = test_data[sequence_length:]
    y_pred = predictions[:len(y_true)]  # Ensure same length as y_true
    
    # Calculate and print metrics
    metrics = tester.calculate_metrics(y_true, y_pred)
    print("\nModel Performance Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Calculate and print directional accuracy
    dir_accuracy = tester.evaluate_directional_accuracy(y_true, y_pred)
    print(f"\nDirectional Accuracy: {dir_accuracy:.2f}%")
    
    # Create visualizations
    print("\nGenerating visualizations...")
    tester.plot_predictions(y_true, y_pred, "Stock Price Predictions")
    tester.plot_predictions(y_true, y_pred, "Stock Price Predictions (with Moving Average)", window=20)
    tester.plot_error_distribution(y_true, y_pred)
    tester.plot_scatter(y_true, y_pred)
    tester.plot_all_in_one(y_true, y_pred, "Analysis")
    
    
def train_and_evaluate_model(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           X_test: np.ndarray, scalers: List[MinMaxScaler]) -> np.ndarray:
    """
    Train LSTM model and make predictions with multiple features.
    """
    sequence_length = X_train.shape[1]
    n_features = X_train.shape[2] 
    current_date = datetime.now().strftime("%Y-%m-%d")
    model_filename = f"LSTM-{current_date}.h5"
    
    print(f"n_features: {n_features}")
    # Initialize model
    model = Model(ticker=config.get_property("ticker"), dropout=config.get_property("dropout"))
    
    # Try to load existing model
    if os.path.exists(model_filename) and config.get_property("load_model") == True:
        print(f"Loading existing model: {model_filename}")
        model.load_model(model_filename)
    else:
        print("Training new model...")
        model.build_model()
        model.train(
            x=X_train,
            y=y_train,
            validation_data=(X_val, y_val),
            epochs=config.get_property("epochs"),
            batch_size=config.get_property("batch_size")
        )
        
        try:
            model.save_model(model_filename)
            print(f"Model saved as: {model_filename}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Inverse transform predictions (using Close price scaler)
    close_price_scaler = scalers[3]  # Scaler for Close prices
    predictions = close_price_scaler.inverse_transform(predictions)
    
    return predictions.flatten()

def infer():
    """
    Perform inference on new data using the trained model.
    Expects space-separated OHLCV values as input.
    """
    ticker = config.get_property("ticker")
    sequence_length = config.get_property("sequence_length")
    current_date = datetime.now().strftime("%Y-%m-%d")
    model_filename = f"LSTM-{current_date}.h5"
    
    # Check if model exists
    if not os.path.exists(model_filename):
        print(f"Error: Model file {model_filename} not found. Please train the model first.")
        return
        
    try:
        # Get historical data for scaling
        train_data, val_data, test_data = load_and_preprocess_data(ticker)
        _, _, _, scalers = scale_data(train_data, val_data, test_data)
        
        # Initialize model
        model = Model(ticker=ticker, dropout=config.get_property("dropout"))
        model.load_model(model_filename)
        
        # Get input values
        print("Enter space-separated OHLCV values (at least {sequence_length} sets):")
        values = input().strip()
        data_points = [float(x) for x in values.split()]
        
        # Reshape input data into OHLCV format
        if len(data_points) < sequence_length * 5:
            print(f"Error: Not enough data points. Need at least {sequence_length * 5} values (5 features * {sequence_length} time steps)")
            return
            
        # Reshape into (timesteps, features)
        n_features = 5
        input_data = np.array(data_points).reshape(-1, n_features)
        
        # Scale the input data using the same scalers
        scaled_input = np.zeros_like(input_data)
        for i in range(n_features):
            scaled_input[:, i] = scalers[i].transform(input_data[:, i].reshape(-1, 1)).flatten()
        
        # Prepare sequence
        if len(scaled_input) >= sequence_length:
            sequence = scaled_input[-sequence_length:]
        else:
            print("Error: Not enough data points for the sequence length")
            print(len(scaled_input))
            print(sequence_length)
            return
            
        # Reshape for model input (batch_size, sequence_length, features)
        model_input = sequence.reshape(1, sequence_length, n_features)
        
        # Make prediction
        prediction = model.predict(model_input)
        
        # Inverse transform the prediction (using Close price scaler)
        final_prediction = scalers[3].inverse_transform(prediction)
        
        print(f"\nPredicted next closing price: ${final_prediction[0][0]:.2f}")
        
        # Optional: Return confidence metrics or additional analysis
        return final_prediction[0][0]
        
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

def main():
    ticker = config.get_property("ticker")
    sequence_length = config.get_property("sequence_length")
    
    # Load and preprocess OHLCV data
    train_data, val_data, test_data = load_and_preprocess_data(ticker)
    
    # Scale data
    scaled_train, scaled_val, scaled_test, scalers = scale_data(
        train_data, val_data, test_data
    )
    
    # Prepare sequences
    X_train, y_train = prepare_sequences(scaled_train, batch_size=config.get_property("batch_size"), sequence_length=sequence_length)
    X_val, y_val = prepare_sequences(scaled_val, batch_size=config.get_property("batch_size"), sequence_length=sequence_length)
    X_test, y_test = prepare_sequences(scaled_test, batch_size=config.get_property("batch_size"), sequence_length=sequence_length)
    
    print("\nData shapes after sequence preparation:")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")
    
    # Train model and get predictions
    predictions = train_and_evaluate_model(
        X_train, y_train,
        X_val, y_val,
        X_test, scalers
    )
    
    # Test the model (using Close price for evaluation)
    test_model(predictions, test_data[:, 3], scalers[3], sequence_length)
    
    return predictions

if __name__ == "__main__":
    if config.get_property("run_state") == "train":
        predictions = main()
        print("Predictions shape:", predictions.shape)
        print("Sample predictions:", predictions[:10])
    elif config.get_property("run_state") == "infer":
        infer()
        