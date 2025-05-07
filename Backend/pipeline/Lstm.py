import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import tensorflow as tf
from optuna.integration import TFKerasPruningCallback
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import mlflow
import os
import logging

# Configurer les logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_sequences(X, y, seq_length=12):
    Xs, ys = [], []
    for i in range(len(X) - seq_length):
        Xs.append(X[i:i+seq_length])
        ys.append(y[i + seq_length])
    return np.array(Xs), np.array(ys)

def train_lstm(df, target_column):
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logger.info(f"Training LSTM model for {target_column}...")
    
    df.index = pd.to_datetime(df.index)
    df.dropna(inplace=True)
    
    feature_columns = [col for col in df.columns if col != target_column] if len(df.columns) > 1 else [target_column]
    
    scalers = {}
    scaled_data = {}
    for column in df.columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[column] = scaler.fit_transform(df[[column]])
        scaled_data[column] = scaled_data[column].flatten()
        scalers[column] = scaler
    
    scaled_data = pd.DataFrame(scaled_data, index=df.index)
    scaled_data.reset_index(inplace=True)
    
    train_size = int(len(scaled_data) * 0.8)
    X_train = scaled_data.iloc[:train_size][feature_columns]
    y_train = scaled_data.iloc[:train_size][target_column]
    X_test = scaled_data.iloc[train_size:][feature_columns]
    y_test = scaled_data.iloc[train_size:][target_column]
    
    detected_freq = pd.infer_freq(df.index)
    if detected_freq in ["D", "B"]:
        seq_length = 10
    elif detected_freq in ["MS", "M"]:
        seq_length = 12
    else:
        seq_length = 1
    
    X_train_seq, y_train_seq = create_sequences(X_train.values, y_train.values, seq_length)
    X_test_seq, y_test_seq = create_sequences(X_test.values, y_test.values, seq_length)
    
    with mlflow.start_run(nested=True, run_name=f"LSTM_{target_column}"):
        try:
            def objective(trial):
                model = Sequential()
                n_layers = trial.suggest_int('n_layers', 1, 3)
                units = trial.suggest_int('units', 32, 128)
                dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
                learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
                
                for i in range(n_layers):
                    return_seq = (i < n_layers - 1)
                    if i == 0:
                        model.add(LSTM(units, activation='tanh', return_sequences=return_seq, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
                    else:
                        model.add(LSTM(units, activation='tanh', return_sequences=return_seq))
                    model.add(Dropout(dropout_rate))
                
                model.add(Dense(units=1, activation='linear'))
                model.compile(optimizer=Adam(learning_rate), loss='mse', metrics=['mse'])
                
                early_stopping = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5, restore_best_weights=True)
                pruning_callback = TFKerasPruningCallback(trial, "val_loss")
                
                history = model.fit(
                    X_train_seq, y_train_seq,
                    validation_data=(X_test_seq, y_test_seq),
                    epochs=100,
                    batch_size=16,
                    callbacks=[early_stopping, pruning_callback],
                    verbose=0
                )
                
                return min(history.history["val_loss"])
            
            study = optuna.create_study(direction='minimize', pruner=optuna.pruners.MedianPruner(n_warmup_steps=5))
            study.optimize(objective, n_trials=50)
            
            best_params = study.best_trial.params
            mlflow.log_params(best_params)
            
            model = Sequential()
            for i in range(best_params['n_layers']):
                return_seq = (i < best_params['n_layers'] - 1)
                if i == 0:
                    model.add(LSTM(best_params['units'], activation='relu', return_sequences=return_seq, input_shape=(X_train_seq.shape[1], X_train_seq.shape[2])))
                else:
                    model.add(LSTM(best_params['units'], activation='relu', return_sequences=return_seq))
                model.add(Dropout(best_params['dropout_rate']))
            
            model.add(Dense(units=1, activation='linear'))
            model.compile(optimizer=Adam(best_params['learning_rate']), loss='mse', metrics=['mse'])
            
            model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_test_seq, y_test_seq),
                epochs=100,
                batch_size=16,
                verbose=1
            )
            
            mlflow.keras.log_model(model, "lstm_model")
            
            predictions = model.predict(X_test_seq)
            
            y_test_rescaled = scalers[target_column].inverse_transform(y_test.values.reshape(-1, 1))
            y_pred_rescaled = scalers[target_column].inverse_transform(predictions)
            y_train_rescaled = scalers[target_column].inverse_transform(y_train.values.reshape(-1, 1))
            
            mse = mean_squared_error(y_test_rescaled[seq_length:], y_pred_rescaled)
            mae = mean_absolute_error(y_test_rescaled[seq_length:], y_pred_rescaled)
            r2 = r2_score(y_test_rescaled[seq_length:], y_pred_rescaled)
            mape = mean_absolute_percentage_error(y_test_rescaled[seq_length:], y_pred_rescaled)
            
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mape", mape)
            logger.info(f"MSE for {target_column}: {mse}")
            
            # Sauvegarder le fichier .png avec vérification
            file_name = f"lstm_forecast_{target_column}.png"
            file_path = os.path.abspath(file_name)
            plt.figure(figsize=(10, 6))
            plt.plot(y_train.index, y_train_rescaled, label="Train")
            plt.plot(y_test.index, y_test_rescaled, label="Test", color="orange")
            plt.plot(y_test[seq_length:].index, y_pred_rescaled.flatten(), label="Forecast", linestyle="dashed", color="red")
            plt.title(f'Prédictions LSTM pour {target_column}')
            plt.legend()
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path, artifact_path="model_plots")
                logger.info(f"Logged artifact: {file_path} in artifact_path='model_plots'")
            else:
                logger.error(f"Failed to save plot: {file_path}")
            
            # Sauvegarder les prévisions dans un fichier CSV
            forecast_df = pd.DataFrame({target_column: y_pred_rescaled.flatten()}, index=y_test[seq_length:].index)
            forecast_file = f"forecast_lstm_{target_column}.csv"
            forecast_df.to_csv(forecast_file, index=False)
            mlflow.log_artifact(forecast_file, artifact_path="forecasts")
        
        except Exception as e:
            logger.error(f"Error training LSTM model for {target_column}: {str(e)}")
            raise
    
    return {"mse": mse, "forecast": forecast_df}