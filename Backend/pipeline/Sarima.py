import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import mlflow
import os
import logging

# Configurer les logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_sarima(df, target_column):
    """
    Train a SARIMA model on the specified target column of the DataFrame.
    
    Parameters:
    - df: pandas DataFrame with a DatetimeIndex and the target column
    - target_column: str, the name of the target column to forecast
    
    Returns:
    - dict: Results including MSE and forecast DataFrame
    """
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logger.info(f"Training SARIMA model for {target_column}...")
    
    df.index = pd.to_datetime(df.index)
    series = df[target_column]
    
    split_index = int(len(df) * 0.8)
    train_df = series.iloc[:split_index]
    test_df = series.iloc[split_index:]
    
    with mlflow.start_run(nested=True, run_name=f"SARIMA_{target_column}"):
        try:
            auto_model = auto_arima(train_df, seasonal=True, m=12, stepwise=True, trace=True, suppress_warnings=True)
            best_order = auto_model.order
            best_seasonal_order = auto_model.seasonal_order
            
            mlflow.log_param("order", best_order)
            mlflow.log_param("seasonal_order", best_seasonal_order)
            
            model = SARIMAX(train_df, 
                           order=best_order, 
                           seasonal_order=best_seasonal_order,
                           enforce_stationarity=False, 
                           enforce_invertibility=False)
            
            sarima_model = model.fit(disp=False)
            mlflow.sklearn.log_model(sarima_model, "sarima_model")
            
            forecast = sarima_model.forecast(steps=len(test_df))
            mse = mean_squared_error(test_df, forecast)
            mlflow.log_metric("mse", mse)
            mlflow.log_metric(f"mse_{target_column}", mse)
            logger.info(f"MSE for {target_column}: {mse}")
            
            # Sauvegarder le fichier .png dans le répertoire courant avec vérification
            file_name = f"sarima_forecast_{target_column}.png"
            file_path = os.path.abspath(file_name)
            plt.figure(figsize=(10, 5))
            plt.plot(train_df.index, train_df, label="Train")
            plt.plot(test_df.index, test_df, label="Test", color="orange")
            plt.plot(test_df.index, forecast, label="Forecast", linestyle="dashed", color="red")
            plt.title(f"Prédictions SARIMA pour {target_column}")
            plt.legend()
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path, artifact_path="model_plots")
                logger.info(f"Logged artifact: {file_path} in artifact_path='model_plots'")
            else:
                logger.error(f"Failed to save plot: {file_path}")
            
            # Sauvegarder les prévisions dans un fichier CSV
            forecast_df = pd.DataFrame({"ds": test_df.index, "yhat": forecast.values})
            forecast_file = f"forecast_sarima_{target_column}.csv"
            forecast_df.to_csv(forecast_file, index=False)
            mlflow.log_artifact(forecast_file, artifact_path="forecasts")
        
        except Exception as e:
            logger.error(f"Error training SARIMA model for {target_column}: {str(e)}")
            raise
    
    return {"mse": mse, "forecast": forecast_df}