import warnings
warnings.filterwarnings('ignore')
import pandas as pd
from prophet import Prophet
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import mlflow
import logging
import os

# Configurer les logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def train_prophet(df, target_column, run_id):
    """
    Train a Prophet model on the specified target column of the DataFrame.
    
    Parameters:
    - df: pandas DataFrame with a DatetimeIndex and the target column
    - target_column: str, the name of the target column to forecast
    
    Returns:
    - dict: Results including MSE and forecast DataFrame
    """
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logger.info(f"Training Prophet model for {target_column}...")
    
    # Vérifier que l'index est un DatetimeIndex
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex for time series analysis.")
    
    # Créer un DataFrame avec uniquement l'index (dates) et la colonne cible
    temp_df = pd.DataFrame({
        "ds": df.index,
        "y": df[target_column]
    })
    logger.info(f"temp_df shape: {temp_df.shape}")
    logger.info(f"temp_df sample:\n{temp_df.head()}")
    
    # Déduire la fréquence des dates
    detected_freq = pd.infer_freq(temp_df["ds"]) or "MS"
    logger.info(f"Detected frequency: {detected_freq}")
    
    # Diviser les données en ensembles d'entraînement et de test
    split_index = int(len(temp_df) * 0.8)
    train_df = temp_df.iloc[:split_index]
    test_df = temp_df.iloc[split_index:]
    logger.info(f"train_df shape: {train_df.shape}, test_df shape: {test_df.shape}")
    
    # Entraîner le modèle Prophet dans un run MLflow imbriqué
    with mlflow.start_run(nested=True, run_name=f"Prophet_{target_column}"):
        try:
            model = Prophet()
            model.fit(train_df)
            mlflow.sklearn.log_model(model, "prophet_model")
            
            # Faire des prédictions
            future = model.make_future_dataframe(periods=len(test_df), freq=detected_freq)
            forecast = model.predict(future)
            logger.info(f"forecast shape: {forecast.shape}")
            logger.info(f"forecast columns: {forecast.columns.tolist()}")
            logger.info(f"forecast sample:\n{forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()}")
            
            # Calculer l'erreur MSE
            y_true = test_df["y"].values
            y_pred = forecast["yhat"].iloc[-len(test_df):].values
            mse = mean_squared_error(y_true, y_pred)
            mlflow.log_metric("mse", mse)
            logger.info(f"MSE for {target_column}: {mse}")
            
            # Sauvegarder le fichier .png dans le répertoire courant
            file_name = f"prophet_forecast_{target_column}.png"
            file_path = os.path.abspath(file_name)
            plt.figure(figsize=(10, 5))
            plt.plot(temp_df["ds"], temp_df["y"], label="Données Réelles", alpha=0.7)
            plt.plot(forecast["ds"], forecast["yhat"], label="Prédictions", linestyle="dashed")
            plt.fill_between(
                forecast["ds"],
                forecast["yhat_lower"],
                forecast["yhat_upper"],
                color="gray",
                alpha=0.2,
            )
            plt.title(f"Prévisions pour {target_column}")
            plt.xlabel("Date")
            plt.ylabel("Valeur")
            plt.legend()
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path, artifact_path="model_plots")
                logger.info(f"Logged artifact: {file_path} in artifact_path='model_plots'")
            else:
                logger.error(f"Failed to save plot: {file_path}")
            
            # Sauvegarder les prévisions dans un fichier CSV
            forecast_df = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]]
            forecast_file = f"forecast_prophet_{target_column}.csv"
            forecast_df.to_csv(forecast_file, index=False)
            mlflow.log_artifact(forecast_file, artifact_path="forecasts")
        
        except Exception as e:
            logger.error(f"Error training Prophet model for {target_column}: {str(e)}")
            raise
    
    return {"mse": mse, "forecast": forecast_df}