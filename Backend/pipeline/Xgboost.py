import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import optuna
import mlflow
import os
import logging

# Configurer les logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def create_features(df, target_column, lags=3, rolling_windows=[]):
    df = df.copy()
    feature_columns = [col for col in df.columns if col != target_column] if len(df.columns) > 1 else [target_column]
    
    for column in feature_columns:
        for lag in range(1, lags + 1):
            df[f"{column}_lag_{lag}"] = df[column].shift(lag)
        
        detected_freq = pd.infer_freq(df.index)
        if detected_freq in ["D", "B"]:
            rolling_windows = [7, 14, 30]
        elif detected_freq in ["MS", "M"]:
            rolling_windows = [3, 6, 12]
        elif detected_freq in ["YS", "Y"]:
            rolling_windows = [2, 3, 5]
        
        for window in rolling_windows:
            df[f"rolling_mean_{window}"] = df[column].rolling(window=window).mean()
            df[f"rolling_std_{window}"] = df[column].rolling(window=window).std()
    
    df.dropna(inplace=True)
    X = df.drop(columns=[target_column], axis=1)
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    return X_train, X_test, y_train, y_test

def train_xgboost(df, target_column):
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    logger.info(f"Training XGBoost model for {target_column}...")
    
    df.index = pd.to_datetime(df.index)
    X_train, X_test, y_train, y_test = create_features(df, target_column)
    
    with mlflow.start_run(nested=True, run_name=f"XGBoost_{target_column}"):
        try:
            def objective(trial):
                params = {
                    'objective': 'reg:squarederror',
                    'eval_metric': 'rmse',
                    'eta': trial.suggest_float("eta", 0.01, 0.1, log=True),
                    'max_depth': trial.suggest_int("max_depth", 2, 8),
                    'subsample': trial.suggest_float("subsample", 0.5, 0.9),
                    'colsample_bytree': trial.suggest_float("colsample_bytree", 0.5, 0.9),
                    'gamma': trial.suggest_float("gamma", 1e-3, 10.0, log=True),
                    'alpha': trial.suggest_float("alpha", 1e-2, 10.0, log=True),
                    'lambda': trial.suggest_float("lambda", 1e-2, 10.0, log=True),
                    'tree_method': 'hist',
                    'seed': 42
                }
                
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_valid_scaled = scaler.transform(X_test)
                
                sample_weight = np.where(y_train > y_train.quantile(0.95), 2.0, 1.0)
                dtrain = xgb.DMatrix(X_train_scaled, label=y_train, weight=sample_weight)
                dvalid = xgb.DMatrix(X_valid_scaled, label=y_test)
                
                model = xgb.train(
                    params,
                    dtrain,
                    num_boost_round=1000,
                    evals=[(dvalid, "validation")],
                    early_stopping_rounds=50,
                    verbose_eval=False,
                    callbacks=[optuna.integration.XGBoostPruningCallback(trial, "validation-rmse")]
                )
                
                preds = model.predict(dvalid)
                mse = mean_squared_error(y_test, preds)
                return mse
            
            study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
            study.optimize(objective, n_trials=50)
            
            best_params = study.best_trial.params
            mlflow.log_params(best_params)
            
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            sample_weight = np.where(y_train > y_train.quantile(0.95), 2.0, 1.0)
            dtrain = xgb.DMatrix(X_train_scaled, label=y_train, weight=sample_weight)
            dtest = xgb.DMatrix(X_test_scaled, label=y_test)
            
            dtrain_params = {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'eta': best_params['eta'],
                'max_depth': best_params['max_depth'],
                'subsample': best_params['subsample'],
                'colsample_bytree': best_params['colsample_bytree'],
                'gamma': best_params['gamma'],
                'alpha': best_params['alpha'],
                'lambda': best_params['lambda'],
                'tree_method': 'hist',
                'seed': 42
            }
            
            model = xgb.train(
                dtrain_params, 
                dtrain, 
                num_boost_round=100,
                evals=[(dtest, 'test')],
                early_stopping_rounds=10,
                verbose_eval=10
            )
            mlflow.sklearn.log_model(model, "xgboost_model")
            
            predictions = model.predict(dtest)
            
            mse = mean_squared_error(y_test, predictions)
            mae = mean_absolute_error(y_test, predictions)
            r2 = r2_score(y_test, predictions)
            mape = mean_absolute_percentage_error(y_test, predictions)
            
            mlflow.log_metric("mse", mse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mape", mape)
            logger.info(f"MSE for {target_column}: {mse}")
            
            # Sauvegarder le fichier .png avec vérification
            file_name = f"xgboost_forecast_{target_column}.png"
            file_path = os.path.abspath(file_name)
            plt.figure(figsize=(10, 5))
            plt.plot(y_train.index, y_train, label="Train")
            plt.plot(y_test.index, y_test, label="Test", color="orange")
            plt.plot(y_test.index, predictions, label="Forecast", linestyle="dashed", color="red")
            plt.title(f"Prédictions XGBoost pour {target_column}")
            plt.legend()
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path, artifact_path="model_plots")
                logger.info(f"Logged artifact: {file_path} in artifact_path='model_plots'")
            else:
                logger.error(f"Failed to save plot: {file_path}")
            
            # Sauvegarder les prévisions dans un fichier CSV
            forecast_df = pd.DataFrame({target_column: predictions}, index=y_test.index)
            forecast_file = f"forecast_xgboost_{target_column}.csv"
            forecast_df.to_csv(forecast_file, index=False)
            mlflow.log_artifact(forecast_file, artifact_path="forecasts")
        
        except Exception as e:
            logger.error(f"Error training XGBoost model for {target_column}: {str(e)}")
            raise
    
    return {"mse": mse, "forecast": forecast_df}