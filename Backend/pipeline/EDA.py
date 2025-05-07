import warnings
warnings.filterwarnings('ignore')
import json
import matplotlib
# Forcer l'utilisation du backend Agg avant toute importation de matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import ydata_profiling as yd
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pandas.api.types import is_datetime64_any_dtype
import mlflow
import os
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def handle_missing_values(df, validation_fraction=0.1, random_state=42):
    if df.isnull().sum().sum() == 0:
        logging.info("No missing values detected after EDA. Skipping handling.")
        return df
    logging.warning("Missing values detected. Selecting best imputation method.")
    
    df_validation = df.copy()
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    np.random.seed(random_state)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df_validation[col] = pd.to_numeric(df_validation[col], errors='coerce')
        non_missing_indices = df[df[col].notnull()].index
        n_to_mask = int(len(non_missing_indices) * validation_fraction)
        if n_to_mask > 0:
            masked_indices = np.random.choice(non_missing_indices, n_to_mask, replace=False)
            mask.loc[masked_indices, col] = True
            df_validation.loc[masked_indices, col] = np.nan

    imputed_dfs = {
        "forward_fill": df_validation.fillna(method='ffill'),
        "backward_fill": df_validation.fillna(method='bfill'),
        "mean_imputation": df_validation.fillna(df_validation.mean()),
        "median_imputation": df_validation.fillna(df_validation.median()),
        "interpolation": df_validation.interpolate()
    }

    mse_scores = {}
    for method_name, imputed_df in imputed_dfs.items():
        y_true = df[mask]
        y_pred = imputed_df[mask]
        mse = np.mean((y_true - y_pred) ** 2)
        mse_scores[method_name] = mse
    logging.info(f"MSE scores for imputation methods: {mse_scores}")
    best_method = min(mse_scores, key=mse_scores.get)
    logging.info(f"Selected best missing value handling method: {best_method}")

    if best_method == "forward_fill":
        return df.fillna(method='ffill')
    elif best_method == "backward_fill":
        return df.fillna(method='bfill')
    elif best_method == "mean_imputation":
        return df.fillna(df.mean())
    elif best_method == "median_imputation":
        return df.fillna(df.median())
    elif best_method == "interpolation":
        return df.interpolate()
    else:
        logging.warning("No valid imputation method selected, returning original df.")
        return df

def generate_mlops_report(df):
    report = {
        "dataset_metadata": {
            "total_columns": len(df.columns),
            "total_rows": len(df),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max())
            }
        },
        "preprocessing_recommendations": {
            "stationarity": {},
            "feature_scaling": [],
            "feature_engineering": []
        },
        "statistical_insights": {
            "descriptive_stats": {},
            "correlations": {
                "significant_correlations": [],
                "correlation_matrix": {}
            }
        }
    }
    
    # Only process numeric columns for stationarity and statistical tests
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logging.info(f"Numeric columns for stationarity test: {numeric_cols.tolist()}")
    
    for column in numeric_cols:
        try:
            adf_result = adfuller(df[column].dropna())
            report["preprocessing_recommendations"]["stationarity"][column] = {
                "is_stationary": str(adf_result[1] < 0.05),
                "p_value": float(adf_result[1]),
                "transformation_needed": "YES" if adf_result[1] >= 0.05 else "NO"
            }
            
            if adf_result[1] >= 0.05:
                report["preprocessing_recommendations"]["feature_engineering"].append({
                    "column": column,
                    "suggested_transformations": [
                        "log_transformation",
                        "differencing",
                        "rolling_mean_normalization"
                    ]
                })
        except Exception as e:
            logging.error(f"Error performing adfuller test on column {column}: {str(e)}")
            report["preprocessing_recommendations"]["stationarity"][column] = {
                "is_stationary": "unknown",
                "p_value": None,
                "transformation_needed": "YES",
                "error": str(e)
            }

    for column in numeric_cols:
        try:
            report["statistical_insights"]["descriptive_stats"][column] = {
                "mean": float(df[column].mean()),
                "std": float(df[column].std()),
                "min": float(df[column].min()),
                "max": float(df[column].max())
            }
        except Exception as e:
            logging.error(f"Error calculating descriptive stats for column {column}: {str(e)}")
    
    # Correlation analysis only for numeric columns
    if len(numeric_cols) > 1:
        try:
            corr_matrix = df[numeric_cols].corr()
            report["statistical_insights"]["correlations"]["correlation_matrix"] = \
                {str(col): {str(subcol): float(corr_matrix.loc[col, subcol]) 
                            for subcol in numeric_cols} 
                 for col in numeric_cols}
            
            for col1 in numeric_cols:
                for col2 in numeric_cols:
                    if col1 != col2:
                        correlation = float(corr_matrix.loc[col1, col2])
                        if abs(correlation) > 0.5:
                            report["statistical_insights"]["correlations"]["significant_correlations"].append({
                                "features": [col1, col2],
                                "correlation": correlation,
                                "strength": "strong" if abs(correlation) > 0.7 else "moderate"
                            })
        except Exception as e:
            logging.error(f"Error calculating correlations: {str(e)}")
    
    for column in numeric_cols:
        try:
            if df[column].std() > 1:
                report["preprocessing_recommendations"]["feature_scaling"].append({
                    "column": column,
                    "recommended_method": ["standardization", "min_max_scaling"]
                })
        except Exception as e:
            logging.error(f"Error checking feature scaling for column {column}: {str(e)}")
    
    return report

def run_eda(df, output_report_path):
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    # Log DataFrame columns and types for debugging
    logging.info(f"Initial DataFrame columns: {df.columns.tolist()}")
    logging.info(f"DataFrame dtypes:\n{df.dtypes}")
    logging.info(f"Initial DataFrame index type: {type(df.index)}")

    # Check if the index is already a DatetimeIndex
    if isinstance(df.index, pd.DatetimeIndex):
        logging.info("DataFrame already has a DatetimeIndex. Proceeding with EDA.")
    else:
        # Détecter et définir une colonne datetime comme index
        try:
            datetime_set = False
            # Prefer columns named 'date' or 'ds' for datetime index
            datetime_candidates = ['date', 'ds']
            for col in datetime_candidates:
                if col in df.columns:
                    temp_series = pd.to_datetime(df[col], errors="coerce")
                    valid_date_ratio = temp_series.notna().mean()
                    if valid_date_ratio > 0.9 and is_datetime64_any_dtype(temp_series):
                        df.set_index(col, inplace=True)
                        logging.info(f"Set '{col}' as the datetime index.")
                        datetime_set = True
                        break
            # Fallback to any column that looks like a datetime
            if not datetime_set:
                for col in df.columns:
                    temp_series = pd.to_datetime(df[col], errors="coerce")
                    valid_date_ratio = temp_series.notna().mean()
                    if valid_date_ratio > 0.9 and is_datetime64_any_dtype(temp_series):
                        df.set_index(col, inplace=True)
                        logging.info(f"Set '{col}' as the datetime index (fallback).")
                        datetime_set = True
                        break
            # If no datetime column is found, create a synthetic datetime index
            if not datetime_set:
                logging.warning("No suitable datetime column found. Creating a synthetic datetime index starting from 2020-01-01.")
                start_date = pd.to_datetime("2020-01-01")
                df.index = pd.date_range(start=start_date, periods=len(df), freq='M')
                logging.info("Synthetic datetime index created.")
        except Exception as e:
            logging.error(f"Error setting datetime index: {str(e)}")
            raise

    # Log DataFrame after setting index
    logging.info(f"DataFrame columns after setting index: {df.columns.tolist()}")
    logging.info(f"DataFrame dtypes after setting index:\n{df.dtypes}")
    logging.info(f"DataFrame index after setting: {type(df.index)}")

    # Gérer les valeurs manquantes
    mlflow.log_metric("missing_values_before", df.isnull().sum().sum())
    df = handle_missing_values(df)
    mlflow.log_metric("missing_values_after", df.isnull().sum().sum())

    # Générer le rapport MLOps
    final_report = generate_mlops_report(df)
    with open(output_report_path, 'w') as f:
        json.dump(final_report, f, indent=4)
    if os.path.exists(output_report_path):
        mlflow.log_artifact(output_report_path)
        logging.info(f"Logged artifact: {output_report_path}")
    else:
        logging.error(f"Failed to save report: {output_report_path}")

    # Visualisations (sauvegardées comme artefacts)
    output_dir = os.path.dirname(output_report_path)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    logging.info(f"Numeric columns for visualization: {numeric_cols.tolist()}")

    if len(numeric_cols) == 0:
        logging.warning("No numeric columns found for visualization. Skipping plot generation.")

    for column in numeric_cols:
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(df.index, df[column], label=column)
            plt.title(f'Time Series Plot for {column}')
            plt.xlabel('Date')
            plt.ylabel('Values')
            plt.legend()
            file_path = os.path.join(output_dir, f"time_series_{column}.png")
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path)
                logging.info(f"Logged artifact: {file_path}")
            else:
                logging.error(f"Failed to save plot: {file_path}")
        except Exception as e:
            logging.error(f"Error generating time series plot for {column}: {str(e)}")
            plt.close()

    for column in numeric_cols:
        try:
            plt.figure(figsize=(10, 5))
            # Ensure enough data points for decomposition
            period = min(12, max(1, len(df) // 2))
            decomposition = seasonal_decompose(df[column], model='additive', period=period)
            decomposition.plot()
            plt.suptitle(f'Seasonal Decomposition of {column}')
            file_path = os.path.join(output_dir, f"decomposition_{column}.png")
            plt.savefig(file_path, bbox_inches='tight')
            plt.close()
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path)
                logging.info(f"Logged artifact: {file_path}")
            else:
                logging.error(f"Failed to save plot: {file_path}")
        except Exception as e:
            logging.error(f"Error generating decomposition plot for {column}: {str(e)}")
            plt.close()

    logging.info("EDA completed and report saved.")