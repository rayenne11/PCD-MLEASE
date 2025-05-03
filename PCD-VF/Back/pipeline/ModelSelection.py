import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, acf
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.stats import skew, kurtosis
import warnings
import mlflow
import logging
import json

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ModelSelector:
    def __init__(self, eda_report_path):
        self.eda_report = self._load_eda_report(eda_report_path)

    @staticmethod
    def _load_eda_report(json_file):
        with open(json_file, 'r') as f:
            return json.load(f)

    def select_models(self, df, df_processed):
        logging.info("Starting model selection.")
        # Use the same target column logic as analyze_timeseries
        target_col = None
        if df_processed.shape[1] == 1:
            target_col = df_processed.columns[0]
        elif 'target_column' in self.eda_report:
            target_col = self.eda_report['target_column']
        else:
            # Assume the last column is the target if not specified
            target_col = df_processed.columns[-1]
            logging.warning(f"Target column not specified in EDA report. Assuming '{target_col}' as target.")

        # Analyze the processed DataFrame
        result = analyze_timeseries(df_processed, target_col=target_col)
        logging.info(f"Model selection result: {result}")

        # Extract the recommended models
        if result['recommended_approach'] == 'single_model':
            recommended_model = result['recommended_model']
            selected_models = [recommended_model]
            logging.info(f"Selected single model: {selected_models}")
        else:  # ensemble
            selected_models = result['ensemble_details']['models']
            logging.info(f"Selected ensemble models: {selected_models}")

        # Ensure we always return a list, even if empty
        # if not selected_models:
        #     logging.warning("No models selected. Falling back to default models: ['SARIMA', 'Prophet'].")
        #     selected_models = ["SARIMA", "Prophet"]

        mlflow.log_param("selected_models", selected_models)
        return selected_models

def analyze_timeseries(df, target_col=None):
    if isinstance(df, pd.Series):
        series = df
        df = pd.DataFrame(df)
        target_col = df.columns[0]
    elif target_col is None and df.shape[1] == 1:
        target_col = df.columns[0]
        series = df[target_col]
    elif target_col is not None:
        series = df[target_col]
    else:
        raise ValueError("For multivariate data, please specify the target column.")
    
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
        except:
            raise ValueError("Index must be convertible to datetime.")
    
    analysis = {}
    analysis['length'] = len(df)
    analysis['variables'] = df.shape[1]
    analysis['missing_values'] = df.isnull().sum().sum()
    
    adf_result = adfuller(series.dropna())
    analysis['stationary'] = adf_result[1] < 0.05
    
    try:
        if df.index.freq is None:
            df = df.asfreq(pd.infer_freq(df.index))
        decomposition = seasonal_decompose(series.dropna(), model='additive')
        trend = decomposition.trend.dropna()
        seasonal = decomposition.seasonal.dropna()
        residual = decomposition.resid.dropna()
        
        analysis['trend_strength'] = np.std(trend) / (np.std(residual) + np.std(trend))
        analysis['seasonality_strength'] = np.std(seasonal) / (np.std(residual) + np.std(seasonal))
        analysis['has_trend'] = analysis['trend_strength'] > 0.3
        analysis['has_seasonality'] = analysis['seasonality_strength'] > 0.3
    except:
        analysis['has_trend'] = None
        analysis['has_seasonality'] = None
    
    analysis['skewness'] = skew(series.dropna())
    analysis['kurtosis'] = kurtosis(series.dropna())
    analysis['linear'] = abs(analysis['skewness']) < 1 and abs(analysis['kurtosis']) < 3
    
    try:
        acf_values = acf(series.dropna(), nlags=min(50, len(series)//4))
        analysis['significant_autocorrelation'] = any(abs(acf_values[1:]) > 1.96/np.sqrt(len(series)))
    except:
        analysis['significant_autocorrelation'] = None
    
    analysis['complex_patterns'] = not analysis.get('linear', True) or analysis['variables'] > 3
    
    if analysis['length'] < 100:
        analysis['size'] = 'small'
    elif analysis['length'] < 1000:
        analysis['size'] = 'medium'
    else:
        analysis['size'] = 'large'
    
    try:
        if df.index.freq is None:
            freq = pd.infer_freq(df.index)
        else:
            freq = df.index.freq
        analysis['frequency'] = freq
        analysis['high_frequency'] = freq in ['H', 'min', 'T', 'S']
    except:
        analysis['frequency'] = None
        analysis['high_frequency'] = False
    
    recommendation, explanation, model_scores = recommend_model(analysis)
    ensemble_recommendation, ensemble_explanation = consider_ensemble(model_scores, analysis)
    
    if ensemble_recommendation:
        return {
            'recommended_approach': 'ensemble',
            'ensemble_details': ensemble_recommendation,
            'explanation': ensemble_explanation,
            'individual_model_scores': model_scores,
            'analysis': analysis
        }
    else:
        return {
            'recommended_approach': 'single_model',
            'recommended_model': recommendation,
            'explanation': explanation,
            'model_scores': model_scores,
            'analysis': analysis
        }

def recommend_model(analysis):
    explanations = []
    scores = {
        'SARIMA': 0,
        'Prophet': 0,
        'XGBoost': 0,
        'LSTM': 0
    }
    
    model_explanations = {}
    
    if analysis.get('stationary', False) or analysis.get('has_seasonality'):
        scores['SARIMA'] += 2
        model_explanations.setdefault('SARIMA', []).append("Handles stationarity or seasonality well")
    if analysis.get('significant_autocorrelation', False):
        scores['SARIMA'] += 2
        model_explanations.setdefault('SARIMA', []).append("Designed for significant autocorrelation")
    if analysis.get('size') == 'small' or analysis.get('size') == 'medium':
        scores['SARIMA'] += 1
        model_explanations.setdefault('SARIMA', []).append("Works well with small/medium datasets")
    if analysis.get('variables', 1) > 1:
        scores['SARIMA'] -= 2
        model_explanations.setdefault('SARIMA', []).append("Less effective with multiple variables")
    if analysis.get('high_frequency', False):
        scores['SARIMA'] -= 1
        model_explanations.setdefault('SARIMA', []).append("Can be computationally intensive for high-frequency data")
    
    if analysis.get('has_seasonality', False):
        scores['Prophet'] += 3
        model_explanations.setdefault('Prophet', []).append("Excels at modeling seasonality")
    if analysis.get('has_trend', False):
        scores['Prophet'] += 2
        model_explanations.setdefault('Prophet', []).append("Handles trends effectively")
    if analysis.get('missing_values', 0) > 0:
        scores['Prophet'] += 2
        model_explanations.setdefault('Prophet', []).append("Handles missing values automatically")
    if analysis.get('variables', 1) > 2:
        scores['Prophet'] -= 1
        model_explanations.setdefault('Prophet', []).append("Primarily designed for univariate forecasting with regressors")
    if analysis.get('complex_patterns', True) and not analysis.get('has_seasonality', False) and not analysis.get('has_trend', False):
        scores['Prophet'] -= 1
        model_explanations.setdefault('Prophet', []).append("May miss complex non-seasonal patterns")
    
    if analysis.get('complex_patterns', False):
        scores['XGBoost'] += 2
        model_explanations.setdefault('XGBoost', []).append("Models complex patterns effectively")
    if not analysis.get('linear', True):
        scores['XGBoost'] += 2
        model_explanations.setdefault('XGBoost', []).append("Handles non-linear relationships well")
    if analysis.get('variables', 1) > 1:
        scores['XGBoost'] += 2
        model_explanations.setdefault('XGBoost', []).append("Effectively utilizes multiple variables")
    if analysis.get('size') == 'small':
        scores['XGBoost'] -= 1
        model_explanations.setdefault('XGBoost', []).append("May need more data to learn patterns effectively")
    if analysis.get('has_seasonality', False) and not analysis.get('variables', 1) > 1:
        scores['XGBoost'] -= 1
        model_explanations.setdefault('XGBoost', []).append("May need feature engineering to capture seasonality")
    
    if analysis.get('complex_patterns', False):
        scores['LSTM'] += 3
        model_explanations.setdefault('LSTM', []).append("Excels at modeling complex temporal patterns")
    if analysis.get('size') == 'large':
        scores['LSTM'] += 2
        model_explanations.setdefault('LSTM', []).append("Performs well with large datasets")
    if analysis.get('variables', 1) > 1:
        scores['LSTM'] += 2
        model_explanations.setdefault('LSTM', []).append("Effectively utilizes multiple variables")
    if analysis.get('size') == 'small':
        scores['LSTM'] -= 3
        model_explanations.setdefault('LSTM', []).append("Needs substantial data for training")
    elif analysis.get('size') == 'medium':
        scores['LSTM'] -= 1
        model_explanations.setdefault('LSTM', []).append("Benefits from more training data")
    if analysis.get('has_seasonality', False) and not analysis.get('complex_patterns', False):
        scores['LSTM'] -= 1
        model_explanations.setdefault('LSTM', []).append("May be unnecessarily complex for simple seasonal patterns")
    
    recommended_model = max(scores, key=scores.get)
    
    model_descriptions = {
        'SARIMA': "SARIMA is recommended for this dataset due to its ability to handle time series with seasonality and autocorrelation.",
        'Prophet': "Prophet is recommended for this dataset as it excels at modeling time series with multiple seasonal patterns and trends.",
        'XGBoost': "XGBoost is recommended for this dataset due to its effectiveness with non-linear relationships and multiple variables.",
        'LSTM': "LSTM is recommended for this dataset due to its ability to model complex temporal dependencies and non-linear patterns."
    }
    
    specific_explanation = model_descriptions[recommended_model] + "\n\nKey factors in this recommendation:"
    for point in model_explanations.get(recommended_model, []):
        specific_explanation += f"\n- {point}"
    
    return recommended_model, specific_explanation, scores

def consider_ensemble(model_scores, analysis):
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    top_model, top_score = sorted_models[0]
    runner_up, runner_up_score = sorted_models[1]
    
    ensemble_recommendation = None
    explanation = ""
    
    score_difference_ratio = (top_score - runner_up_score) / max(top_score, 1)
    
    ensemble_favored = False
    reasons = []
    
    if score_difference_ratio < 0.2:
        ensemble_favored = True
        reasons.append(f"The top models ({top_model} and {runner_up}) have similar performance scores.")
    
    if analysis.get('complex_patterns', False):
        ensemble_favored = True
        reasons.append("The dataset exhibits complex patterns that could benefit from multiple modeling approaches.")
    
    if analysis.get('size') == 'medium' or analysis.get('size') == 'large':
        ensemble_favored = True
        reasons.append("The dataset is large enough to support training multiple models effectively.")
    
    if analysis.get('has_seasonality', False) and analysis.get('complex_patterns', False):
        ensemble_favored = True
        reasons.append("The data shows both seasonality and complex patterns.")
    
    if analysis.get('variables', 1) > 2:
        ensemble_favored = True
        reasons.append("The multivariate nature of the data suggests different models may excel at capturing different relationships.")
    
    if ensemble_favored and len(reasons) >= 2:
        ensemble_pairs = []
        
        if 'SARIMA' in [top_model, runner_up] and 'XGBoost' in [top_model, runner_up]:
            ensemble_pairs.append(('SARIMA', 'XGBoost', "SARIMA captures seasonal components while XGBoost models non-linear relationships"))
        if 'Prophet' in [top_model, runner_up] and 'XGBoost' in [top_model, runner_up]:
            ensemble_pairs.append(('Prophet', 'XGBoost', "Prophet handles trends and seasonality while XGBoost captures non-linear relationships"))
        if 'LSTM' in [top_model, runner_up] and 'SARIMA' in [top_model, runner_up]:
            ensemble_pairs.append(('LSTM', 'SARIMA', "LSTM models complex temporal patterns while SARIMA provides statistical rigor"))
        if 'Prophet' in [top_model, runner_up] and 'LSTM' in [top_model, runner_up]:
            ensemble_pairs.append(('Prophet', 'LSTM', "Prophet decomposes seasonality while LSTM captures complex dependencies"))
        
        if not ensemble_pairs:
            if analysis.get('has_seasonality', False) and analysis.get('complex_patterns', False):
                ensemble_pairs.append(('Prophet', 'XGBoost', "Prophet for seasonality and XGBoost for complex patterns"))
            elif analysis.get('has_seasonality', False) and analysis.get('size') == 'large':
                ensemble_pairs.append(('SARIMA', 'LSTM', "SARIMA for statistical modeling of seasonality and LSTM for complex patterns"))
            elif analysis.get('variables', 1) > 2 and not analysis.get('has_seasonality', False):
                ensemble_pairs.append(('XGBoost', 'LSTM', "XGBoost and LSTM both handle multivariate data with different approaches"))
            else:
                ensemble_pairs.append((top_model, runner_up, f"Combining the strengths of the top two models ({top_model} and {runner_up})"))
        
        if ensemble_pairs:
            model1, model2, pair_reason = ensemble_pairs[0]
            ensemble_recommendation = {
                'models': [model1, model2],
                'ensemble_method': 'weighted_average',
                'weights': [0.5, 0.5]
            }
            
            if model1 in model_scores and model2 in model_scores:
                total = model_scores[model1] + model_scores[model2]
                if total > 0:
                    ensemble_recommendation['weights'] = [
                        round(model_scores[model1] / total, 2),
                        round(model_scores[model2] / total, 2)
                    ]
            
            explanation = f"An ensemble approach is recommended combining {model1} and {model2}. {pair_reason}.\n\n"
            explanation += "Reasons for recommending an ensemble:\n"
            for reason in reasons:
                explanation += f"- {reason}\n"
    
    return ensemble_recommendation, explanation