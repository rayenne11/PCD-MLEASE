from EDA import run_eda
from Preprocessing import TimeSeriesPreprocessor
from ModelSelection import analyze_timeseries
from Sarima import train_sarima
from Prophet import train_prophet
from Xgboost import train_xgboost
from Lstm import train_lstm

import prefect
from prefect import Flow, task, Parameter
import pandas as pd
import mlflow
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from threading import Thread
import threading
import os
import time
from mlflow.tracking import MlflowClient
import tempfile
import logging

# Configurer les logs
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configurer MLflow
mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

# Terminer tous les runs MLflow actifs au démarrage
def clean_active_mlflow_runs():
    active_run = mlflow.active_run()
    if active_run:
        mlflow.end_run()
    logger.info("Cleaned up active MLflow runs.")

# Initialiser Flask
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Variable globale pour suivre l'état du pipeline
pipeline_state = {
    "current_step": "Not Started",
    "status": "inactive",
    "completed_steps": [],
    "selected_models": [],
    "recommended_models": [],
}

# Variable pour stocker les données importées
imported_data_path = None
default_output_path = "uploaded_data.csv"

# Variable pour stocker le nom du dernier fichier uploadé
last_uploaded_file_name = "Unknown"

# Variable pour vérifier si le pipeline est déjà en cours d'exécution
pipeline_running = False
pipeline_thread = None

# Définir les tâches Prefect avec mise à jour de l'état
@task
def run_eda_task(data_path: str, output_report_path: str):
    global pipeline_state
    pipeline_state["current_step"] = "EDA"
    pipeline_state["status"] = "active"
    with mlflow.start_run(run_name="EDA", nested=True):
        df = pd.read_csv(data_path, parse_dates=True, index_col=0)
        run_eda(df, output_report_path)
        mlflow.log_artifact(output_report_path, artifact_path="eda_report")
        if os.path.exists("index.html"):
            mlflow.log_artifact("index.html", artifact_path="eda_html_report")
        artifacts = client.list_artifacts(mlflow.active_run().info.run_id)
        logger.info(f"EDA artifacts logged: {[artifact.path for artifact in artifacts]}")
    time.sleep(3)
    pipeline_state["status"] = "completed"
    pipeline_state["completed_steps"].append("EDA")
    return output_report_path

@task
def preprocess_task(data_path: str, eda_report_path: str, output_processed_path: str):
    global pipeline_state
    pipeline_state["current_step"] = "Preprocessing"
    pipeline_state["status"] = "active"
    with mlflow.start_run(run_name="Preprocessing", nested=True):
        preprocessor = TimeSeriesPreprocessor(eda_report_path)
        df = pd.read_csv(data_path, parse_dates=True, index_col=0)
        df_processed = preprocessor.preprocess(df)
        df_processed.to_csv(output_processed_path)
        mlflow.log_artifact(output_processed_path, artifact_path="preprocessed_data")
    time.sleep(3)
    pipeline_state["status"] = "completed"
    pipeline_state["completed_steps"].append("Preprocessing")
    return output_processed_path

@task
def model_selection_task(processed_data_path: str):
    global pipeline_state
    pipeline_state["current_step"] = "Evaluate"
    pipeline_state["status"] = "active"
    with mlflow.start_run(run_name="ModelSelection", nested=True):
        df = pd.read_csv(processed_data_path, parse_dates=True, index_col=0)
        if df.empty or len(df.columns) == 0:
            raise ValueError("Processed data is empty or has no columns.")
        target_col = df.columns[-1]
        result = analyze_timeseries(df, target_col=target_col)
        recommended_approach = result["recommended_approach"]
        mlflow.log_param("recommended_approach", recommended_approach)
        if recommended_approach == "ensemble":
            recommended_models = result["ensemble_details"]["models"]
        else:
            recommended_models = [result["recommended_model"]]
        mlflow.log_param("recommended_models", recommended_models)
    time.sleep(3)
    pipeline_state["status"] = "completed"
    pipeline_state["completed_steps"].append("Evaluate")
    pipeline_state["recommended_models"] = recommended_models
    return recommended_models

@task
def train_models_task(processed_data_path: str, selected_models: list):
    global pipeline_state
    df = pd.read_csv(processed_data_path, parse_dates=True, index_col=0)
    results = {}

    models_to_train = pipeline_state["selected_models"] if pipeline_state["selected_models"] else (selected_models if selected_models else ["Prophet"])

    for model_name in models_to_train:
        step_name = f"Training {model_name}"
        pipeline_state["current_step"] = step_name
        pipeline_state["status"] = "active"
        with mlflow.start_run(run_name=f"Training_{model_name}", nested=True) as run:
            mlflow.log_param("model_name", model_name)
            if model_name == "SARIMA":
                for target_col in df.columns:
                    result = train_sarima(df, target_col)
                    results["SARIMA"] = result
                    mlflow.log_metric(f"mse_{target_col}", result.get("mse", 0))
            elif model_name == "Prophet":
                for target_col in df.columns:
                    result = train_prophet(df, target_col)
                    results["Prophet"] = result
                    mlflow.log_metric(f"mse_{target_col}", result.get("mse", 0))
            elif model_name == "XGBoost":
                for target_col in df.columns:
                    result = train_xgboost(df, target_col)
                    results["XGBoost"] = result
                    mlflow.log_metric(f"mse_{target_col}", result.get("mse", 0))
            elif model_name == "LSTM":
                for target_col in df.columns:
                    result = train_lstm(df, target_col)
                    results["LSTM"] = result
                    mlflow.log_metric(f"mse_{target_col}", result.get("mse", 0))
            # Attendre que MLflow finalise la journalisation des artefacts
            time.sleep(2)
            artifacts = client.list_artifacts(run.info.run_id)
            logger.info(f"{model_name} artifacts logged: {[artifact.path for artifact in artifacts]}")
        pipeline_state["status"] = "completed"
        pipeline_state["completed_steps"].append(step_name)

    pipeline_state["current_step"] = "Deployment"
    pipeline_state["status"] = "completed"
    pipeline_state["completed_steps"].append("Deployment")
    time.sleep(1)
    return results

# Route Flask pour uploader les données
@app.route('/upload_data', methods=['POST'])
def upload_data():
    global imported_data_path, last_uploaded_file_name
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part in the request."}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No file selected."}), 400
    if not file.filename.endswith('.csv'):
        return jsonify({"status": "error", "message": "File must be a CSV."}), 400
    if file:
        file.save(default_output_path)
        try:
            df = pd.read_csv(default_output_path, parse_dates=True, index_col=0)
            if df.empty:
                return jsonify({"status": "error", "message": "Uploaded CSV file is empty."}), 400
            with mlflow.start_run(run_name="UploadData"):
                mlflow.log_param("data_file_name", file.filename)
            # Mettre à jour le nom du dernier fichier uploadé
            last_uploaded_file_name = file.filename
        except Exception as e:
            return jsonify({"status": "error", "message": f"Invalid CSV file: {str(e)}"}), 400
        imported_data_path = default_output_path
        return jsonify({"status": "success", "message": "File uploaded successfully."})

# Route Flask pour servir le fichier index.html
@app.route('/get_eda_report', methods=['GET'])
def get_eda_report():
    try:
        if os.path.exists("index.html"):
            return send_file("index.html")
        else:
            return jsonify({"status": "error", "message": "EDA report (index.html) not found. Please run the pipeline first."}), 404
    except Exception as e:
        return jsonify({"status": "error", "message": f"Error retrieving EDA report: {str(e)}"}), 500

# Route pour recevoir les modèles sélectionnés depuis Models.jsx
@app.route('/set_selected_models', methods=['POST'])
def set_selected_models():
    global pipeline_state
    data = request.get_json()
    if not data or 'selectedModels' not in data:
        return jsonify({"status": "error", "message": "No selectedModels provided."}), 400

    selected_models = data['selectedModels']
    standardized_models = []
    for model in selected_models:
        if model.upper() == "PROPHET":
            standardized_models.append("Prophet")
        elif model.upper() == "XGBOOST":
            standardized_models.append("XGBoost")
        else:
            standardized_models.append(model)

    pipeline_state["selected_models"] = standardized_models
    return jsonify({"status": "success", "message": "Selected models updated successfully."})

# Route pour servir les fichiers artefacts (images .png)
@app.route('/serve_artifact', methods=['GET'])
def serve_artifact():
    try:
        run_id = request.args.get('run_id')
        path = request.args.get('path')
        if not run_id or not path:
            return jsonify({"status": "error", "message": "Missing run_id or path parameter."}), 400
        
        logger.info(f"Serving artifact - run_id: {run_id}, path: {path}")
        local_path = client.download_artifacts(run_id, path)
        logger.info(f"Artifact downloaded to: {local_path}")
        return send_file(local_path, mimetype='image/png')
    except Exception as e:
        logger.error(f"Error serving artifact: {str(e)}")
        return jsonify({"status": "error", "message": f"Error serving artifact: {str(e)}"}), 500

# Route pour récupérer les artefacts MLflow
@app.route('/get_mlflow_artifacts', methods=['GET'])
def get_mlflow_artifacts():
    try:
        experiment_id = "0"
        runs = client.search_runs(experiment_ids=[experiment_id], order_by=["start_time DESC"])
        
        artifacts_by_model = {}
        eda_artifacts = {
            "metrics": [],
            "images": [],
            "html_reports": [],
            "run_id": None,
            "start_time": 0
        }
        
        def list_artifacts_recursive(run_id, path=""):
            """Parcourir récursivement les artefacts dans un run MLflow."""
            artifacts = client.list_artifacts(run_id, path)
            result = []
            for artifact in artifacts:
                if artifact.is_dir:
                    # Si c'est un dossier, parcourir récursivement
                    sub_artifacts = list_artifacts_recursive(run_id, artifact.path)
                    result.extend(sub_artifacts)
                else:
                    # Si c'est un fichier, l'ajouter à la liste
                    result.append(artifact)
            return result
        
        for run in runs:
            run_name = run.data.tags.get("mlflow.runName", "")
            logger.info(f"Processing run: {run_name}, run_id: {run.info.run_id}")
            
            if run_name == "UploadData":
                continue
                
            if run_name == "EDA":
                eda_artifacts["run_id"] = run.info.run_id
                eda_artifacts["start_time"] = run.info.start_time
                for metric_name, metric_value in run.data.metrics.items():
                    eda_artifacts["metrics"].append({
                        "name": metric_name,
                        "value": metric_value
                    })
                artifacts = list_artifacts_recursive(run.info.run_id)
                logger.info(f"EDA artifacts found: {[artifact.path for artifact in artifacts]}")
                for artifact in artifacts:
                    if artifact.path.endswith(".png"):
                        artifact_url = f"http://localhost:5001/serve_artifact?run_id={run.info.run_id}&path={artifact.path}"
                        eda_artifacts["images"].append({
                            "url": artifact_url,
                            "path": artifact.path
                        })
                    elif artifact.path.endswith(".html"):
                        artifact_url = f"http://localhost:5001/serve_artifact?run_id={run.info.run_id}&path={artifact.path}"
                        eda_artifacts["html_reports"].append({
                            "url": artifact_url,
                            "path": artifact.path
                        })
                continue
                
            # Vérifier les runs principaux comme "Training_Prophet"
            model_name = None
            if run_name.startswith("Training_"):
                model_name = run_name.replace("Training_", "")
            # Vérifier les runs imbriqués comme "Prophet_TRFVOLUSM227NFWA"
            elif run_name.startswith("Prophet_") or run_name.startswith("SARIMA_") or run_name.startswith("XGBoost_") or run_name.startswith("LSTM_"):
                model_name = run_name.split("_")[0]
            
            if not model_name:
                continue
                
            if model_name not in artifacts_by_model:
                artifacts_by_model[model_name] = {
                    "metrics": [],
                    "images": [],
                    "run_id": run.info.run_id,
                    "start_time": run.info.start_time
                }
            
            for metric_name, metric_value in run.data.metrics.items():
                artifacts_by_model[model_name]["metrics"].append({
                    "name": metric_name,
                    "value": metric_value
                })
            
            artifacts = list_artifacts_recursive(run.info.run_id)
            logger.info(f"{model_name} artifacts found: {[artifact.path for artifact in artifacts]}")
            for artifact in artifacts:
                # Vérifier les fichiers .png dans le sous-répertoire model_plots
                if artifact.path.startswith("model_plots/") and artifact.path.endswith(".png"):
                    artifact_url = f"http://localhost:5001/serve_artifact?run_id={run.info.run_id}&path={artifact.path}"
                    artifacts_by_model[model_name]["images"].append({
                        "url": artifact_url,
                        "path": artifact.path
                    })
        
        artifacts_list = [
            {"model": model, **data} for model, data in artifacts_by_model.items()
        ]
        artifacts_list.sort(key=lambda x: x["start_time"], reverse=True)
        
        return jsonify({
            "status": "success",
            "artifacts": artifacts_list,
            "eda_artifacts": eda_artifacts,
            "data_file_name": last_uploaded_file_name
        })
    except Exception as e:
        logger.error(f"Error retrieving MLflow artifacts: {str(e)}")
        return jsonify({"status": "error", "message": f"Error retrieving MLflow artifacts: {str(e)}"}), 500

# Route pour obtenir l'état du pipeline
@app.route('/pipeline_status', methods=['GET'])
def get_pipeline_status():
    return jsonify(pipeline_state)

# Route pour relancer le pipeline
@app.route('/start_pipeline', methods=['POST'])
def start_pipeline():
    global pipeline_thread, pipeline_running, imported_data_path
    if pipeline_running:
        return jsonify({"status": "error", "message": "Pipeline is already running."}), 400
    if not imported_data_path:
        return jsonify({"status": "error", "message": "No data uploaded. Please upload a dataset first."}), 400
    pipeline_state["current_step"] = "Not Started"
    pipeline_state["status"] = "inactive"
    pipeline_state["completed_steps"] = []
    pipeline_thread = Thread(target=run_pipeline)
    pipeline_thread.start()
    return jsonify({"status": "success", "message": "Pipeline started successfully."})

# Route pour réinitialiser le pipeline
@app.route('/reset_pipeline', methods=['POST'])
def reset_pipeline():
    global pipeline_state, last_uploaded_file_name
    pipeline_state["current_step"] = "Not Started"
    pipeline_state["status"] = "inactive"
    pipeline_state["completed_steps"] = []
    pipeline_state["selected_models"] = []
    last_uploaded_file_name = "Unknown"  # Réinitialiser le nom du fichier
    return jsonify({"status": "success", "message": "Pipeline state reset successfully."})

# Définir le workflow Prefect
with Flow("TimeSeriesPipeline") as flow:
    data_path = Parameter("data_path")
    eda_report_path = Parameter("eda_report_path", default="mlops_eda_report.json")
    processed_data_path = Parameter("processed_data_path", default="preprocessed_timeseries.csv")
    selected_models = Parameter("selected_models", default=None)

    eda_output = run_eda_task(data_path, eda_report_path)
    processed_output = preprocess_task(data_path, eda_output, processed_data_path)
    recommended_models = model_selection_task(processed_output)
    train_results = train_models_task(processed_output, recommended_models)

# Fonction pour exécuter le pipeline dans un thread séparé
def run_pipeline():
    global pipeline_running, imported_data_path
    pipeline_running = True
    try:
        with mlflow.start_run(run_name="TimeSeriesPipeline"):
            flow.run(parameters={"data_path": imported_data_path})
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
    finally:
        pipeline_running = False

# Lancer le serveur Flask
if __name__ == "__main__":
    clean_active_mlflow_runs()
    app.run(host='0.0.0.0', port=5001, debug=True)