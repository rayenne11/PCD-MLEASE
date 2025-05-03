import React, { useState, useEffect } from 'react';
import Swal from 'sweetalert2';
import './Pipelines.css';

const Pipelines = () => {
  const [pipelineState, setPipelineState] = useState({
    current_step: "Not Started",
    status: "inactive",
    completed_steps: [],
    selected_models: [],
    recommended_models: [], // Ajout pour gérer les modèles recommandés
  });
  const [error, setError] = useState(null);

  // Interroger l'API Flask pour obtenir l'état du pipeline
  useEffect(() => {
    const fetchPipelineStatus = async () => {
      try {
        const response = await fetch('http://localhost:5001/pipeline_status');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setPipelineState(data);
        setError(null);
      } catch (error) {
        console.error('Error fetching pipeline status:', error);
        setError('Unable to connect to the backend. Please ensure the server is running on port 5001.');
      }
    };

    fetchPipelineStatus();
    const interval = setInterval(fetchPipelineStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  // Définir les étapes
  const baseStepsBeforeTraining = ["EDA", "Preprocessing", "Evaluate"];
  // Utiliser selected_models si non vide, sinon utiliser recommended_models
  const modelsToTrain = pipelineState.selected_models.length > 0 ? pipelineState.selected_models : (pipelineState.recommended_models || []);
  const trainingSteps = modelsToTrain.map(model => `Training ${model}`);
  const baseStepsAfterTraining = ["Deployment"];
  const allSteps = [...baseStepsBeforeTraining, ...trainingSteps, ...baseStepsAfterTraining];

  // Déterminer l'état de chaque étape
  const getStepStatus = (step) => {
    if (step === pipelineState.current_step && pipelineState.status === "active") {
      return "active";
    } else if (pipelineState.completed_steps.includes(step)) {
      return "completed";
    } else {
      return "inactive";
    }
  };

  // Afficher une pop-up si une erreur persiste
  useEffect(() => {
    if (error) {
      Swal.fire({
        title: 'Connection Error',
        text: error,
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
    }
  }, [error]);

  return (
    <div className="main-content">
      <div className="page-content">
        <h3>Pipeline Status</h3>
        {error ? (
          <p className="error-message">{error}</p>
        ) : (
          <div className="pipeline-container">
            {/* Étapes avant les entraînements (horizontales) */}
            {baseStepsBeforeTraining.map((step, index) => (
              <React.Fragment key={index}>
                <div className={`pipeline-step ${getStepStatus(step)}`}>
                  <span className="step-label">{step}</span>
                  {getStepStatus(step) === "completed" && (
                    <span className="checkmark">✔</span>
                  )}
                </div>
                {index < baseStepsBeforeTraining.length - 1 && (
                  <div className="pipeline-arrow">
                    <svg width="40" height="20" viewBox="0 0 40 20">
                      <line x1="0" y1="10" x2="30" y2="10" stroke="#6b7280" strokeWidth="2" />
                      <polygon points="30,5 40,10 30,15" fill="#6b7280" />
                    </svg>
                  </div>
                )}
              </React.Fragment>
            ))}

            {/* Flèche avant les étapes d'entraînement */}
            {trainingSteps.length > 0 && (
              <div className="pipeline-arrow">
                <svg width="40" height="20" viewBox="0 0 40 20">
                  <line x1="0" y1="10" x2="30" y2="10" stroke="#6b7280" strokeWidth="2" />
                  <polygon points="30,5 40,10 30,15" fill="#6b7280" />
                </svg>
              </div>
            )}

            {/* Étapes d'entraînement (verticales) */}
            {trainingSteps.length > 0 && (
              <div className="training-steps-container">
                {trainingSteps.map((step, index) => (
                  <React.Fragment key={index}>
                    <div className={`pipeline-step training-step ${getStepStatus(step)}`}>
                      <span className="step-label">{step}</span>
                      {getStepStatus(step) === "completed" && (
                        <span className="checkmark">✔</span>
                      )}
                    </div>
                    {index < trainingSteps.length - 1 && (
                      <div className="pipeline-arrow-vertical">
                        <svg width="20" height="40" viewBox="0 0 20 40">
                          <line x1="10" y1="0" x2="10" y2="30" stroke="#6b7280" strokeWidth="2" />
                          <polygon points="5,30 10,40 15,30" fill="#6b7280" />
                        </svg>
                      </div>
                    )}
                  </React.Fragment>
                ))}
              </div>
            )}

            {/* Flèche après les étapes d'entraînement */}
            {trainingSteps.length > 0 && (
              <div className="pipeline-arrow">
                <svg width="40" height="20" viewBox="0 0 40 20">
                  <line x1="0" y1="10" x2="30" y2="10" stroke="#6b7280" strokeWidth="2" />
                  <polygon points="30,5 40,10 30,15" fill="#6b7280" />
                </svg>
              </div>
            )}

            {/* Étapes après les entraînements (horizontales) */}
            {baseStepsAfterTraining.map((step, index) => (
              <React.Fragment key={index}>
                <div className={`pipeline-step ${getStepStatus(step)}`}>
                  <span className="step-label">{step}</span>
                  {getStepStatus(step) === "completed" && (
                    <span className="checkmark">✔</span>
                  )}
                </div>
              </React.Fragment>
            ))}
          </div>
        )}
      </div>
    </div>
  );
};

export default Pipelines;