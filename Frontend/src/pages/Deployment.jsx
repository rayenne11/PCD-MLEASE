import React, { useState, useEffect } from 'react';
import axios from 'axios';

const Deployment = () => {
  const [deployedModel, setDeployedModel] = useState(null);
  const [inputData, setInputData] = useState('');
  const [predictions, setPredictions] = useState(null);
  const [error, setError] = useState(null);

  // Récupérer les informations du modèle déployé au chargement
  useEffect(() => {
    axios.get('http://localhost:5001/pipeline_status')
      .then(response => {
        const deployed = response.data.deployed_model;
        if (deployed) {
          setDeployedModel(deployed);
        } else {
          setError('No model deployed. Please run the pipeline to deploy a model.');
        }
      })
      .catch(err => {
        setError('Failed to fetch deployed model. Please ensure the backend is running.');
        console.error(err);
      });
  }, []);

  // Gérer la modification des données d'entrée
  const handleInputChange = (e) => {
    setInputData(e.target.value);
  };

  // Envoyer la requête de prédiction
  const handlePredict = () => {
    if (!inputData) {
      setError('Please provide input data.');
      return;
    }

    let requestData;
    if (deployedModel && deployedModel.model_name === "Prophet") {
      const inputArray = inputData.split(',').map(val => val.trim());
      if (inputArray.length === 0) {
        setError('Invalid input data. Please provide comma-separated dates.');
        return;
      }
      requestData = {
        model_name: deployedModel.model_name,
        input_data: { dates: inputArray }
      };
    } else if (deployedModel && deployedModel.model_name === "XGBoost") {
      try {
        const parsedInput = JSON.parse(inputData);
        if (!parsedInput || typeof parsedInput !== 'object') {
          setError('Invalid input data. Please provide a valid JSON object with feature names and lists of values.');
          return;
        }
        requestData = {
          model_name: deployedModel.model_name,
          input_data: parsedInput
        };
      } catch (e) {
        setError('Invalid JSON format. Please provide a valid JSON object (e.g., {"lag1": [1, 2, 3], "lag2": [0.5, 1.0, 1.5]}).');
        return;
      }
    } else {
      setError('Prediction not implemented for this model type.');
      setPredictions(null);
      return;
    }

    axios.post('http://localhost:5001/predict', requestData)
      .then(response => {
        if (response.data.status === 'success') {
          setPredictions(response.data.predictions);
          setError(null);
        } else {
          setError(response.data.message || 'Prediction failed.');
          setPredictions(null);
        }
      })
      .catch(err => {
        setError('Failed to get predictions. Please check the input data and backend.');
        setPredictions(null);
        console.error(err);
      });
  };

  return (
    <div style={{ padding: '50px', paddingLeft: '80px' }}>
      {deployedModel ? (
        <div>
          <h3>Deployed Model: {deployedModel.model_name}</h3>
          <p>Run ID: {deployedModel.run_id}</p>
          <p>Version: {deployedModel.version}</p>
          
          <div style={{ marginTop: '20px' }}>
            <label>
              Input Data (
              {deployedModel.model_name === "Prophet" 
                ? "comma-separated dates, YYYY-MM-DD" 
                : "JSON object with features, e.g., {\"lag1\": [1, 2, 3], \"lag2\": [0.5, 1.0, 1.5]}"}):
              <input
                type="text"
                value={inputData}
                onChange={handleInputChange}
                placeholder={
                  deployedModel.model_name === "Prophet"
                    ? "e.g., 2025-02-01, 2025-02-02, 2025-02-03"
                    : "e.g., {\"lag1\": [1, 2, 3], \"lag2\": [0.5, 1.0, 1.5]}"
                }
                style={{ marginLeft: '10px', padding: '5px', width: '500px' }}
              />
            </label>
            <button
              onClick={handlePredict}
              style={{
                marginLeft: '12px',
                padding: '8px 15px',
                backgroundColor: '#10b981',
                color: 'white',
                border: 'none',
                borderRadius: '7px',
                cursor: 'pointer'
              }}
            >
              Predict
            </button>
          </div>

          {predictions && (
            <div style={{ marginTop: '20px' }}>
              <h4>Predictions:</h4>
              <pre>{JSON.stringify(predictions, null, 2)}</pre>
            </div>
          )}

          {error && (
            <p style={{ color: 'red', marginTop: '10px' }}>{error}</p>
          )}
        </div>
      ) : (
        <p>{error || 'No model deployed. Please run the pipeline to train and deploy a model.'}</p>
      )}
    </div>
  );
};

export default Deployment;