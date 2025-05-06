import React, { useEffect } from 'react';
import { useAppContext } from '../context/AppContext'; // Import du contexte
import './Models.css'; // Chemin corrigé pour les styles

const Models = () => {
  const { appState, setAppState } = useAppContext();

  // Fonction pour gérer la sélection des modèles
  const handleChange = (modelName) => {
    setAppState((prevState) => {
      const selectedModels = prevState.selectedModels.includes(modelName)
        ? prevState.selectedModels.filter((m) => m !== modelName) // Désélectionner
        : [...prevState.selectedModels, modelName]; // Sélectionner
      return { ...prevState, selectedModels };
    });
  };

  // Envoyer les modèles sélectionnés au backend à chaque changement
  useEffect(() => {
    const sendSelectedModels = async () => {
      try {
        const response = await fetch('http://localhost:5001/set_selected_models', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({ selectedModels: appState.selectedModels }),
        });
        const data = await response.json();
        if (data.status !== 'success') {
          console.error('Failed to send selected models:', data.message);
        }
      } catch (error) {
        console.error('Error sending selected models:', error);
      }
    };

    sendSelectedModels();
  }, [appState.selectedModels]); // Déclencher à chaque changement de selectedModels

  const models = ['SARIMA', 'LSTM', 'PROPHET', 'XGBOOST'];

  return (
    <>
      <div className="main-content">
        <div className="page-content">
          <div className="model-options">
            {models.map((model) => (
              <label
                key={model}
                className={appState.selectedModels.includes(model) ? 'selected' : ''}
              >
                <input
                  type="checkbox"
                  checked={appState.selectedModels.includes(model)}
                  onChange={() => handleChange(model)}
                />
                {model}
              </label>
            ))}
          </div>
        </div>
      </div>
    </>
  );
};

export default Models;