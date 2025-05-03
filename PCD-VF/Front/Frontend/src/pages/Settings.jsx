import React, { useState } from 'react';
import { useAppContext } from '../context/AppContext';
import Swal from 'sweetalert2';
import './Settings.css';

const Settings = () => {
  const { appState, setAppState, setProjectId } = useAppContext();
  const [projectNameInput, setProjectNameInput] = useState(appState.projectName);
  const [isSubmitted, setIsSubmitted] = useState(!!appState.projectName && !!appState.projectId); // Vérifie si déjà soumis

  // Fonction pour générer l'ID du projet
  const handleGenerateId = () => {
    if (!projectNameInput.trim()) {
      Swal.fire({
        title: 'Error',
        text: 'Please enter a project name before generating an ID.',
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
      return;
    }

    const projectId = setProjectId(projectNameInput);
    Swal.fire({
      title: 'ID Generated',
      text: `Project ID: ${projectId}`,
      icon: 'success',
      timer: 1500,
      showConfirmButton: false,
    });
  };

  // Fonction pour gérer la soumission du formulaire
  const handleSubmit = (e) => {
    e.preventDefault();
    if (!projectNameInput.trim()) {
      Swal.fire({
        title: 'Error',
        text: 'Please enter a project name.',
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
      return;
    }

    if (!appState.projectId) {
      Swal.fire({
        title: 'Error',
        text: 'Please generate a Project ID before submitting.',
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
      return;
    }

    setAppState((prevState) => ({
      ...prevState,
      projectName: projectNameInput,
    }));

    setIsSubmitted(true);

    Swal.fire({
      title: 'Settings Saved',
      text: 'Project settings have been saved successfully.',
      icon: 'success',
      timer: 1500,
      showConfirmButton: false,
    });
  };

  return (
    <div className="main-content">
      <div className="page-content">
        <h3>Project Settings</h3>
        {isSubmitted ? (
          <div className="settings-display">
            <p><strong>Project Name:</strong> {appState.projectName}</p>
            <p><strong>Project ID:</strong> {appState.projectId}</p>
          </div>
        ) : (
          <div className="settings-form">
            <label>
              Project Name:
              <input
                type="text"
                value={projectNameInput}
                onChange={(e) => setProjectNameInput(e.target.value)}
                placeholder="Enter project name"
              />
            </label>
            <div className="settings-actions">
              <button className="generate-btn" onClick={handleGenerateId}>
                Generate ID
              </button>
              <button
                className="submit-btn"
                onClick={handleSubmit}
                disabled={!appState.projectId} // Désactiver si aucun ID n'est généré
              >
                Submit
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Settings;