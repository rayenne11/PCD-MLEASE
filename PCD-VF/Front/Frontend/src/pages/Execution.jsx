import React, { useState, useEffect } from 'react';
import { useAppContext } from '../context/AppContext';
import Swal from 'sweetalert2';
import './Execution.css';

const Execution = () => {
  const { appState, setAppState } = useAppContext();
  const [isRunning, setIsRunning] = useState(false);
  const [showFileSelectionModal, setShowFileSelectionModal] = useState(false);
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [selectedFileIndex, setSelectedFileIndex] = useState(null);
  const [selectedExecutionIndex, setSelectedExecutionIndex] = useState(null);
  const [pipelineStatus, setPipelineStatus] = useState({});

  // Vérifier l'état du pipeline toutes les 2 secondes
  useEffect(() => {
    const checkPipelineStatus = async () => {
      try {
        const response = await fetch('http://localhost:5001/pipeline_status');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        setIsRunning(data.status === 'active');
        setPipelineStatus(data);
      } catch (error) {
        console.error('Error fetching pipeline status:', error);
        // Swal.fire({
        //   title: 'Connection Error',
        //   text: 'Unable to connect to the backend. Please ensure the server is running on port 5001.',
        //   icon: 'error',
        //   confirmButtonColor: '#f97316',
        // });
      }
    };

    checkPipelineStatus();
    const interval = setInterval(checkPipelineStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  // Déterminer l'état d'une exécution
  const getExecutionStatus = (execution) => {
    const lastExecutionId = appState.executions.length;
    if (execution.id === lastExecutionId) {
      if (pipelineStatus.completed_steps && pipelineStatus.completed_steps.includes('Deployment')) {
        return 'Done';
      }
      return 'In Progress';
    }
    return 'Done';
  };

  // Fonction pour gérer le basculement entre Run et Stop
  const handleToggleRunStop = () => {
    if (isRunning) {
      Swal.fire({
        title: 'Pipeline Running',
        text: 'The pipeline is currently running. Stopping is not supported in this version. Please wait for it to complete.',
        icon: 'info',
        confirmButtonColor: '#f97316',
      });
    } else {
      handleCreateExecution();
    }
  };

  // Fonction pour créer une exécution et lancer le pipeline
  const handleCreateExecution = async () => {
    if (!appState.projectName) {
      Swal.fire({
        title: 'Error',
        text: 'Please set a project name in Settings first.',
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
      return;
    }

    if (appState.dataFiles.length === 0) {
      Swal.fire({
        title: 'Error',
        text: 'Please import at least one data file in Data section.',
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
      return;
    }

    if (appState.dataFiles.length === 1) {
      createExecution(0);
    } else {
      setShowFileSelectionModal(true);
    }
  };

  // Fonction pour créer une exécution après sélection du fichier
  const createExecution = async (fileIndex) => {
    const selectedFile = appState.dataFiles[fileIndex];

    try {
      const pipelineResponse = await fetch('http://localhost:5001/start_pipeline', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      const pipelineData = await pipelineResponse.json();
      if (pipelineData.status !== 'success') {
        Swal.fire({
          title: 'Error',
          text: pipelineData.message || 'Failed to start the pipeline.',
          icon: 'error',
          confirmButtonColor: '#f97316',
        });
        return;
      }

      const newExecution = {
        id: appState.executions.length + 1,
        projectId: appState.projectId || 'N/A',
        name: `Execution ${appState.executions.length + 1}`,
        file: selectedFile.name,
        date: new Date().toLocaleString('fr-FR', {
          day: '2-digit',
          month: '2-digit',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
        }),
      };

      setAppState((prevState) => ({
        ...prevState,
        executions: [...prevState.executions, newExecution],
      }));

      setIsRunning(true);

      Swal.fire({
        title: 'Execution Started',
        text: `Pipeline started with file: ${selectedFile.name}`,
        icon: 'success',
        timer: 1500,
        showConfirmButton: false,
      });
    } catch (error) {
      Swal.fire({
        title: 'Error',
        text: 'Failed to start the pipeline. Please ensure the backend is running.',
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
    }
  };

  // Fonction pour confirmer la sélection du fichier
  const confirmFileSelection = () => {
    if (selectedFileIndex === null) {
      Swal.fire({
        title: 'Error',
        text: 'Please select a file to proceed.',
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
      return;
    }

    createExecution(selectedFileIndex);
    setShowFileSelectionModal(false);
    setSelectedFileIndex(null);
  };

  // Fonction pour fermer la pop-up sans sélection
  const closeModal = () => {
    setShowFileSelectionModal(false);
    setSelectedFileIndex(null);
  };

  // Fonction pour gérer le clic sur le bouton Delete
  const handleDeleteClick = () => {
    if (appState.executions.length === 0) {
      Swal.fire({
        title: 'Error',
        text: 'There are no executions to delete.',
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
      return;
    }
    setShowDeleteModal(true);
  };

  // Fonction pour confirmer la suppression de l'exécution
  const confirmDelete = async () => {
    if (selectedExecutionIndex === null) {
      Swal.fire({
        title: 'Error',
        text: 'Please select an execution to delete.',
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
      return;
    }

    Swal.fire({
      title: 'Are you sure?',
      text: 'Do you really want to delete this execution? This action cannot be undone.',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonColor: '#f97316',
      cancelButtonColor: '#e5e7eb',
      confirmButtonText: 'Yes, delete it!',
      cancelButtonText: 'Cancel',
    }).then(async (result) => {
      if (result.isConfirmed) {
        // Vérifier si l'exécution supprimée est la dernière (associée au pipeline graphique)
        const isLastExecution = selectedExecutionIndex === appState.executions.length - 1;

        // Supprimer l'exécution
        setAppState((prevState) => ({
          ...prevState,
          executions: prevState.executions.filter((_, index) => index !== selectedExecutionIndex),
        }));

        // Si c'est la dernière exécution, réinitialiser le pipeline graphique
        if (isLastExecution) {
          try {
            const response = await fetch('http://localhost:5001/reset_pipeline', {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
            });
            const data = await response.json();
            if (data.status !== 'success') {
              console.error('Failed to reset pipeline:', data.message);
            }
          } catch (error) {
            console.error('Error resetting pipeline:', error);
          }
        }

        Swal.fire({
          title: 'Deleted!',
          text: 'The execution has been successfully deleted.',
          icon: 'success',
          timer: 1500,
          showConfirmButton: false,
        });
        setShowDeleteModal(false);
        setSelectedExecutionIndex(null);
      }
    });
  };

  // Fonction pour fermer la pop-up de suppression sans supprimer
  const closeDeleteModal = () => {
    setShowDeleteModal(false);
    setSelectedExecutionIndex(null);
  };

  return (
    <>
      <div className="main-content">
        <div className="page-content">
          <div className="execution-actions">
            <button className="action-btn create-btn" onClick={handleCreateExecution}>
              + Create execution
            </button>
            <button
              className={`action-btn ${isRunning ? 'stop-btn' : 'run-btn'}`}
              onClick={handleToggleRunStop}
              disabled={isRunning}
            >
              {isRunning ? 'Pipeline Running...' : '▶ Run'}
            </button>
            <button className="action-btn" onClick={handleDeleteClick}>
              🗑️ Delete
            </button>
            <button className="action-btn">🔍 Compare</button>
            <button className="action-btn">🏷️ Tag</button>
          </div>
          {appState.executions.length === 0 ? (
            <p>This project has no executions yet.</p>
          ) : (
            <table className="execution-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Project ID</th>
                  <th>Name</th>
                  <th>File</th>
                  <th>Date</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {appState.executions.map((execution, index) => (
                  <tr key={execution.id}>
                    <td>{execution.id}</td>
                    <td>{execution.projectId}</td>
                    <td>{execution.name}</td>
                    <td>{execution.file}</td>
                    <td>{execution.date}</td>
                    <td>{getExecutionStatus(execution)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Pop-up pour sélectionner le fichier */}
        {showFileSelectionModal && (
          <div className="modal-overlay">
            <div className="modal-content">
              <h3>Select Data File for Execution</h3>
              <div className="file-list">
                {appState.dataFiles.map((file, index) => (
                  <label key={index} className="file-option">
                    <input
                      type="radio"
                      name="file-to-use"
                      checked={selectedFileIndex === index}
                      onChange={() => setSelectedFileIndex(index)}
                    />
                    <span>{file.name}</span>
                  </label>
                ))}
              </div>
              <div className="modal-actions">
                <button className="modal-btn cancel-btn" onClick={closeModal}>
                  Cancel
                </button>
                <button
                  className="modal-btn confirm-btn"
                  onClick={confirmFileSelection}
                  disabled={selectedFileIndex === null}
                >
                  Confirm
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Pop-up pour sélectionner l'exécution à supprimer */}
        {showDeleteModal && (
          <div className="modal-overlay">
            <div className="modal-content">
              <h3>Select Execution to Delete</h3>
              <div className="file-list">
                {appState.executions.map((execution, index) => (
                  <label key={index} className="file-option">
                    <input
                      type="radio"
                      name="execution-to-delete"
                      checked={selectedExecutionIndex === index}
                      onChange={() => setSelectedExecutionIndex(index)}
                    />
                    <span>{execution.name} (File: {execution.file})</span>
                  </label>
                ))}
              </div>
              <div className="modal-actions">
                <button className="modal-btn cancel-btn" onClick={closeDeleteModal}>
                  Cancel
                </button>
                <button
                  className="modal-btn confirm-btn"
                  onClick={confirmDelete}
                  disabled={selectedExecutionIndex === null}
                >
                  Delete
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default Execution;