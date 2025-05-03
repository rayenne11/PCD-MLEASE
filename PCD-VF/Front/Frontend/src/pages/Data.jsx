import React, { useState } from 'react';
import Swal from 'sweetalert2';
import { useAppContext } from '../context/AppContext';
import './Data.css';

const Data = () => {
  const { appState, setAppState } = useAppContext();
  const [showPurgeModal, setShowPurgeModal] = useState(false);
  const [selectedFileIndex, setSelectedFileIndex] = useState(null);
  const [showReportModal, setShowReportModal] = useState(false);

  // Fonction pour gérer l'importation des fichiers
  const handleFileUpload = async (event) => {
    const uploadedFiles = event.target.files;
    if (uploadedFiles.length === 0) {
      Swal.fire({
        title: 'Error',
        text: 'Please select at least one file to upload.',
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
      return;
    }

    const existingFileNames = appState.dataFiles.map((file) => file.name); // Liste des noms existants
    const newFiles = [];
    const duplicateFiles = [];

    // Vérifier chaque fichier pour détecter les doublons
    Array.from(uploadedFiles).forEach((file) => {
      if (existingFileNames.includes(file.name)) {
        duplicateFiles.push(file.name); // Ajouter à la liste des doublons
      } else {
        newFiles.push({
          name: file.name,
          size: (file.size / 1024).toFixed(0) + ' Ko',
          date: new Date().toLocaleString('fr-FR', {
            day: '2-digit',
            month: '2-digit',
            year: 'numeric',
            hour: '2-digit',
            minute: '2-digit',
          }),
          file: file, // Stocker l'objet fichier pour l'upload
        });
      }
    });

    // Si des doublons sont détectés, afficher une erreur
    if (duplicateFiles.length > 0) {
      Swal.fire({
        title: 'Error',
        text: `The following files already exist: ${duplicateFiles.join(', ')}.`,
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
    }

    // Uploader chaque nouveau fichier vers le backend
    for (const newFile of newFiles) {
      const formData = new FormData();
      formData.append('file', newFile.file);

      try {
        const response = await fetch('http://localhost:5001/upload_data', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();
        if (data.status !== 'success') {
          Swal.fire({
            title: 'Error',
            text: data.message || 'Failed to upload file.',
            icon: 'error',
            confirmButtonColor: '#f97316',
          });
          return;
        }
      } catch (error) {
        Swal.fire({
          title: 'Error',
          text: 'Failed to upload file. Please ensure the backend is running.',
          icon: 'error',
          confirmButtonColor: '#f97316',
        });
        return;
      }
    }

    // Ajouter les nouveaux fichiers (non doublons) à l'état global
    if (newFiles.length > 0) {
      setAppState((prevState) => ({
        ...prevState,
        dataFiles: [...prevState.dataFiles, ...newFiles],
      }));
      Swal.fire({
        title: 'Success',
        text: 'File(s) uploaded successfully!',
        icon: 'success',
        timer: 1500,
        showConfirmButton: false,
      });
    }
  };

  // Fonction pour afficher la pop-up de confirmation avec SweetAlert2
  const showConfirmation = (onConfirm) => {
    Swal.fire({
      title: 'Are you sure?',
      text: 'Do you really want to purge this file? This action cannot be undone.',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonColor: '#f97316',
      cancelButtonColor: '#e5e7eb',
      confirmButtonText: 'Yes, purge it!',
      cancelButtonText: 'Cancel',
    }).then((result) => {
      if (result.isConfirmed) {
        onConfirm();
        Swal.fire({
          title: 'Purged!',
          text: 'The file has been successfully purged.',
          icon: 'success',
          timer: 1500,
          showConfirmButton: false,
        });
      }
    });
  };

  // Fonction pour gérer le clic sur le bouton Purge
  const handlePurgeClick = () => {
    if (appState.dataFiles.length === 0) {
      Swal.fire({
        title: 'Error',
        text: 'There are no files to purge.',
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
      return;
    }
    if (appState.dataFiles.length === 1) {
      showConfirmation(() => {
        setAppState((prevState) => ({ ...prevState, dataFiles: [] }));
      });
    } else {
      setShowPurgeModal(true);
    }
  };

  // Fonction pour confirmer la suppression du fichier sélectionné
  const confirmPurge = () => {
    if (selectedFileIndex !== null) {
      showConfirmation(() => {
        setAppState((prevState) => ({
          ...prevState,
          dataFiles: prevState.dataFiles.filter((_, index) => index !== selectedFileIndex),
        }));
        setShowPurgeModal(false);
        setSelectedFileIndex(null);
      });
    }
  };

  // Fonction pour fermer la pop-up sans supprimer
  const closeModal = () => {
    setShowPurgeModal(false);
    setSelectedFileIndex(null);
  };

  // Fonction pour gérer le clic sur le bouton Discover
  const handleDiscoverClick = () => {
    if (appState.dataFiles.length === 0) {
      Swal.fire({
        title: 'Error',
        text: 'No data files uploaded. Please upload a file first.',
        icon: 'error',
        confirmButtonColor: '#f97316',
      });
      return;
    }

    // Vérifier si index.html existe en faisant une requête au backend
    fetch('http://localhost:5001/get_eda_report')
      .then((response) => {
        if (!response.ok) {
          throw new Error('EDA report not found.');
        }
        setShowReportModal(true);
      })
      .catch((error) => {
        Swal.fire({
          title: 'Error',
          text: 'EDA report not available. Please run the pipeline first to generate the report.',
          icon: 'error',
          confirmButtonColor: '#f97316',
        });
      });
  };

  // Fonction pour fermer la pop-up du rapport
  const closeReportModal = () => {
    setShowReportModal(false);
  };

  return (
    <>
      <div className="main-content">
        <div className="page-content">
          {appState.dataFiles.length === 0 ? (
            <>
              <p>This project has no files yet.</p>
              <label htmlFor="file-upload" className="import-btn">
                Import Data
              </label>
              <input
                id="file-upload"
                type="file"
                accept=".csv"
                multiple
                onChange={handleFileUpload}
                style={{ display: 'none' }}
              />
            </>
          ) : (
            <>
              <div className="data-actions">
                <label htmlFor="file-upload" className="import-btn">
                  Import Data
                </label>
                <input
                  id="file-upload"
                  type="file"
                  accept=".csv"
                  multiple
                  onChange={handleFileUpload}
                  style={{ display: 'none' }}
                />
                <button className="action-btn" onClick={handlePurgeClick}>
                  🗑️ Purge
                </button>
                <button className="action-btn discover-btn" onClick={handleDiscoverClick}>
                  🔍 Discover
                </button>
              </div>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Taille</th>
                    <th>Date</th>
                  </tr>
                </thead>
                <tbody>
                  {appState.dataFiles.map((file, index) => (
                    <tr key={index}>
                      <td>{file.name}</td>
                      <td>{file.size}</td>
                      <td>{file.date}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </>
          )}
        </div>

        {/* Pop-up pour la purge */}
        {showPurgeModal && (
          <div className="modal-overlay">
            <div className="modal-content">
              <h3>Select File to Purge</h3>
              <div className="file-list">
                {appState.dataFiles.map((file, index) => (
                  <label key={index} className="file-option">
                    <input
                      type="radio"
                      name="file-to-purge"
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
                  onClick={confirmPurge}
                  disabled={selectedFileIndex === null}
                >
                  Purge
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Pop-up pour afficher le rapport EDA (index.html) */}
        {showReportModal && (
          <div className="modal-overlay">
            <div className="modal-content modal-report-content">
              <h3>EDA Report</h3>
              <div className="report-container">
                <iframe
                  src="http://localhost:5001/get_eda_report"
                  title="EDA Report"
                  style={{ width: '100%', height: '650px', border: 'none' }}
                />
              </div>
              <div className="modal-actions">
                <button className="modal-btn cancel-btn" onClick={closeReportModal}>
                  Close
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
};

export default Data;