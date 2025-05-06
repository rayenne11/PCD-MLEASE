import React, { useState, useEffect } from 'react';
import Navbar from '../components/Navbar';
import '../pages/Dashboard.css';

const Dashboard = () => {
  const [artifacts, setArtifacts] = useState([]);
  const [edaArtifacts, setEdaArtifacts] = useState(null);
  const [dataFileName, setDataFileName] = useState(null);
  const [error, setError] = useState(null);

  // Récupérer les artefacts MLflow via la route Flask
  useEffect(() => {
    const fetchArtifacts = async () => {
      try {
        const response = await fetch('http://localhost:5001/get_mlflow_artifacts');
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        if (data.status === 'success') {
          setArtifacts(data.artifacts);
          setEdaArtifacts(data.eda_artifacts);
          setDataFileName(data.data_file_name);
          setError(null);
        } else {
          throw new Error(data.message);
        }
      } catch (err) {
        console.error('Error fetching MLflow artifacts:', err);
        setError('Unable to fetch artifacts. Please ensure the backend server is running on port 5001.');
      }
    };

    fetchArtifacts();
    const interval = setInterval(fetchArtifacts, 5000); // Rafraîchir toutes les 5 secondes
    return () => clearInterval(interval);
  }, []);

  return (
    <>
      <div className="page-content">
        <h2>Dashboard - MLflow Artifacts</h2>
        {dataFileName && (
          <p className="data-file-name">
            Dataset: <strong>{dataFileName}</strong>
          </p>
        )}
        {error ? (
          <p className="error-message">{error}</p>
        ) : (
          <>
            {/* Section pour les artefacts de l'EDA */}
            {edaArtifacts && edaArtifacts.run_id ? (
              <div className="model-section">
                <h3>EDA Artifacts</h3>
                <div className="metrics-section">
                  <h4>Metrics</h4>
                  {edaArtifacts.metrics.length > 0 ? (
                    <ul>
                      {edaArtifacts.metrics.map((metric, idx) => (
                        <li key={idx}>
                          {metric.name}: {metric.value.toFixed(2)}
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p>No metrics available for EDA.</p>
                  )}
                </div>
                <div className="images-section">
                  <h4>Images</h4>
                  {edaArtifacts.images.length > 0 ? (
                    <div className="images-grid">
                      {edaArtifacts.images.map((image, idx) => (
                        <div key={idx} className="image-container">
                          <img src={image.url} alt={image.path} className="artifact-image" onError={(e) => console.error(`Failed to load image: ${image.url}`)} />
                          <p>{image.path}</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p>No images available for EDA.</p>
                  )}
                </div>
                <div className="html-reports-section">
                  <h4>HTML Reports</h4>
                  {edaArtifacts.html_reports.length > 0 ? (
                    <ul>
                      {edaArtifacts.html_reports.map((report, idx) => (
                        <li key={idx}>
                          <a href={report.url} target="_blank" rel="noopener noreferrer">{report.path}</a>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p>No HTML reports available for EDA.</p>
                  )}
                </div>
              </div>
            ) : (
              <p>No EDA artifacts available. Please run the pipeline to generate EDA artifacts.</p>
            )}

            {/* Section pour les artefacts des modèles */}
            {artifacts.length === 0 ? (
              <p>No model artifacts available. Please run the pipeline to generate artifacts.</p>
            ) : (
              <div className="artifacts-container">
                {artifacts.map((modelData, index) => (
                  <div key={index} className="model-section">
                    <h3>{modelData.model} Artifacts</h3>
                    <div className="metrics-section">
                      <h4>Metrics</h4>
                      {modelData.metrics.length > 0 ? (
                        <ul>
                          {modelData.metrics.map((metric, idx) => (
                            <li key={idx}>
                              {metric.name}: {metric.value.toFixed(2)}
                            </li>
                          ))}
                        </ul>
                      ) : (
                        <p>No metrics available for this model.</p>
                      )}
                    </div>
                    <div className="images-section">
                      <h4>Images</h4>
                      {modelData.images.length > 0 ? (
                        <div className="images-grid">
                          {modelData.images.map((image, idx) => (
                            <div key={idx} className="image-container">
                              <img src={image.url} alt={image.path} className="artifact-image" onError={(e) => console.error(`Failed to load image: ${image.url}`)} />
                              <p>{image.path}</p>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <p>No images available for this model.</p>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </>
  );
};

export default Dashboard;