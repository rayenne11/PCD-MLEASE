import React, { createContext, useState, useContext } from 'react';

// Créer le contexte
const AppContext = createContext();

// Fonction pour générer un ID unique basé sur le nom du projet
const generateProjectId = (projectName) => {
  const timestamp = Date.now();
  const random = Math.random().toString(36).substring(2, 8); // Générer une chaîne aléatoire
  return `${projectName.toLowerCase().replace(/\s/g, '-')}-${timestamp}-${random}`;
};

// Fournisseur de contexte
export const AppProvider = ({ children }) => {
  // État initial pour toutes les pages
  const initialState = {
    dataFiles: [], // Fichiers importés dans Data.jsx
    selectedModels: [], // Modèles sélectionnés dans Models.jsx
    executions: [], // Données pour Execution.jsx
    projectName: '', // Nom du projet
    projectId: '', // ID du projet
  };

  const [appState, setAppState] = useState(initialState);

  // Fonction pour réinitialiser l'état
  const resetAppState = () => {
    setAppState(initialState);
  };

  // Fonction pour générer et enregistrer l'ID du projet
  const setProjectId = (projectName) => {
    const newProjectId = generateProjectId(projectName);
    setAppState((prevState) => ({
      ...prevState,
      projectId: newProjectId,
    }));
    return newProjectId;
  };

  return (
    <AppContext.Provider value={{ appState, setAppState, resetAppState, setProjectId }}>
      {children}
    </AppContext.Provider>
  );
};

// Hook personnalisé pour utiliser le contexte
export const useAppContext = () => useContext(AppContext);