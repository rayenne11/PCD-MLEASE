import React from 'react';
import { BrowserRouter as Router, Routes, Route, useNavigate, Navigate, useLocation } from 'react-router-dom';
import Swal from 'sweetalert2';
import Sidebar from './components/Sidebar';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import Data from './pages/Data';
import Models from './pages/Models';

import Pipelines from './pages/Pipelines';
import Execution from './pages/Execution';
import Settings from './pages/Settings';
import HomePage from './pages/Homepage';
import AuthPage from './pages/AuthPage';
import { AppProvider, useAppContext } from './context/AppContext';
import { AuthProvider, useAuth } from './context/AuthContext';
import './styles.css';

// Composant pour gérer l'authentification et les restrictions du projet
const ProtectedRoute = ({ children, requireProjectSetup = false }) => {
  const { user, isLoading } = useAuth();
  const { appState } = useAppContext();
  const location = useLocation();

  // Attendre que le chargement soit terminé avant de décider de la redirection
  if (isLoading) {
    return <div>Loading...</div>;
  }

  // Vérifier l'authentification
  if (!user) {
    return <Navigate to="/signin" state={{ from: location }} replace />;
  }

  // Vérifier la configuration du projet si nécessaire
  if (requireProjectSetup) {
    const isProjectSetup = appState.projectName && appState.projectId;
    if (!isProjectSetup && location.pathname !== '/settings') {
      Swal.fire({
        title: 'Setup Required',
        text: 'Please configure the project settings (name and ID) before proceeding.',
        icon: 'warning',
        confirmButtonColor: '#f97316',
        allowOutsideClick: false,
        allowEscapeKey: false,
      }).then(() => {
        window.location.href = '/settings';
      });
      return <Navigate to="/settings" replace />;
    }
  }

  return children;
};

// Composant pour gérer le bouton Quit dans Navbar
const NavbarWithQuit = () => {
  const navigate = useNavigate();
  const { resetAppState } = useAppContext();
  const { logout } = useAuth();

  const handleQuit = () => {
    Swal.fire({
      title: 'Quit Project',
      text: 'Are you sure you want to quit this project? All progress will be lost.',
      icon: 'warning',
      showCancelButton: true,
      confirmButtonColor: '#f97316',
      cancelButtonColor: '#e5e7eb',
      confirmButtonText: 'Yes, quit',
      cancelButtonText: 'Cancel',
    }).then((result) => {
      if (result.isConfirmed) {
        resetAppState();
        logout();
        Swal.fire({
          title: 'Project Quit',
          text: 'You have successfully quit the project.',
          icon: 'success',
          timer: 2000,
          showConfirmButton: false,
        });
        setTimeout(() => {
          navigate('/');
        }, 2000);
      }
    });
  };

  return <Navbar onQuit={handleQuit} />;
};

// Composant pour gérer le contenu principal
const AppContent = () => {
  const { appState } = useAppContext();
  const location = useLocation();
  const isFullWidth = ['/', '/signin', '/signup'].includes(location.pathname);

  return (
    <div className={`app ${isFullWidth ? 'app-full-width' : ''}`}>
      <Routes>
        {/* Routes publiques (pas de sidebar) */}
        <Route path="/" element={<HomePage />} />
        <Route path="/signin" element={<AuthPage isSignIn={true} />} />
        <Route path="/signup" element={<AuthPage isSignIn={false} />} />

        {/* Routes protégées avec sidebar */}
        <Route
          path="/dashboard"
          element={
            <ProtectedRoute requireProjectSetup>
              <Sidebar isProjectSetup={appState.projectName && appState.projectId} />
              <div className="main-content">
                <NavbarWithQuit />
                <Dashboard />
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/data"
          element={
            <ProtectedRoute requireProjectSetup>
              <Sidebar isProjectSetup={appState.projectName && appState.projectId} />
              <div className="main-content">
                <NavbarWithQuit />
                <Data />
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/models"
          element={
            <ProtectedRoute requireProjectSetup>
              <Sidebar isProjectSetup={appState.projectName && appState.projectId} />
              <div className="main-content">
                <NavbarWithQuit />
                <Models />
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/pipelines"
          element={
            <ProtectedRoute requireProjectSetup>
              <Sidebar isProjectSetup={appState.projectName && appState.projectId} />
              <div className="main-content">
                <NavbarWithQuit />
                <Pipelines />
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/execution"
          element={
            <ProtectedRoute requireProjectSetup>
              <Sidebar isProjectSetup={appState.projectName && appState.projectId} />
              <div className="main-content">
                <NavbarWithQuit />
                <Execution />
              </div>
            </ProtectedRoute>
          }
        />
        <Route
          path="/settings"
          element={
            <ProtectedRoute>
              <Sidebar isProjectSetup={appState.projectName && appState.projectId} />
              <div className="main-content">
                <NavbarWithQuit />
                <Settings />
              </div>
            </ProtectedRoute>
          }
        />
        <Route path="*" element={<Navigate to="/" />} />
      </Routes>
    </div>
  );
};

function App() {
  return (
    <AuthProvider>
      <AppProvider>
        <Router>
          <AppContent />
        </Router>
      </AppProvider>
    </AuthProvider>
  );
}

export default App;