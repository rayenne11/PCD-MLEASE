import React from 'react';
import { NavLink } from 'react-router-dom';
import { FaHome, FaChartBar, FaUser, FaLink, FaRocket, FaBolt, FaCog } from 'react-icons/fa';
import Swal from 'sweetalert2';
import './Sidebar.css'; // Chemin corrigé

const Sidebar = ({ isProjectSetup }) => {
  const handleRestrictedClick = (e) => {
    if (!isProjectSetup) {
      e.preventDefault(); // Empêche la navigation
      Swal.fire({
        title: 'Setup Required',
        text: 'Please configure the project settings (name and ID) before proceeding.',
        icon: 'warning',
        confirmButtonColor: '#f97316',
      });
    }
  };

  return (
    <div className="sidebar">
      <h1 className="logo">MLEASE.</h1>
      <nav>
        <ul>
          <li>
            <NavLink
              to="/dashboard"
              className={({ isActive }) => (isActive ? 'active' : '')}
              onClick={handleRestrictedClick}
            >
              <span className="icon">
                <FaHome />
              </span>
              Dashboard
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/data"
              className={({ isActive }) => (isActive ? 'active' : '')}
              onClick={handleRestrictedClick}
            >
              <span className="icon">
                <FaChartBar />
              </span>
              Data
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/models"
              className={({ isActive }) => (isActive ? 'active' : '')}
              onClick={handleRestrictedClick}
            >
              <span className="icon">
                <FaUser />
              </span>
              Models
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/pipelines"
              className={({ isActive }) => (isActive ? 'active' : '')}
              onClick={handleRestrictedClick}
            >
              <span className="icon">
                <FaLink />
              </span>
              Pipelines
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/deployment"
              className={({ isActive }) => (isActive ? 'active' : '')}
              onClick={handleRestrictedClick}
            >
              <span className="icon">
                <FaRocket />
              </span>
              Deployment
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/execution"
              className={({ isActive }) => (isActive ? 'active' : '')}
              onClick={handleRestrictedClick}
            >
              <span className="icon">
                <FaBolt />
              </span>
              Execution
            </NavLink>
          </li>
          <li>
            <NavLink
              to="/settings"
              className={({ isActive }) => (isActive ? 'active' : '')}
            >
              <span className="icon">
                <FaCog />
              </span>
              Settings
            </NavLink>
          </li>
        </ul>
      </nav>
    </div>
  );
};

export default Sidebar;