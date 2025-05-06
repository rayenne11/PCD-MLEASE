import React from 'react';
import { useLocation } from 'react-router-dom';
import './Navbar.css';

const Navbar = ({ onQuit }) => {
  const location = useLocation();
  const pageTitle = location.pathname === '/' ? 'Dashboard' : location.pathname.slice(1).charAt(0).toUpperCase() + location.pathname.slice(2);

  return (
    <div className="navbar">
      <h2>{pageTitle}</h2>

      <button className="quit-btn" onClick={onQuit}>
        Quit
      </button>
      
    </div>
  );
};

export default Navbar;