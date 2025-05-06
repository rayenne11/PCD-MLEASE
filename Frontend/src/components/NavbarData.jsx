import React from 'react';
import { useLocation } from 'react-router-dom';

const NavbarData = () => {
  const location = useLocation();
  const pageTitle = location.pathname === '/' ? 'Dashboard' : location.pathname.slice(1).charAt(0).toUpperCase() + location.pathname.slice(2);

  return (
    <div className="navbar">
      <h2>{pageTitle}</h2>
      <div className="navbar-actions">
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
        <button className="action-btn">🗑️ Purge</button>
        <button className="action-btn">🔍 Discover</button>
        <button className="action-btn">⋮</button>
      </div>
    </div>
  );
};

export default NavbarData;