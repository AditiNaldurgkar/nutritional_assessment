import React, { useState } from 'react';
import { FaSearch, FaBars, FaTimes } from 'react-icons/fa';
import './Navbar.css';

function Navbar() {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <h1 className="logo">Health<span>Plus</span></h1>

        <div className={`nav-links ${menuOpen ? 'open' : ''}`}>
          <a href="/">Home</a>
          <a href="/vitamins">Vitamins</a>
          <a href="/symptoms">Symptom Checker</a>
          <a href="/about">About</a>
          <a href="/contact">Contact</a>
        </div>

        <div className="nav-actions">
          <input type="text" placeholder="Search..." className="search-bar" />
          <FaSearch className="search-icon" />
        </div>

        <div className="menu-toggle" onClick={() => setMenuOpen(!menuOpen)}>
          {menuOpen ? <FaTimes /> : <FaBars />}
        </div>
      </div>
    </nav>
  );
}

export default Navbar;
