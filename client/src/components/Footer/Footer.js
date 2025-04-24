import React from 'react';

function Footer() {
  return (
    <footer className="footer">
      <h2>Sign up for our free Good Health Newsletter</h2>
      <p>Get wellness tips to help you live happier and healthier</p>
      <input type="email" placeholder="Enter your email address" />
      <button>Subscribe</button>
      <p>By clicking Subscribe, I agree to the WebMD Terms & Conditions & Privacy Policy.</p>
      
      <h3>Follow WebMD on Social Media</h3>
      <p>Download WebMD App</p>
      <button>App Store</button>
      <button>Android Store</button>

      <h3>Policies</h3>
      <ul>
        <li><a href="/privacy-policy">Privacy Policy</a></li>
        <li><a href="/terms">Terms of Use</a></li>
        <li><a href="/about">About</a></li>
      </ul>

      <p>&copy; 2005 - 2025 WebMD LLC. All rights reserved.</p>
    </footer>
  );
}

export default Footer;
