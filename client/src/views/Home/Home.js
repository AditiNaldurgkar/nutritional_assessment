import React, { useState } from 'react';
import Navbar from './../../components/Navbar/Navbar';
import Footer from './../../components/Footer/Footer';
import Slider from './../../components/Slider/Slider';
import TopSearched from './../../components/TopSearched/TopSearched';

function Home() {
  const [selectedVitamin, setSelectedVitamin] = useState(null);

  return (
    <div>
      <h>THis is the home page </h>
      <Navbar />
      <Slider onVitaminSelect={setSelectedVitamin} />
      {selectedVitamin && (
        <div className="vitamin-info">
          <h2>{selectedVitamin.name}</h2>
          <p>{selectedVitamin.description}</p>
        </div>
      )}
      <TopSearched />
      <Footer />
    </div>
  );
}

export default Home;
