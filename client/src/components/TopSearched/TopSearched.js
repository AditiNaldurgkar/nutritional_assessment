import React from 'react';

const topVitamins = [
  { name: "Omega-3", description: "Good for heart and brain health." },
  { name: "Iron", description: "Helps in oxygen transport in blood." },
  { name: "Magnesium", description: "Supports muscle and nerve function." },
];

function TopSearched() {
  return (
    <div className="top-searched">
      <h2>Top Searched Vitamins & Supplements</h2>
      <ul>
        {topVitamins.map((vitamin, index) => (
          <li key={index}>
            <a href={`/vitamin/${vitamin.name.toLowerCase()}`}>{vitamin.name}</a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default TopSearched;
