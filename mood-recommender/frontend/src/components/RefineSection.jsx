import React, { useState } from 'react';

const RefineSection = ({ onRefine, isLoading }) => {
  const [feedback, setFeedback] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (feedback.trim() && !isLoading) {
      onRefine(feedback);
      setFeedback('');
    }
  };

  return (
    <div className="refine-section animate-fade-in">
      <p className="refine-label">Not quite right? Adjust your mood:</p>
      <form onSubmit={handleSubmit} className="refine-form">
        <input
          type="text"
          className="refine-input"
          placeholder="e.g., 'Too dark, want something lighter' or 'More sci-fi please'"
          value={feedback}
          onChange={(e) => setFeedback(e.target.value)}
          disabled={isLoading}
        />
        <button 
          type="submit" 
          className="refine-btn"
          disabled={isLoading || !feedback.trim()}
        >
          {isLoading ? '...' : 'Adjust'}
        </button>
      </form>
    </div>
  );
};

export default RefineSection;
