import React, { useState } from 'react';

const MoodInput = ({ onSubmit, isLoading }) => {
  const [value, setValue] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (value.trim() && !isLoading) {
      onSubmit(value);
    }
  };

  return (
    <div className="mood-input-section animate-fade-in">
      <form onSubmit={handleSubmit}>
        <textarea
          className="mood-textarea"
          placeholder="How are you feeling? Tell me about your mood, your day, or what you're looking for..."
          value={value}
          onChange={(e) => setValue(e.target.value)}
          disabled={isLoading}
        />
        <button 
          type="submit" 
          className={`submit-btn ${isLoading ? 'loading' : ''}`}
          disabled={isLoading || !value.trim()}
        >
          {isLoading ? 'Interpreting Mood...' : 'Get Recommendations'}
        </button>
      </form>
    </div>
  );
};

export default MoodInput;
