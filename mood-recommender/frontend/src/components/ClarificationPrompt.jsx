import React from 'react';

const ClarificationPrompt = ({ message }) => {
  if (!message) return null;

  return (
    <div className="clarification-bubble animate-fade-in">
      <div className="agent-avatar">🤖</div>
      <div className="agent-message">
        <p>{message}</p>
      </div>
    </div>
  );
};

export default ClarificationPrompt;
