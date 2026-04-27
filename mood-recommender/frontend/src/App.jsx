import React, { useState } from 'react';
import MoodInput from './components/MoodInput';
import RecommendationCard from './components/RecommendationCard';
import ClarificationPrompt from './components/ClarificationPrompt';
import RefineSection from './components/RefineSection';
import { useSession } from './hooks/useSession';
import { getRecommendations, submitFeedback } from './services/api';

function App() {
  const { sessionId, shownTitles, updateSession, resetSession } = useSession();
  const [recommendations, setRecommendations] = useState([]);
  const [clarification, setClarification] = useState('');
  const [interpretedMood, setInterpretedMood] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleMoodSubmit = async (moodInput) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await getRecommendations(moodInput, sessionId);
      
      setRecommendations(data.data || []);
      setClarification(data.follow_up || '');
      setInterpretedMood(data.interpreted_mood);
      
      const newTitles = (data.data || []).map(r => r.title);
      updateSession(data.session_id, newTitles);
    } catch (err) {
      console.error(err);
      setError('Failed to get recommendations. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRefine = async (feedbackText) => {
    setIsLoading(true);
    setError(null);
    try {
      const data = await submitFeedback(sessionId, feedbackText);
      
      setRecommendations(data.data || []);
      setClarification(data.follow_up || '');
      setInterpretedMood(data.interpreted_mood);
      
      const newTitles = (data.data || []).map(r => r.title);
      updateSession(data.session_id, newTitles);
    } catch (err) {
      console.error(err);
      setError('Failed to update recommendations. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="container">
      <header className="hero animate-fade-in">
        <h1 className="hero-title">CineMatch</h1>
        <p className="hero-subtitle">Your emotionally intelligent movie concierge.</p>
      </header>

      <MoodInput onSubmit={handleMoodSubmit} isLoading={isLoading} />

      {error && <div className="error-message">{error}</div>}

      {interpretedMood && !isLoading && (
        <div className="mood-badge animate-fade-in">
          Interpreted Mood: <span>{interpretedMood.interpreted_mood} ({interpretedMood.intensity})</span>
        </div>
      )}

      <ClarificationPrompt message={clarification} />

      <div className="results-grid">
        {recommendations.map((movie, index) => (
          <RecommendationCard key={index} movie={movie} />
        ))}
      </div>

      {recommendations.length > 0 && (
        <RefineSection onRefine={handleRefine} isLoading={isLoading} />
      )}

      {recommendations.length > 0 && (
        <button className="reset-btn" onClick={resetSession}>Start Over</button>
      )}
    </div>
  );
}

export default App;
