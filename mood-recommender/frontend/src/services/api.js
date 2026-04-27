import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getRecommendations = async (moodInput, sessionId = null) => {
  const response = await api.post('/recommend', {
    input: moodInput,
    session_id: sessionId,
  });
  return response.data;
};

export const submitFeedback = async (sessionId, feedbackText) => {
  const response = await api.post('/feedback', {
    session_id: sessionId,
    feedback: feedbackText,
  });
  return response.data;
};

export default api;
