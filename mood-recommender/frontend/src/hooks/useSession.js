import { useState, useCallback } from 'react';

export const useSession = () => {
  const [sessionId, setSessionId] = useState(null);
  const [shownTitles, setShownTitles] = useState([]);

  const updateSession = useCallback((newSessionId, newTitles) => {
    if (newSessionId) setSessionId(newSessionId);
    if (newTitles && newTitles.length > 0) {
      setShownTitles((prev) => {
        const uniqueTitles = new Set([...prev, ...newTitles]);
        return Array.from(uniqueTitles);
      });
    }
  }, []);

  const resetSession = useCallback(() => {
    setSessionId(null);
    setShownTitles([]);
  }, []);

  return {
    sessionId,
    shownTitles,
    updateSession,
    resetSession,
  };
};
