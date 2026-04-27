import React from 'react';

const RecommendationCard = ({ movie }) => {
  return (
    <div className="movie-card animate-fade-in">
      <div className="movie-header">
        <h3 className="movie-title">{movie.title}</h3>
        <span className="movie-year">{movie.year}</span>
      </div>
      
      <div className="movie-tags">
        <span className="mood-tag">{movie.mood_tag}</span>
        {movie.genres?.map((genre, idx) => (
          <span key={idx} className="genre-tag">{genre}</span>
        ))}
      </div>

      <p className="movie-explanation">{movie.explanation}</p>

      <div className="movie-platforms">
        {movie.platforms?.length > 0 && (
          <div className="platform-list">
            Available on: {movie.platforms.join(', ')}
          </div>
        )}
        {movie.imdb_rating && movie.imdb_rating !== 'N/A' && (
          <div className="imdb-rating">⭐ {movie.imdb_rating}</div>
        )}
      </div>
    </div>
  );
};

export default RecommendationCard;
