
// TrackingTimeline.jsx - 추적 타임라인 컴포넌트
import React, { useState } from 'react';
import './TrackingTimeline.css';

const TrackingTimeline = ({ track, videoDuration, onFrameClick }) => {
  const [hoveredFrame, setHoveredFrame] = useState(null);

  if (!track || !track.key_frames) return null;

  const timelineWidth = 300;
  const timelineHeight = 40;

  const getPositionOnTimeline = (timestamp) => {
    return (timestamp / videoDuration) * timelineWidth;
  };

  const formatTime = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="tracking-timeline">
      <div className="timeline-header">
        <span className="timeline-title">📍 추적 타임라인</span>
        <span className="timeline-duration">
          {formatTime(track.first_appearance)} ~ {formatTime(track.last_appearance)}
        </span>
      </div>
      
      <div className="timeline-container">
        <svg width={timelineWidth} height={timelineHeight} className="timeline-svg">
          {/* 배경 타임라인 */}
          <line
            x1={0}
            y1={timelineHeight / 2}
            x2={timelineWidth}
            y2={timelineHeight / 2}
            stroke="#e5e5e5"
            strokeWidth="2"
          />
          
          {/* 추적 구간 표시 */}
          <line
            x1={getPositionOnTimeline(track.first_appearance)}
            y1={timelineHeight / 2}
            x2={getPositionOnTimeline(track.last_appearance)}
            y2={timelineHeight / 2}
            stroke="#4F46E5"
            strokeWidth="4"
          />
          
          {/* 키 프레임 점들 */}
          {track.key_frames.map((frame, index) => (
            <g key={index}>
              <circle
                cx={getPositionOnTimeline(frame.timestamp)}
                cy={timelineHeight / 2}
                r="6"
                fill={hoveredFrame === index ? "#EC4899" : "#10B981"}
                stroke="#fff"
                strokeWidth="2"
                className="timeline-point"
                onMouseEnter={() => setHoveredFrame(index)}
                onMouseLeave={() => setHoveredFrame(null)}
                onClick={() => onFrameClick(frame)}
                style={{ cursor: 'pointer' }}
              />
              
              {/* 호버 시 정보 표시 */}
              {hoveredFrame === index && (
                <g>
                  <rect
                    x={getPositionOnTimeline(frame.timestamp) - 25}
                    y={5}
                    width="50"
                    height="15"
                    fill="rgba(0,0,0,0.8)"
                    rx="3"
                  />
                  <text
                    x={getPositionOnTimeline(frame.timestamp)}
                    y={15}
                    textAnchor="middle"
                    fill="white"
                    fontSize="10"
                  >
                    {formatTime(frame.timestamp)}
                  </text>
                </g>
              )}
            </g>
          ))}
        </svg>
        
        {/* 시간 눈금 */}
        <div className="timeline-labels">
          <span className="timeline-label start">0:00</span>
          <span className="timeline-label end">{formatTime(videoDuration)}</span>
        </div>
      </div>
      
      <div className="timeline-info">
        <span className="frame-count">📊 {track.key_frames.length}개 키프레임</span>
        <span className="confidence">🎯 평균 신뢰도: {(track.avg_match_score * 100).toFixed(1)}%</span>
      </div>
    </div>
  );
};

export default TrackingTimeline;