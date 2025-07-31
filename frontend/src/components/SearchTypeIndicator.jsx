
// SearchTypeIndicator.jsx - 검색 타입 표시 컴포넌트
import React from 'react';
import './SearchTypeIndicator.css';

const SearchTypeIndicator = ({ type, size = 'medium' }) => {
  const typeConfig = {
    'cross_video_search': {
      icon: '🎬',
      label: '영상간 검색',
      color: '#10B981',
      description: '조건에 맞는 영상들을 찾았습니다'
    },
    'in_video_tracking': {
      icon: '🎯',
      label: '객체 추적',
      color: '#F59E0B',
      description: '특정 객체의 움직임을 추적했습니다'
    },
    'temporal_analysis': {
      icon: '📊',
      label: '시간별 분석',
      color: '#8B5CF6',
      description: '시간대별 통계를 분석했습니다'
    },
    'advanced_search': {
      icon: '🔍',
      label: '고급 검색',
      color: '#3B82F6',
      description: '고급 AI 기능으로 검색했습니다'
    },
    'intelligent_search': {
      icon: '🧠',
      label: '지능형 검색',
      color: '#EC4899',
      description: 'AI가 자동으로 최적의 방법을 선택했습니다'
    },
    'welcome': {
      icon: '👋',
      label: '환영 메시지',
      color: '#6B7280',
      description: '시스템 안내'
    }
  };

  const config = typeConfig[type] || typeConfig['intelligent_search'];

  return (
    <div className={`search-type-indicator ${size}`} style={{ borderColor: config.color }}>
      <div className="indicator-content">
        <span className="indicator-icon">{config.icon}</span>
        <span className="indicator-label" style={{ color: config.color }}>
          {config.label}
        </span>
      </div>
      <div className="indicator-description">
        {config.description}
      </div>
    </div>
  );
};

export default SearchTypeIndicator;

// SearchTypeIndicator.css
const searchTypeIndicatorCSS = `
.search-type-indicator {
  display: inline-block;
  border-left: 3px solid;
  background: #f8fafc;
  padding: 8px 12px;
  margin: 4px 0;
  border-radius: 0 6px 6px 0;
  max-width: 100%;
}

.search-type-indicator.small {
  padding: 4px 8px;
  font-size: 12px;
}

.search-type-indicator.large {
  padding: 12px 16px;
  font-size: 16px;
}

.indicator-content {
  display: flex;
  align-items: center;
  gap: 8px;
  margin-bottom: 2px;
}

.indicator-icon {
  font-size: 16px;
}

.indicator-label {
  font-weight: 600;
  font-size: 13px;
}

.indicator-description {
  font-size: 11px;
  color: #6b7280;
  margin-left: 24px;
}

.search-type-indicator.small .indicator-icon {
  font-size: 14px;
}

.search-type-indicator.small .indicator-label {
  font-size: 12px;
}

.search-type-indicator.small .indicator-description {
  font-size: 10px;
  margin-left: 20px;
}

.search-type-indicator.large .indicator-icon {
  font-size: 20px;
}

.search-type-indicator.large .indicator-label {
  font-size: 15px;
}

.search-type-indicator.large .indicator-description {
  font-size: 13px;
  margin-left: 28px;
}

@media (max-width: 768px) {
  .search-type-indicator {
    padding: 6px 10px;
  }
  
  .indicator-content {
    gap: 6px;
  }
  
  .indicator-description {
    margin-left: 20px;
  }
}
`;