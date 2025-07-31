// StatisticsChart.jsx - 통계 차트 컴포넌트
import React from 'react';
import { PieChart, Pie, BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import './StatisticsChart.css';

const StatisticsChart = ({ type, title, data, width = 400, height = 300 }) => {
  if (!data) return null;

  const renderPieChart = () => (
    <ResponsiveContainer width="100%" height={height}>
      <PieChart>
        <Pie
          data={data.labels.map((label, index) => ({
            name: label,
            value: data.data[index]
          }))}
          cx="50%"
          cy="50%"
          labelLine={false}
          label={({ name, percent }) => `${name}: ${(percent * 100).toFixed(0)}%`}
          outerRadius={80}
          fill="#8884d8"
          dataKey="value"
        >
          {data.labels.map((_, index) => (
            <Cell key={`cell-${index}`} fill={data.colors[index]} />
          ))}
        </Pie>
        <Tooltip />
      </PieChart>
    </ResponsiveContainer>
  );

  const renderBarChart = () => (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart
        data={data.labels.map((label, index) => ({
          name: label,
          value: data.data[index]
        }))}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Bar dataKey="value" fill={data.colors[0] || '#8884d8'} />
      </BarChart>
    </ResponsiveContainer>
  );

  const renderLineChart = () => (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart
        data={data.labels.map((label, index) => ({
          name: label,
          value: data.data[index]
        }))}
        margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="name" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="value" stroke={data.colors[0] || '#8884d8'} />
      </LineChart>
    </ResponsiveContainer>
  );

  return (
    <div className="statistics-chart">
      <h5 className="chart-title">{title}</h5>
      <div className="chart-container">
        {type === 'pie' && renderPieChart()}
        {type === 'bar' && renderBarChart()}
        {type === 'line' && renderLineChart()}
      </div>
    </div>
  );
};

export default StatisticsChart;


// CSS 파일들
// StatisticsChart.css
const statisticsChartCSS = `
.statistics-chart {
  background: white;
  border-radius: 8px;
  padding: 16px;
  box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
  margin: 8px;
  min-width: 300px;
}

.chart-title {
  margin: 0 0 16px 0;
  font-size: 16px;
  font-weight: 600;
  color: #374151;
  text-align: center;
}

.chart-container {
  width: 100%;
  height: 300px;
}

@media (max-width: 768px) {
  .statistics-chart {
    min-width: 250px;
    margin: 4px;
    padding: 12px;
  }
  
  .chart-container {
    height: 250px;
  }
}
`;

// TrackingTimeline.css
const trackingTimelineCSS = `
.tracking-timeline {
  background: #f8fafc;
  border: 1px solid #e2e8f0;
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
}

.timeline-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}

.timeline-title {
  font-weight: 600;
  color: #374151;
  font-size: 14px;
}

.timeline-duration {
  font-size: 12px;
  color: #6b7280;
  background: #e5e7eb;
  padding: 2px 8px;
  border-radius: 12px;
}

.timeline-container {
  position: relative;
  margin: 12px 0;
}

.timeline-svg {
  width: 100%;
  height: 40px;
}

.timeline-point {
  transition: all 0.2s ease;
}

.timeline-point:hover {
  r: 8;
  filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.2));
}

.timeline-labels {
  display: flex;
  justify-content: space-between;
  margin-top: 4px;
}

.timeline-label {
  font-size: 10px;
  color: #6b7280;
}

.timeline-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-top: 8px;
  font-size: 12px;
  color: #6b7280;
}

.frame-count, .confidence {
  background: #f3f4f6;
  padding: 2px 6px;
  border-radius: 4px;
}

@media (max-width: 768px) {
  .timeline-container {
    margin: 8px 0;
  }
  
  .timeline-info {
    flex-direction: column;
    gap: 4px;
    align-items: flex-start;
  }
}
`;

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

// EnhancedIntegratedChatPage.css (주요 스타일만)
const enhancedChatPageCSS = `
.enhanced-integrated-chat-page {
  display: flex;
  height: 100vh;
  background: #f8fafc;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

.chat-container {
  display: flex;
  width: 100%;
  max-width: 1400px;
  margin: 0 auto;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border-radius: 12px;
  overflow: hidden;
}

.chat-sidebar {
  width: 320px;
  background: white;
  border-right: 1px solid #e5e7eb;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}

.sidebar-header {
  padding: 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.system-status {
  margin-top: 8px;
  font-size: 12px;
}

.status-good {
  color: #10b981;
}

.status-error {
  color: #ef4444;
}

.search-mode-selector {
  padding: 16px;
  border-bottom: 1px solid #f3f4f6;
}

.mode-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 8px;
}

.mode-button {
  padding: 8px;
  border: 2px solid #e5e7eb;
  border-radius: 6px;
  background: white;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
  text-align: center;
}

.mode-button:hover {
  border-color: #d1d5db;
  background: #f9fafb;
}

.mode-button.active {
  border-color: #4f46e5;
  background: #eef2ff;
  color: #4f46e5;
  font-weight: 600;
}

.quick-actions {
  padding: 16px;
  border-bottom: 1px solid #f3f4f6;
}

.action-categories {
  margin-top: 12px;
}

.action-category {
  margin-bottom: 16px;
}

.action-category h5 {
  margin: 0 0 8px 0;
  font-size: 13px;
  color: #374151;
  font-weight: 600;
}

.quick-action-button {
  display: block;
  width: 100%;
  padding: 8px;
  margin-bottom: 4px;
  border: 1px solid #e5e7eb;
  border-radius: 4px;
  background: white;
  text-align: left;
  font-size: 11px;
  cursor: pointer;
  transition: all 0.2s ease;
  line-height: 1.3;
}

.quick-action-button:hover {
  background: #f3f4f6;
  border-color: #d1d5db;
}

.quick-action-button.small {
  padding: 6px;
  font-size: 10px;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: white;
}

.chat-header {
  padding: 16px;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.current-video-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.video-name {
  font-weight: 600;
  color: #374151;
}

.advanced-badge {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.chat-actions {
  display: flex;
  gap: 8px;
}

.chat-actions button {
  padding: 6px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: white;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.chat-actions button:hover {
  background: #f3f4f6;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  max-height: 400px;
}

.message {
  margin-bottom: 16px;
  display: flex;
  max-width: 85%;
}

.message.user {
  margin-left: auto;
}

.message.user .message-content {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  margin-left: auto;
}

.message.bot .message-content {
  background: #f3f4f6;
  color: #374151;
}

.message.system .message-content {
  background: #fef3c7;
  color: #92400e;
  border-left: 3px solid #f59e0b;
}

.message.error .message-content {
  background: #fef2f2;
  color: #dc2626;
  border-left: 3px solid #ef4444;
}

.message-content {
  padding: 12px 16px;
  border-radius: 12px;
  max-width: 100%;
  line-height: 1.5;
}

.message-text {
  white-space: pre-wrap;
  word-wrap: break-word;
}

.message-timestamp {
  font-size: 10px;
  opacity: 0.7;
  margin-top: 4px;
}

.message-metadata {
  display: flex;
  gap: 8px;
  margin-top: 6px;
  font-size: 10px;
}

.result-count, .duration, .cached {
  background: rgba(255, 255, 255, 0.2);
  padding: 2px 6px;
  border-radius: 10px;
}

.results-container {
  background: #f8fafc;
  border-top: 1px solid #e5e7eb;
  max-height: 50vh;
  overflow-y: auto;
}

.result-tabs {
  display: flex;
  background: white;
  border-bottom: 1px solid #e5e7eb;
  padding: 0 16px;
}

.result-tab {
  padding: 12px 16px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 13px;
  color: #6b7280;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
}

.result-tab:hover {
  color: #374151;
}

.result-tab.active {
  color: #4f46e5;
  border-bottom-color: #4f46e5;
  font-weight: 600;
}

.result-tab:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.results-content {
  padding: 16px;
}

.input-container {
  padding: 16px;
  background: white;
  border-top: 1px solid #e5e7eb;
}

.input-wrapper {
  display: flex;
  gap: 8px;
  align-items: flex-end;
}

.input-wrapper textarea {
  flex: 1;
  padding: 12px;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 14px;
  resize: vertical;
  min-height: 44px;
  max-height: 120px;
  font-family: inherit;
  transition: border-color 0.2s ease;
}

.input-wrapper textarea:focus {
  outline: none;
  border-color: #4f46e5;
}

.send-button {
  padding: 12px 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s ease;
  min-width: 60px;
}

.send-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.input-hints {
  margin-top: 8px;
  font-size: 12px;
  color: #6b7280;
  text-align: center;
}

.loading-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px;
}

.loading-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid #e5e7eb;
  border-top-color: #4f46e5;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.error-banner {
  background: #fef2f2;
  color: #dc2626;
  padding: 12px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #fecaca;
}

.error-banner button {
  background: none;
  border: none;
  color: #dc2626;
  cursor: pointer;
  font-size: 18px;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 반응형 디자인 */
@media (max-width: 768px) {
  .chat-container {
    flex-direction: column;
    height: 100vh;
  }
  
  .chat-sidebar {
    width: 100%;
    height: 200px;
    border-right: none;
    border-bottom: 1px solid #e5e7eb;
  }
  
  .mode-buttons {
    grid-template-columns: 1fr 1fr 1fr 1fr;
  }
  
  .mode-button {
    font-size: 10px;
    padding: 6px;
  }
  
  .messages-container {
    max-height: 300px;
  }
  
  .results-container {
    max-height: 40vh;
  }
}
`;

// 내보내기
export { StatisticsChart, TrackingTimeline, SearchTypeIndicator };
export const cssStyles = {
  statisticsChart: statisticsChartCSS,
  trackingTimeline: trackingTimelineCSS,
  searchTypeIndicator: searchTypeIndicatorCSS,
  enhancedChatPage: enhancedChatPageCSS
};