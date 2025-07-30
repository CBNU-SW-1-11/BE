// pages/VideoDetailPage.js - 고급 분석 결과 포함
import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { videoAnalysisService } from '../services/videoAnalysisService';
import './VideoDetailPage.css';

const VideoDetailPage = () => {
  const { videoId } = useParams();
  const navigate = useNavigate();
  const [video, setVideo] = useState(null);
  const [scenes, setScenes] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  
  // 고급 분석 관련 상태
  const [analysisResults, setAnalysisResults] = useState(null);
  const [analysisInsights, setAnalysisInsights] = useState('');
  const [activeTab, setActiveTab] = useState('overview'); // overview, frames, scenes, insights, export
  const [advancedFrameData, setAdvancedFrameData] = useState({});
  const [showAdvancedSearch, setShowAdvancedSearch] = useState(false);

  useEffect(() => {
    if (videoId) {
      loadVideoData();
    }
  }, [videoId]);

  const loadVideoData = async () => {
    try {
      setLoading(true);
      setError(null);

      // 비디오 상세 정보 로드
      const videoDetails = await videoAnalysisService.getAnalysisStatus(videoId);
      setVideo(videoDetails);

      // 씬 정보 로드 (고급 분석 포함)
      const scenesData = await videoAnalysisService.getScenes(videoId);
      setScenes(scenesData.scenes || []);

      // 고급 분석 결과가 있으면 추가 데이터 로드
      if (videoDetails.enhanced_analysis) {
        await loadAdvancedAnalysisData();
      }

    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  };

  const loadAdvancedAnalysisData = async () => {
    try {
      // 종합 분석 결과 로드
      const analysisResults = await fetch(`/api/analysis_results/${videoId}/`)
        .then(res => res.json());
      setAnalysisResults(analysisResults);

      // AI 인사이트 로드
      const insightsResponse = await videoAnalysisService.sendVideoChatMessage(
        '이 비디오의 고급 분석 결과를 종합하여 주요 특징, 발견된 내용, 활용 방안을 상세히 설명해주세요.',
        parseInt(videoId)
      );
      setAnalysisInsights(insightsResponse.response || insightsResponse);

    } catch (error) {
      console.error('고급 분석 데이터 로드 실패:', error);
    }
  };

  const searchInVideo = async () => {
    if (!searchQuery.trim()) return;

    try {
      const response = await videoAnalysisService.searchVideoAdvanced(
        parseInt(videoId),
        searchQuery,
        {
          include_clip_analysis: true,
          include_ocr_text: true,
          include_vqa_results: true,
          include_scene_graph: false
        }
      );
      
      if (response.search_results) {
        setSearchResults(response.search_results);
      }
    } catch (error) {
      console.error('고급 검색 실패:', error);
      // 기본 검색으로 fallback
      try {
        const fallbackResponse = await videoAnalysisService.sendVideoChatMessage(
          `찾아줘 ${searchQuery}`,
          parseInt(videoId)
        );
        setSearchResults(fallbackResponse.search_results || []);
      } catch (fallbackError) {
        console.error('기본 검색도 실패:', fallbackError);
      }
    }
  };

  const viewFrame = async (frameNumber) => {
    try {
      // 고급 프레임 데이터 로드
      const frameData = await fetch(`/api/frame/${videoId}/${frameNumber}/enhanced/`)
        .then(res => res.json());
      
      const frameUrl = videoAnalysisService.getFrameImageUrl(videoId, frameNumber);
      setSelectedFrame({
        number: frameNumber,
        url: frameUrl,
        data: frameData
      });
      
      // 고급 프레임 데이터 캐시
      setAdvancedFrameData(prev => ({
        ...prev,
        [frameNumber]: frameData
      }));
    } catch (error) {
      console.error('프레임 데이터 로드 실패:', error);
      // 기본 프레임 표시
      const frameUrl = videoAnalysisService.getFrameImageUrl(videoId, frameNumber);
      setSelectedFrame({
        number: frameNumber,
        url: frameUrl,
        data: null
      });
    }
  };

  const exportAnalysis = async (format = 'json') => {
    try {
      const response = await fetch(`/api/analysis_export/${videoId}/?format=${format}`);
      const blob = await response.blob();
      
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${video?.video_filename || 'video'}_analysis.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('분석 결과 내보내기 실패:', error);
      alert('내보내기에 실패했습니다.');
    }
  };

  const formatDuration = (seconds) => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const renderAdvancedFeatures = (features) => {
    if (!features || Object.keys(features).length === 0) return null;

    const featureIcons = {
      clip: '🖼️',
      ocr: '📝', 
      vqa: '❓',
      scene_graph: '🕸️'
    };

    const featureNames = {
      clip: 'CLIP 씬 분석',
      ocr: 'OCR 텍스트 추출',
      vqa: 'VQA 질문답변', 
      scene_graph: 'Scene Graph'
    };

    return (
      <div className="advanced-features-display">
        <h4>🚀 사용된 고급 AI 기능</h4>
        <div className="features-grid">
          {Object.entries(features).map(([feature, enabled]) => (
            <div key={feature} className={`feature-card ${enabled ? 'enabled' : 'disabled'}`}>
              <div className="feature-icon">{featureIcons[feature] || '🔧'}</div>
              <div className="feature-name">{featureNames[feature] || feature}</div>
              <div className="feature-status">
                {enabled ? '✅ 활성화' : '❌ 비활성화'}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderOverviewTab = () => (
    <div className="overview-tab">
      {/* 기본 정보 */}
      <div className="info-section">
        <h3>📊 분석 개요</h3>
        <div className="info-grid">
          <div className="info-item">
            <span className="info-label">분석 상태:</span>
            <span className={`status-badge status-${video?.status}`}>
              {video?.status === 'completed' ? '분석완료' : 
               video?.status === 'processing' ? '분석중' : video?.status}
            </span>
          </div>
          <div className="info-item">
            <span className="info-label">분석 타입:</span>
            <span className="analysis-type-badge">
              {analysisResults?.video_info?.analysis_type === 'comprehensive' ? '🧠 종합분석' :
               analysisResults?.video_info?.analysis_type === 'enhanced' ? '⚡ 향상된분석' : '🔍 기본분석'}
            </span>
          </div>
          {video?.success_rate && (
            <div className="info-item">
              <span className="info-label">성공률:</span>
              <span className="success-rate">{video.success_rate}%</span>
            </div>
          )}
          {video?.processing_time && (
            <div className="info-item">
              <span className="info-label">처리 시간:</span>
              <span>{video.processing_time}초</span>
            </div>
          )}
        </div>
      </div>

      {/* 고급 기능 표시 */}
      {analysisResults?.advanced_features && (
        renderAdvancedFeatures(analysisResults.advanced_features)
      )}

      {/* 분석 통계 */}
      {analysisResults?.analysis_summary && (
        <div className="statistics-section">
          <h3>📈 분석 통계</h3>
          <div className="stats-grid">
            <div className="stat-card">
              <div className="stat-number">{analysisResults.analysis_summary.total_scenes}</div>
              <div className="stat-label">총 씬 수</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">{analysisResults.analysis_summary.total_frames_analyzed}</div>
              <div className="stat-label">분석된 프레임</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">{analysisResults.analysis_summary.unique_objects}</div>
              <div className="stat-label">감지된 객체 종류</div>
            </div>
            <div className="stat-card">
              <div className="stat-number">{analysisResults.analysis_summary.features_used?.length || 0}</div>
              <div className="stat-label">사용된 AI 기능</div>
            </div>
          </div>
        </div>
      )}

      {/* 콘텐츠 인사이트 */}
      {analysisResults?.content_insights && (
        <div className="insights-section">
          <h3>💡 콘텐츠 분석 결과</h3>
          <div className="insights-grid">
            {analysisResults.content_insights.dominant_objects?.length > 0 && (
              <div className="insight-item">
                <h4>🎯 주요 감지 객체</h4>
                <div className="object-tags">
                  {analysisResults.content_insights.dominant_objects.slice(0, 10).map((obj, idx) => (
                    <span key={idx} className="object-tag">{obj}</span>
                  ))}
                </div>
              </div>
            )}

            {analysisResults.analysis_summary.scene_types?.length > 0 && (
              <div className="insight-item">
                <h4>🎬 감지된 씬 타입</h4>
                <div className="scene-tags">
                  {analysisResults.analysis_summary.scene_types.map((scene, idx) => (
                    <span key={idx} className="scene-tag">{scene}</span>
                  ))}
                </div>
              </div>
            )}

            {analysisResults.content_insights.text_content_length > 0 && (
              <div className="insight-item">
                <h4>📝 추출된 텍스트</h4>
                <p>{analysisResults.content_insights.text_content_length}자의 텍스트가 OCR로 추출되었습니다.</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );

  const renderFramesTab = () => (
    <div className="frames-tab">
      <div className="frames-header">
        <h3>🎞️ 프레임 분석 결과</h3>
        <div className="frames-info">
          {analysisResults?.frame_statistics && (
            <span>
              총 {analysisResults.frame_statistics.total_frames}개 프레임 중 
              {analysisResults.frame_statistics.clip_analyzed_frames}개 CLIP 분석, 
              {analysisResults.frame_statistics.ocr_processed_frames}개 OCR 처리 완료
            </span>
          )}
        </div>
      </div>

      {/* 프레임 커버리지 */}
      {analysisResults?.frame_statistics?.coverage && (
        <div className="coverage-section">
          <h4>📊 분석 커버리지</h4>
          <div className="coverage-bars">
            <div className="coverage-item">
              <span>CLIP 분석</span>
              <div className="coverage-bar">
                <div 
                  className="coverage-fill clip"
                  style={{ width: `${analysisResults.frame_statistics.coverage.clip}%` }}
                ></div>
              </div>
              <span>{analysisResults.frame_statistics.coverage.clip.toFixed(1)}%</span>
            </div>
            <div className="coverage-item">
              <span>OCR 처리</span>
              <div className="coverage-bar">
                <div 
                  className="coverage-fill ocr"
                  style={{ width: `${analysisResults.frame_statistics.coverage.ocr}%` }}
                ></div>
              </div>
              <span>{analysisResults.frame_statistics.coverage.ocr.toFixed(1)}%</span>
            </div>
            <div className="coverage-item">
              <span>VQA 분석</span>
              <div className="coverage-bar">
                <div 
                  className="coverage-fill vqa"
                  style={{ width: `${analysisResults.frame_statistics.coverage.vqa}%` }}
                ></div>
              </div>
              <span>{analysisResults.frame_statistics.coverage.vqa.toFixed(1)}%</span>
            </div>
          </div>
        </div>
      )}

      {/* 검색 결과 프레임들 */}
      {searchResults.length > 0 && (
        <div className="search-results-frames">
          <h4>🔍 검색 결과</h4>
          <div className="frames-grid">
            {searchResults.map((result, index) => (
              <div key={index} className="frame-result-card">
                <div className="frame-header">
                  <span className="frame-number">#{result.frame_id}</span>
                  <span className="frame-time">{formatDuration(result.timestamp)}</span>
                  <span className="match-score">점수: {result.match_score?.toFixed(2)}</span>
                </div>
                
                <div className="frame-preview" onClick={() => viewFrame(result.frame_id)}>
                  <div className="frame-placeholder">🖼️ 프레임 보기</div>
                </div>
                
                <div className="frame-details">
                  {result.caption && (
                    <div className="frame-caption">
                      <strong>캡션:</strong> {result.caption.substring(0, 100)}...
                    </div>
                  )}
                  
                  {result.detected_objects?.length > 0 && (
                    <div className="frame-objects">
                      <strong>객체:</strong> {result.detected_objects.slice(0, 3).join(', ')}
                    </div>
                  )}

                  {/* 고급 분석 결과 */}
                  <div className="advanced-analysis-badges">
                    {result.clip_analysis && (
                      <span className="analysis-badge clip">🖼️ CLIP</span>
                    )}
                    {result.ocr_text?.full_text && (
                      <span className="analysis-badge ocr">📝 OCR</span>
                    )}
                    {result.vqa_insights && (
                      <span className="analysis-badge vqa">❓ VQA</span>
                    )}
                  </div>

                  {result.match_reasons?.length > 0 && (
                    <div className="match-reasons">
                      <strong>매칭 이유:</strong> {result.match_reasons.join(', ')}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );

  const renderScenesTab = () => (
    <div className="scenes-tab">
      <h3>🎬 씬 분석</h3>
      {scenes.length > 0 ? (
        <div className="scenes-grid">
          {scenes.map((scene) => (
            <div key={scene.scene_id} className="scene-card enhanced">
              <div className="scene-header">
                <h4>씬 #{scene.scene_id}</h4>
                <span className="scene-duration">
                  {formatDuration(scene.duration)}
                </span>
              </div>
              <div className="scene-info">
                <div>시간: {formatDuration(scene.start_time)} - {formatDuration(scene.end_time)}</div>
                <div>프레임 수: {scene.frame_count}</div>
                <div>분석 타입: {scene.caption_type}</div>
                
                {/* 고급 분석 정보 */}
                {scene.advanced_features && (
                  <div className="scene-advanced-info">
                    {scene.advanced_features.clip_scene_confidence && (
                      <div>CLIP 신뢰도: {(scene.advanced_features.clip_scene_confidence * 100).toFixed(1)}%</div>
                    )}
                    {scene.advanced_features.ocr_text_density && (
                      <div>텍스트 밀도: {(scene.advanced_features.ocr_text_density * 100).toFixed(1)}%</div>
                    )}
                    {scene.average_complexity && (
                      <div>평균 복잡도: {scene.average_complexity.toFixed(1)}</div>
                    )}
                  </div>
                )}
              </div>
              {scene.dominant_objects && scene.dominant_objects.length > 0 && (
                <div className="dominant-objects">
                  <strong>주요 객체:</strong> {scene.dominant_objects.join(', ')}
                </div>
              )}

              {/* 고급 기능 사용 통계 */}
              {scene.advanced_features && (
                <div className="scene-features-stats">
                  <div className="feature-stat">
                    <span>🖼️ CLIP: {scene.advanced_features.clip_analysis_frames || 0}개</span>
                  </div>
                  <div className="feature-stat">
                    <span>📝 OCR: {scene.advanced_features.ocr_text_frames || 0}개</span>
                  </div>
                  <div className="feature-stat">
                    <span>❓ VQA: {scene.advanced_features.vqa_analysis_frames || 0}개</span>
                  </div>
                </div>
              )}
            </div>
          ))}
        </div>
      ) : (
        <div className="no-scenes">
          <p>씬 정보가 없습니다.</p>
        </div>
      )}
    </div>
  );

  const renderInsightsTab = () => (
    <div className="insights-tab">
      <h3>💡 AI 분석 인사이트</h3>
      
      {analysisInsights ? (
        <div className="ai-insights">
          <div className="insights-content">
            <div className="insights-text">
              {analysisInsights}
            </div>
          </div>
          
          <div className="insights-actions">
            <button 
              onClick={() => navigate(`/integrated-chat`)}
              className="chat-button"
            >
              💬 더 자세한 분석 요청하기
            </button>
          </div>
        </div>
      ) : (
        <div className="loading-insights">
          <div className="loading-spinner"></div>
          <p>AI 인사이트를 생성하는 중...</p>
        </div>
      )}

      {/* 분석 품질 메트릭 */}
      {analysisResults?.video_info && (
        <div className="quality-metrics">
          <h4>📊 분석 품질 지표</h4>
          <div className="metrics-grid">
            <div className="metric-item">
              <span className="metric-label">성공률</span>
              <span className="metric-value">{analysisResults.video_info.success_rate}%</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">처리 시간</span>
              <span className="metric-value">{analysisResults.video_info.processing_time}초</span>
            </div>
            <div className="metric-item">
              <span className="metric-label">분석 깊이</span>
              <span className="metric-value">
                {analysisResults.video_info.analysis_type === 'comprehensive' ? '매우 높음' :
                 analysisResults.video_info.analysis_type === 'enhanced' ? '높음' : '보통'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );

  const renderExportTab = () => (
    <div className="export-tab">
      <h3>📤 분석 결과 내보내기</h3>
      
      <div className="export-options">
        <div className="export-format">
          <h4>내보내기 형식</h4>
          <div className="format-buttons">
            <button 
              onClick={() => exportAnalysis('json')}
              className="export-button json"
            >
              📄 JSON
              <span>전체 분석 데이터</span>
            </button>
            <button 
              onClick={() => exportAnalysis('csv')}
              className="export-button csv"
            >
              📊 CSV
              <span>프레임별 데이터</span>
            </button>
          </div>
        </div>

        <div className="export-content">
          <h4>포함된 데이터</h4>
          <ul>
            <li>✅ 비디오 메타데이터</li>
            <li>✅ 분석 설정 및 통계</li>
            <li>✅ 씬별 분석 결과</li>
            <li>✅ 프레임별 분석 결과</li>
            <li>✅ 감지된 객체 정보</li>
            <li>✅ 고급 분석 결과 (CLIP, OCR, VQA, Scene Graph)</li>
            <li>✅ AI 생성 캡션</li>
          </ul>
        </div>
      </div>
    </div>
  );

  if (loading) {
    return (
      <div className="video-detail-page">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>비디오 정보를 불러오는 중...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="video-detail-page">
        <div className="error-container">
          <h2>❌ 오류 발생</h2>
          <p>{error}</p>
          <button onClick={() => navigate('/video-analysis')} className="back-button">
            목록으로 돌아가기
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="video-detail-page enhanced">
      <div className="detail-container">
        {/* 헤더 */}
        <div className="detail-header">
          <button 
            onClick={() => navigate('/video-analysis')} 
            className="back-button"
          >
            ← 뒤로가기
          </button>
          <div className="header-title">
            <h1>🧠 {video?.video_filename || 'Video Detail'}</h1>
            {video?.enhanced_analysis && (
              <span className="enhanced-badge">고급 분석 완료</span>
            )}
          </div>
          <button 
            onClick={() => navigate('/integrated-chat')} 
            className="chat-button"
          >
            💬 AI 채팅으로 탐색
          </button>
        </div>

        {/* 검색 섹션 */}
        <div className="search-section enhanced">
          <div className="search-header">
            <h3>🔍 고급 비디오 검색</h3>
            <button 
              onClick={() => setShowAdvancedSearch(!showAdvancedSearch)}
              className="advanced-toggle"
            >
              {showAdvancedSearch ? '간단 검색' : '고급 검색'} {showAdvancedSearch ? '🔼' : '🔽'}
            </button>
          </div>
          
          <div className="search-bar">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder={showAdvancedSearch ? 
                "고급 검색: CLIP, OCR, VQA 결과를 활용한 정밀 검색 (예: 빨간 차와 사람이 함께 있는 실외 장면)" :
                "찾고 싶은 객체나 장면을 입력하세요 (예: 빨간 차, 사람, 'STOP' 텍스트)"
              }
              onKeyPress={(e) => e.key === 'Enter' && searchInVideo()}
            />
            <button onClick={searchInVideo} className="search-button">
              {showAdvancedSearch ? '🚀 고급 검색' : '🔍 검색'}
            </button>
          </div>

          {showAdvancedSearch && (
            <div className="search-features-info">
              <div className="feature-info">
                <span>🖼️ CLIP:</span> 씬의 의미적 내용 분석
              </div>
              <div className="feature-info">
                <span>📝 OCR:</span> 이미지 내 텍스트 추출
              </div>
              <div className="feature-info">
                <span>❓ VQA:</span> 이미지에 대한 질문-답변
              </div>
            </div>
          )}
        </div>

        {/* 탭 네비게이션 */}
        <div className="tab-navigation">
          <button 
            className={`tab-button ${activeTab === 'overview' ? 'active' : ''}`}
            onClick={() => setActiveTab('overview')}
          >
            📊 개요
          </button>
          <button 
            className={`tab-button ${activeTab === 'frames' ? 'active' : ''}`}
            onClick={() => setActiveTab('frames')}
          >
            🎞️ 프레임 ({searchResults.length || 0})
          </button>
          <button 
            className={`tab-button ${activeTab === 'scenes' ? 'active' : ''}`}
            onClick={() => setActiveTab('scenes')}
          >
            🎬 씬 ({scenes.length})
          </button>
          <button 
            className={`tab-button ${activeTab === 'insights' ? 'active' : ''}`}
            onClick={() => setActiveTab('insights')}
          >
            💡 AI 인사이트
          </button>
          <button 
            className={`tab-button ${activeTab === 'export' ? 'active' : ''}`}
            onClick={() => setActiveTab('export')}
          >
            📤 내보내기
          </button>
        </div>

        {/* 탭 콘텐츠 */}
        <div className="tab-content">
          {activeTab === 'overview' && renderOverviewTab()}
          {activeTab === 'frames' && renderFramesTab()}
          {activeTab === 'scenes' && renderScenesTab()}
          {activeTab === 'insights' && renderInsightsTab()}
          {activeTab === 'export' && renderExportTab()}
        </div>

        {/* 프레임 뷰어 모달 */}
        {selectedFrame && (
          <div className="modal-overlay" onClick={() => setSelectedFrame(null)}>
            <div className="modal-content frame-modal enhanced" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>🖼️ 프레임 #{selectedFrame.number}</h3>
                <button 
                  onClick={() => setSelectedFrame(null)}
                  className="modal-close"
                >
                  ✕
                </button>
              </div>
              <div className="modal-body">
                <div className="frame-display">
                  <img 
                    src={selectedFrame.url} 
                    alt={`Frame ${selectedFrame.number}`}
                    className="frame-image"
                  />
                </div>
                
                {/* 고급 분석 결과 표시 */}
                {selectedFrame.data && (
                  <div className="frame-analysis-results">
                    <h4>🧠 고급 분석 결과</h4>
                    
                    {selectedFrame.data.advanced_analysis && (
                      <div className="advanced-analysis-details">
                        {/* CLIP 분석 */}
                        {selectedFrame.data.advanced_analysis.clip_analysis && (
                          <div className="analysis-section">
                            <h5>🖼️ CLIP 씬 분석</h5>
                            <div className="clip-results">
                              {selectedFrame.data.advanced_analysis.clip_analysis.top_matches?.map((match, idx) => (
                                <div key={idx} className="clip-match">
                                  <span>{match.description}</span>
                                  <span className="confidence">{(match.confidence * 100).toFixed(1)}%</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* OCR 결과 */}
                        {selectedFrame.data.advanced_analysis.ocr_text?.texts?.length > 0 && (
                          <div className="analysis-section">
                            <h5>📝 OCR 텍스트 추출</h5>
                            <div className="ocr-results">
                              {selectedFrame.data.advanced_analysis.ocr_text.texts.map((text, idx) => (
                                <div key={idx} className="ocr-text">
                                  <span className="text-content">"{text.text}"</span>
                                  <span className="text-confidence">{(text.confidence * 100).toFixed(1)}%</span>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}

                        {/* VQA 결과 */}
                        {selectedFrame.data.advanced_analysis.vqa_results?.qa_pairs?.length > 0 && (
                          <div className="analysis-section">
                            <h5>❓ VQA 질문답변</h5>
                            <div className="vqa-results">
                              {selectedFrame.data.advanced_analysis.vqa_results.qa_pairs.slice(0, 3).map((qa, idx) => (
                                <div key={idx} className="vqa-pair">
                                  <div className="question">Q: {qa.question}</div>
                                  <div className="answer">A: {qa.answer}</div>
                                </div>
                              ))}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* 기본 정보 */}
                    <div className="frame-basic-info">
                      <div className="info-item">
                        <span>타임스탬프:</span>
                        <span>{formatDuration(selectedFrame.data.timestamp || 0)}</span>
                      </div>
                      <div className="info-item">
                        <span>분석 품질:</span>
                        <span className={`quality-${selectedFrame.data.analysis_quality}`}>
                          {selectedFrame.data.analysis_quality === 'enhanced' ? '고급' : '기본'}
                        </span>
                      </div>
                      {selectedFrame.data.quality_score && (
                        <div className="info-item">
                          <span>품질 점수:</span>
                          <span>{(selectedFrame.data.quality_score * 100).toFixed(1)}%</span>
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* 액션 버튼 */}
        <div className="action-buttons">
          <button 
            onClick={() => navigate('/video-analysis')}
            className="secondary-button"
          >
            📊 분석 현황
          </button>
          <button 
            onClick={() => navigate('/integrated-chat')}
            className="primary-button"
          >
            💬 AI 채팅으로 탐색
          </button>
          <button 
            onClick={() => navigate('/video-upload')}
            className="secondary-button"
          >
            📁 새 비디오 업로드
          </button>
        </div>
      </div>
    </div>
  );
};

export default VideoDetailPage;