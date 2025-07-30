// pages/VideoUploadPage.js - 고급 분석 옵션 추가
import React, { useState, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { videoAnalysisService } from '../services/videoAnalysisService';
import './VideoUploadPage.css';

const VideoUploadPage = () => {
  const navigate = useNavigate();
  const [dragActive, setDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState(null);
  const [error, setError] = useState(null);
  
  // 새로 추가: 분석 옵션 상태
  const [analysisType, setAnalysisType] = useState('enhanced');
  const [customAnalysisOptions, setCustomAnalysisOptions] = useState({
    object_detection: true,
    clip_analysis: true,
    ocr: false,
    vqa: false,
    scene_graph: false,
    enhanced_caption: true
  });
  const [autoStartAnalysis, setAutoStartAnalysis] = useState(true);

  // 분석 타입 옵션
  const analysisTypes = {
    basic: {
      name: '기본 분석',
      description: '객체 감지 및 기본 캡션 생성',
      features: ['객체 감지', '기본 캡션'],
      estimatedTime: '빠름 (2-5분)',
      icon: '🔍'
    },
    enhanced: {
      name: '향상된 분석',
      description: 'CLIP 기반 씬 분석 및 OCR 포함',
      features: ['객체 감지', 'CLIP 씬 분석', 'OCR 텍스트 추출', '고급 캡션'],
      estimatedTime: '보통 (5-10분)',
      icon: '🎯'
    },
    comprehensive: {
      name: '종합 분석',
      description: '모든 AI 기능을 활용한 완전 분석',
      features: ['객체 감지', 'CLIP 분석', 'OCR', 'VQA', 'Scene Graph', '고급 캡션'],
      estimatedTime: '느림 (10-20분)',
      icon: '🧠'
    },
    custom: {
      name: '사용자 정의',
      description: '원하는 분석 기능만 선택',
      features: ['선택한 기능만'],
      estimatedTime: '선택에 따라 다름',
      icon: '⚙️'
    }
  };

  // 개별 분석 옵션 정보
  const analysisFeatures = {
    object_detection: {
      name: '객체 감지',
      description: 'YOLO 기반 객체 감지 및 분류',
      icon: '🎯',
      required: true
    },
    clip_analysis: {
      name: 'CLIP 씬 분석',
      description: '고급 씬 이해 및 컨텍스트 분석',
      icon: '🖼️',
      required: false
    },
    ocr: {
      name: 'OCR 텍스트 추출',
      description: '이미지 내 텍스트 인식 및 추출',
      icon: '📝',
      required: false
    },
    vqa: {
      name: 'VQA 질문답변',
      description: '이미지에 대한 질문 생성 및 답변',
      icon: '❓',
      required: false
    },
    scene_graph: {
      name: 'Scene Graph',
      description: '객체간 관계 및 상호작용 분석',
      icon: '🕸️',
      required: false
    },
    enhanced_caption: {
      name: '고급 캡션',
      description: '모든 분석 결과를 통합한 상세 캡션',
      icon: '💬',
      required: false
    }
  };

  // 드래그 앤 드롭 이벤트 핸들러 (기존과 동일)
  const handleDrag = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileSelect = (file) => {
    setError(null);
    
    // 파일 타입 검증
    const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/webm'];
    if (!allowedTypes.includes(file.type)) {
      setError('지원하지 않는 파일 형식입니다. MP4, AVI, MOV, MKV, WEBM 파일만 업로드할 수 있습니다.');
      return;
    }

    // 파일 크기 검증 (500MB 제한으로 증가)
    const maxSize = 500 * 1024 * 1024; // 500MB
    if (file.size > maxSize) {
      setError('파일 크기가 너무 큽니다. 500MB 이하의 파일만 업로드할 수 있습니다.');
      return;
    }

    setSelectedFile(file);
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleAnalysisTypeChange = (type) => {
    setAnalysisType(type);
    
    // 미리 정의된 타입인 경우 옵션 자동 설정
    if (type !== 'custom') {
      const typeConfig = getAnalysisConfig(type);
      setCustomAnalysisOptions(typeConfig);
    }
  };

  const handleCustomOptionChange = (option, value) => {
    setCustomAnalysisOptions(prev => ({
      ...prev,
      [option]: value
    }));
  };

  const getAnalysisConfig = (type) => {
    const configs = {
      basic: {
        object_detection: true,
        clip_analysis: false,
        ocr: false,
        vqa: false,
        scene_graph: false,
        enhanced_caption: false
      },
      enhanced: {
        object_detection: true,
        clip_analysis: true,
        ocr: true,
        vqa: false,
        scene_graph: false,
        enhanced_caption: true
      },
      comprehensive: {
        object_detection: true,
        clip_analysis: true,
        ocr: true,
        vqa: true,
        scene_graph: true,
        enhanced_caption: true
      }
    };
    
    return configs[type] || configs.enhanced;
  };

  const uploadVideo = async () => {
    if (!selectedFile) return;

    setUploading(true);
    setUploadProgress(0);
    setError(null);

    try {
      const result = await videoAnalysisService.uploadVideo(
        selectedFile,
        (progress) => setUploadProgress(progress)
      );

      setUploadResult(result);
      setSelectedFile(null);
      
      // 업로드 완료 후 자동 분석 시작 옵션 확인
      if (autoStartAnalysis) {
        startAnalysis(result.video_id);
      } else if (window.confirm('업로드가 완료되었습니다. 지금 분석을 시작하시겠습니까?')) {
        startAnalysis(result.video_id);
      }

    } catch (error) {
      setError(error.message);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const startAnalysis = async (videoId) => {
    try {
      // 분석 설정 준비
      const analysisConfig = analysisType === 'custom' ? customAnalysisOptions : getAnalysisConfig(analysisType);
      
      await videoAnalysisService.analyzeVideoEnhanced(videoId, {
        analysisType: analysisType,
        analysisConfig: analysisConfig,
        enhancedAnalysis: analysisType !== 'basic'
      });
      
      alert(`${analysisTypes[analysisType].name} 분석이 시작되었습니다. 분석 페이지에서 진행 상황을 확인할 수 있습니다.`);
      navigate('/video-analysis');
    } catch (error) {
      setError(`분석 시작 실패: ${error.message}`);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const getSelectedFeatures = () => {
    if (analysisType === 'custom') {
      return Object.keys(customAnalysisOptions).filter(key => customAnalysisOptions[key]);
    }
    return analysisTypes[analysisType].features;
  };

  return (
    <div className="video-upload-page">
      <div className="upload-container">
        {/* 헤더 */}
        <div className="upload-header">
          <h1>🎥 고급 비디오 업로드 & 분석</h1>
          <p>AI 기반 비디오 분석으로 더 깊이 있는 인사이트를 얻으세요</p>
        </div>

        {/* 분석 타입 선택 */}
        <div className="analysis-selection-section">
          <h3>📊 분석 타입 선택</h3>
          <div className="analysis-types-grid">
            {Object.entries(analysisTypes).map(([key, config]) => (
              <div 
                key={key}
                className={`analysis-type-card ${analysisType === key ? 'selected' : ''}`}
                onClick={() => handleAnalysisTypeChange(key)}
              >
                <div className="analysis-type-header">
                  <span className="analysis-icon">{config.icon}</span>
                  <h4>{config.name}</h4>
                </div>
                <p className="analysis-description">{config.description}</p>
                <div className="analysis-features">
                  {config.features.slice(0, 3).map((feature, idx) => (
                    <span key={idx} className="feature-tag">{feature}</span>
                  ))}
                  {config.features.length > 3 && (
                    <span className="feature-tag more">+{config.features.length - 3}</span>
                  )}
                </div>
                <div className="analysis-time">
                  <span className="time-estimate">{config.estimatedTime}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* 사용자 정의 분석 옵션 */}
        {analysisType === 'custom' && (
          <div className="custom-analysis-section">
            <h3>⚙️ 사용자 정의 분석 옵션</h3>
            <div className="custom-options-grid">
              {Object.entries(analysisFeatures).map(([key, feature]) => (
                <div key={key} className="custom-option-card">
                  <div className="option-header">
                    <span className="option-icon">{feature.icon}</span>
                    <div className="option-info">
                      <h5>{feature.name}</h5>
                      <p>{feature.description}</p>
                    </div>
                    <label className="option-toggle">
                      <input
                        type="checkbox"
                        checked={customAnalysisOptions[key]}
                        onChange={(e) => handleCustomOptionChange(key, e.target.checked)}
                        disabled={feature.required}
                      />
                      <span className="toggle-slider"></span>
                    </label>
                  </div>
                  {feature.required && (
                    <span className="required-badge">필수</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 선택된 기능 요약 */}
        <div className="selected-features-summary">
          <h4>🎯 선택된 분석 기능</h4>
          <div className="features-list">
            {getSelectedFeatures().map((feature, idx) => (
              <span key={idx} className="selected-feature">{feature}</span>
            ))}
          </div>
          <div className="analysis-estimate">
            <span>예상 분석 시간: {analysisTypes[analysisType].estimatedTime}</span>
          </div>
        </div>

        {/* 업로드 영역 */}
        <div 
          className={`upload-area ${dragActive ? 'drag-active' : ''} ${selectedFile ? 'file-selected' : ''}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          {!selectedFile ? (
            <>
              <div className="upload-icon">📁</div>
              <h3>파일을 드래그하여 놓거나 클릭하여 선택하세요</h3>
              <p>지원 형식: MP4, AVI, MOV, MKV, WEBM (최대 500MB)</p>
              <input
                type="file"
                accept="video/*"
                onChange={handleFileInput}
                className="file-input"
                id="file-input"
              />
              <label htmlFor="file-input" className="file-label">
                파일 선택
              </label>
            </>
          ) : (
            <div className="selected-file">
              <div className="file-icon">🎬</div>
              <div className="file-info">
                <h4>{selectedFile.name}</h4>
                <p>크기: {formatFileSize(selectedFile.size)}</p>
                <p>형식: {selectedFile.type}</p>
                <p>분석 타입: {analysisTypes[analysisType].name}</p>
              </div>
              <button 
                className="remove-file"
                onClick={() => setSelectedFile(null)}
              >
                ✕
              </button>
            </div>
          )}
        </div>

        {/* 자동 분석 시작 옵션 */}
        <div className="auto-analysis-option">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={autoStartAnalysis}
              onChange={(e) => setAutoStartAnalysis(e.target.checked)}
            />
            <span>업로드 완료 후 자동으로 분석 시작</span>
          </label>
        </div>

        {/* 업로드 진행률 */}
        {uploading && (
          <div className="upload-progress">
            <div className="progress-header">
              <span>업로드 중...</span>
              <span>{uploadProgress}%</span>
            </div>
            <div className="progress-bar">
              <div 
                className="progress-fill"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          </div>
        )}

        {/* 오류 메시지 */}
        {error && (
          <div className="error-message">
            <span className="error-icon">⚠️</span>
            {error}
          </div>
        )}

        {/* 성공 메시지 */}
        {uploadResult && (
          <div className="success-message">
            <span className="success-icon">✅</span>
            <div>
              <h4>업로드 성공!</h4>
              <p>{uploadResult.message}</p>
              <p>선택된 분석: <strong>{analysisTypes[analysisType].name}</strong></p>
            </div>
          </div>
        )}

        {/* 버튼 영역 */}
        <div className="button-area">
          <button
            className="upload-button primary"
            onClick={uploadVideo}
            disabled={!selectedFile || uploading}
          >
            {uploading ? '업로드 중...' : `업로드 & ${analysisTypes[analysisType].name} 시작`}
          </button>
          
          <button
            className="secondary-button"
            onClick={() => navigate('/video-analysis')}
          >
            분석 현황 보기
          </button>
          
          <button
            className="secondary-button"
            onClick={() => navigate('/integrated-chat')}
          >
            채팅으로 탐색
          </button>
        </div>

        {/* 고급 분석 기능 소개 */}
        <div className="features-introduction">
          <h3>🚀 고급 분석 기능 소개</h3>
          <div className="features-showcase">
            <div className="feature-showcase">
              <div className="feature-icon">🎯</div>
              <div className="feature-content">
                <h4>CLIP 기반 씬 분석</h4>
                <p>OpenAI CLIP 모델을 활용하여 이미지의 의미적 컨텍스트를 이해하고 고급 씬 분류를 수행합니다.</p>
              </div>
            </div>
            
            <div className="feature-showcase">
              <div className="feature-icon">📝</div>
              <div className="feature-content">
                <h4>OCR 텍스트 추출</h4>
                <p>EasyOCR을 사용하여 비디오 내 한글/영문 텍스트를 정확하게 인식하고 추출합니다.</p>
              </div>
            </div>
            
            <div className="feature-showcase">
              <div className="feature-icon">❓</div>
              <div class="feature-content">
                <h4>VQA 질문답변</h4>
                <p>BLIP 모델을 활용하여 이미지에 대한 질문을 생성하고 답변하여 더 깊이 있는 분석을 제공합니다.</p>
              </div>
            </div>
            
            <div className="feature-showcase">
              <div className="feature-icon">🕸️</div>
              <div className="feature-content">
                <h4>Scene Graph 생성</h4>
                <p>객체간의 관계와 상호작용을 분석하여 복잡한 씬의 구조를 이해합니다.</p>
              </div>
            </div>
          </div>
        </div>

        {/* 네비게이션 */}
        <div className="navigation">
          <button onClick={() => navigate('/')} className="nav-button">
            🏠 홈으로
          </button>
          <button onClick={() => navigate('/video-analysis')} className="nav-button">
            📊 분석 현황
          </button>
          <button onClick={() => navigate('/integrated-chat')} className="nav-button">
            💬 채팅
          </button>
        </div>
      </div>
    </div>
  );
};

export default VideoUploadPage;