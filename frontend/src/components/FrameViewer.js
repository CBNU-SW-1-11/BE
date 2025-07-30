// components/FrameViewer.js - 바운딩 박스 표시 기능 포함
import React, { useRef, useEffect, useState } from 'react';
import './FrameViewer.css';

const FrameViewer = ({ 
  frameUrl, 
  frameData, 
  bboxAnnotations = [], 
  onClose,
  frameId,
  timestamp 
}) => {
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });

  // 바운딩 박스 색상 맵
  const colorMap = {
    'person': '#FF6B6B',      // 빨간색
    'car': '#4ECDC4',         // 청록색
    'bicycle': '#45B7D1',     // 파란색
    'motorcycle': '#96CEB4',  // 연두색
    'dog': '#FECA57',         // 노란색
    'cat': '#FF9FF3',         // 분홍색
    'chair': '#54A0FF',       // 파란색
    'cup': '#5F27CD',         // 보라색
    'cell_phone': '#00D2D3',  // 사이안
    'laptop': '#FF6348',      // 주황색
    'default': '#FFFFFF'      // 기본 흰색
  };

  useEffect(() => {
    if (imageLoaded && bboxAnnotations.length > 0) {
      drawBoundingBoxes();
    }
  }, [imageLoaded, bboxAnnotations]);

  const handleImageLoad = () => {
    const img = imageRef.current;
    if (img) {
      setImageDimensions({
        width: img.naturalWidth,
        height: img.naturalHeight
      });
      setImageLoaded(true);
    }
  };

  const drawBoundingBoxes = () => {
    const canvas = canvasRef.current;
    const image = imageRef.current;
    
    if (!canvas || !image || !imageLoaded) return;

    const ctx = canvas.getContext('2d');
    
    // Canvas 크기를 이미지 표시 크기에 맞춤
    const rect = image.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    
    // Canvas 초기화
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // 바운딩 박스 그리기
    bboxAnnotations.forEach((annotation, index) => {
      if (annotation.bbox && annotation.bbox.length === 4) {
        const [x1, y1, x2, y2] = annotation.bbox;
        
        // 정규화된 좌표를 실제 Canvas 좌표로 변환
        const canvasX1 = x1 * canvas.width;
        const canvasY1 = y1 * canvas.height;
        const canvasX2 = x2 * canvas.width;
        const canvasY2 = y2 * canvas.height;
        
        const width = canvasX2 - canvasX1;
        const height = canvasY2 - canvasY1;
        
        // 색상 선택
        const color = colorMap[annotation.match] || colorMap.default;
        
        // 바운딩 박스 그리기
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(canvasX1, canvasY1, width, height);
        
        // 반투명 배경
        ctx.fillStyle = color + '20'; // 투명도 20%
        ctx.fillRect(canvasX1, canvasY1, width, height);
        
        // 라벨 배경
        const label = `${annotation.match} ${(annotation.confidence * 100).toFixed(0)}%`;
        const labelWidth = ctx.measureText(label).width + 10;
        const labelHeight = 20;
        
        ctx.fillStyle = color;
        ctx.fillRect(canvasX1, canvasY1 - labelHeight, labelWidth, labelHeight);
        
        // 라벨 텍스트
        ctx.fillStyle = '#FFFFFF';
        ctx.font = '12px Arial';
        ctx.fillText(label, canvasX1 + 5, canvasY1 - 5);
      }
    });
  };

  const formatTimestamp = (seconds) => {
    if (!seconds) return "0:00";
    const minutes = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${minutes}:${secs.toString().padStart(2, '0')}`;
  };

  return (
    <div className="frame-viewer-overlay" onClick={onClose}>
      <div className="frame-viewer-content" onClick={(e) => e.stopPropagation()}>
        {/* 헤더 */}
        <div className="frame-viewer-header">
          <div className="frame-info">
            <h3>🖼️ 프레임 #{frameId}</h3>
            <span className="timestamp">⏱️ {formatTimestamp(timestamp)}</span>
            {bboxAnnotations.length > 0 && (
              <span className="bbox-count">
                📦 {bboxAnnotations.length}개 객체 감지됨
              </span>
            )}
          </div>
          <button className="close-button" onClick={onClose}>
            ✕
          </button>
        </div>

        {/* 이미지 및 바운딩 박스 */}
        <div className="frame-image-container">
          <img
            ref={imageRef}
            src={frameUrl}
            alt={`Frame ${frameId}`}
            className="frame-image"
            onLoad={handleImageLoad}
            onError={(e) => {
              console.error('프레임 이미지 로드 실패:', e);
              e.target.src = '/placeholder-image.png'; // fallback 이미지
            }}
          />
          <canvas
            ref={canvasRef}
            className="bbox-overlay"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              pointerEvents: 'none'
            }}
          />
        </div>

        {/* 감지된 객체 목록 */}
        {bboxAnnotations.length > 0 && (
          <div className="detected-objects-panel">
            <h4>🎯 감지된 객체</h4>
            <div className="objects-list">
              {bboxAnnotations.map((annotation, index) => (
                <div key={index} className="object-item">
                  <div 
                    className="object-color-indicator"
                    style={{ backgroundColor: colorMap[annotation.match] || colorMap.default }}
                  ></div>
                  <div className="object-info">
                    <span className="object-name">{annotation.match}</span>
                    <span className="object-confidence">
                      {(annotation.confidence * 100).toFixed(1)}%
                    </span>
                    {annotation.color_description && (
                      <span className="object-color">
                        색상: {annotation.color_description}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* 프레임 상세 정보 */}
        {frameData && (
          <div className="frame-details-panel">
            <h4>📊 프레임 상세 정보</h4>
            <div className="frame-details">
              {frameData.caption && (
                <div className="detail-item">
                  <strong>설명:</strong>
                  <p>{frameData.caption}</p>
                </div>
              )}
              
              {frameData.advanced_analysis && (
                <div className="advanced-analysis-details">
                  {/* CLIP 분석 결과 */}
                  {frameData.advanced_analysis.clip_analysis && (
                    <div className="analysis-section">
                      <h5>🖼️ CLIP 씬 분석</h5>
                      <div className="clip-results">
                        {frameData.advanced_analysis.clip_analysis.scene_type && (
                          <span className="scene-type-tag">
                            {frameData.advanced_analysis.clip_analysis.scene_type}
                          </span>
                        )}
                      </div>
                    </div>
                  )}

                  {/* OCR 결과 */}
                  {frameData.advanced_analysis.ocr_text?.texts?.length > 0 && (
                    <div className="analysis-section">
                      <h5>📝 추출된 텍스트</h5>
                      <div className="ocr-results">
                        {frameData.advanced_analysis.ocr_text.texts.map((text, idx) => (
                          <div key={idx} className="ocr-text-item">
                            <span className="text-content">"{text.text}"</span>
                            <span className="text-confidence">
                              {(text.confidence * 100).toFixed(1)}%
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* 액션 버튼 */}
        <div className="frame-actions">
          <button 
            className="action-button download"
            onClick={() => {
              const link = document.createElement('a');
              link.href = frameUrl;
              link.download = `frame_${frameId}.jpg`;
              link.click();
            }}
          >
            💾 이미지 저장
          </button>
          <button 
            className="action-button share"
            onClick={() => {
              if (navigator.share) {
                navigator.share({
                  title: `프레임 #${frameId}`,
                  text: `${bboxAnnotations.length}개 객체가 감지된 프레임`,
                  url: frameUrl
                });
              } else {
                navigator.clipboard.writeText(frameUrl);
                alert('이미지 URL이 클립보드에 복사되었습니다.');
              }
            }}
          >
            🔗 공유
          </button>
        </div>
      </div>
    </div>
  );
};

export default FrameViewer;