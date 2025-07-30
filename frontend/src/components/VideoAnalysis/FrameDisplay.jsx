import React, { useState } from 'react';

const FrameDisplay = ({ frameData, selectedVideo }) => {
  const [imageLoading, setImageLoading] = useState(false);
  const [imageError, setImageError] = useState(false);

  const handleImageLoad = () => {
    setImageLoading(false);
    setImageError(false);
  };

  const handleImageError = () => {
    setImageLoading(false);
    setImageError(true);
  };

  const formatTimestamp = (seconds) => {
    if (!seconds) return '0:00';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getResponseTypeIcon = (type) => {
    switch (type) {
      case 'enhanced_search_with_image':
        return '🔍';
      case 'highlight_detection':
        return '⭐';
      case 'video_summary':
        return '📝';
      case 'video_info':
        return 'ℹ️';
      default:
        return '📸';
    }
  };

  const getResponseTypeName = (type) => {
    switch (type) {
      case 'enhanced_search_with_image':
        return '검색 결과';
      case 'highlight_detection':
        return '하이라이트';
      case 'video_summary':
        return '요약';
      case 'video_info':
        return '정보';
      default:
        return '프레임';
    }
  };

  if (!frameData && !selectedVideo) {
    return (
      <div className="bg-white rounded-lg shadow p-6 text-center">
        <div className="text-gray-400 text-4xl mb-4">🖼️</div>
        <h3 className="text-lg font-medium text-gray-900 mb-2">프레임 미리보기</h3>
        <p className="text-gray-600">
          비디오를 선택하고 AI와 대화하면<br />
          관련 프레임이 여기에 표시됩니다.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200">
        <h3 className="font-medium text-gray-900">🖼️ 프레임 미리보기</h3>
        {frameData && (
          <div className="flex items-center space-x-2 mt-1">
            <span className="text-lg">{getResponseTypeIcon(frameData.responseType)}</span>
            <span className="text-sm text-gray-600">
              {getResponseTypeName(frameData.responseType)}
            </span>
          </div>
        )}
      </div>

      {/* Image Display */}
      <div className="p-4">
        {frameData ? (
          <div>
            {/* Frame Image */}
            <div className="relative bg-gray-100 rounded-lg overflow-hidden mb-4">
              {imageLoading && (
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="animate-spin h-8 w-8 border-2 border-blue-600 border-t-transparent rounded-full"></div>
                </div>
              )}
              
              {imageError ? (
                <div className="aspect-video flex items-center justify-center bg-gray-100">
                  <div className="text-center text-gray-500">
                    <div className="text-4xl mb-2">❌</div>
                    <p>이미지를 불러올 수 없습니다</p>
                  </div>
                </div>
              ) : (
                <img
                  src={frameData.imageUrl}
                  alt={`Frame ${frameData.frameNumber}`}
                  className="w-full h-auto object-contain"
                  onLoad={handleImageLoad}
                  onError={handleImageError}
                  style={{ minHeight: '200px' }}
                />
              )}
            </div>

            {/* Frame Info */}
            <div className="space-y-3">
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <span className="font-medium text-gray-700">프레임 번호:</span>
                  <div className="text-gray-900">{frameData.frameNumber}</div>
                </div>
                <div>
                  <span className="font-medium text-gray-700">타임스탬프:</span>
                  <div className="text-gray-900">
                    {formatTimestamp(frameData.timestamp)}
                  </div>
                </div>
              </div>

              {/* Enhanced Info */}
              {frameData.enhancedInfo && (
                <div className="bg-blue-50 rounded-md p-3">
                  <h4 className="font-medium text-blue-900 mb-2">상세 정보</h4>
                  <div className="space-y-2 text-sm">
                    {frameData.enhancedInfo.analysis_type && (
                      <div>
                        <span className="text-blue-700">분석 타입:</span>
                        <span className="ml-2 text-blue-900 capitalize">
                          {frameData.enhancedInfo.analysis_type}
                        </span>
                      </div>
                    )}
                    
                    {frameData.enhancedInfo.match_reasons && frameData.enhancedInfo.match_reasons.length > 0 && (
                      <div>
                        <span className="text-blue-700">매칭 이유:</span>
                        <div className="mt-1">
                          {frameData.enhancedInfo.match_reasons.map((reason, index) => (
                            <span key={index} className="inline-block bg-blue-100 text-blue-800 text-xs px-2 py-1 rounded mr-1 mb-1">
                              {reason}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    
                    {frameData.enhancedInfo.highlight_score && (
                      <div>
                        <span className="text-blue-700">하이라이트 점수:</span>
                        <span className="ml-2 text-blue-900">
                          {frameData.enhancedInfo.highlight_score}
                        </span>
                      </div>
                    )}
                    
                    {frameData.enhancedInfo.reasons && frameData.enhancedInfo.reasons.length > 0 && (
                      <div>
                        <span className="text-blue-700">선정 이유:</span>
                        <div className="mt-1">
                          {frameData.enhancedInfo.reasons.map((reason, index) => (
                            <div key={index} className="text-blue-800 text-xs">
                              • {reason}
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              {/* Action Buttons */}
              <div className="flex space-x-2 pt-2">
                <button
                  onClick={() => window.open(frameData.imageUrl, '_blank')}
                  className="flex-1 px-3 py-2 text-sm bg-gray-100 text-gray-700 rounded hover:bg-gray-200 transition-colors"
                >
                  🔍 크게 보기
                </button>
                <button
                  onClick={() => {
                    const link = document.createElement('a');
                    link.href = frameData.imageUrl;
                    link.download = `frame_${frameData.frameNumber}.jpg`;
                    link.click();
                  }}
                  className="flex-1 px-3 py-2 text-sm bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors"
                >
                  💾 다운로드
                </button>
              </div>
            </div>
          </div>
        ) : (
          <div className="text-center py-8">
            <div className="text-gray-400 text-4xl mb-4">🎬</div>
            <h4 className="font-medium text-gray-900 mb-2">
              {selectedVideo.filename}
            </h4>
            <p className="text-gray-600 text-sm mb-4">
              AI와 대화하여 프레임을 탐색해보세요
            </p>
            <div className="text-xs text-gray-500 space-y-1">
              <div>• "사람 찾아줘"로 검색하기</div>
              <div>• "하이라이트 보여줘"로 주요 장면 찾기</div>
              <div>• "요약해줘"로 전체 내용 파악</div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default FrameDisplay;