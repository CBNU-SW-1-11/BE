import React, { useState } from 'react';
import { videoAnalysisService } from '../../services/videoAnalysisService';

const VideoList = ({ videos, loading, onVideoSelect, onRefresh }) => {
  const [analyzingVideos, setAnalyzingVideos] = useState(new Set());
  const [error, setError] = useState('');

  const handleAnalyzeVideo = async (video, enableEnhanced = true) => {
    try {
      setAnalyzingVideos(prev => new Set([...prev, video.id]));
      setError('');
      
      await videoAnalysisService.analyzeVideo(video.id, enableEnhanced);
      
      // 분석 시작 후 목록 새로고침
      setTimeout(() => {
        onRefresh();
        setAnalyzingVideos(prev => {
          const newSet = new Set(prev);
          newSet.delete(video.id);
          return newSet;
        });
      }, 1000);
      
    } catch (err) {
      setError(err.message);
      setAnalyzingVideos(prev => {
        const newSet = new Set(prev);
        newSet.delete(video.id);
        return newSet;
      });
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString) => {
    return new Date(dateString).toLocaleDateString('ko-KR', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    });
  };

  const getStatusBadge = (video) => {
    if (analyzingVideos.has(video.id)) {
      return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
          <span className="animate-spin mr-1">⏳</span>
          분석 중
        </span>
      );
    }
    
    switch (video.analysis_status) {
      case 'completed':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-green-100 text-green-800">
            ✅ 완료
          </span>
        );
      case 'processing':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-yellow-100 text-yellow-800">
            🔄 처리 중
          </span>
        );
      case 'failed':
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-red-100 text-red-800">
            ❌ 실패
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-gray-100 text-gray-800">
            ⏸️ 대기
          </span>
        );
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow p-6">
        <div className="animate-pulse">
          <div className="h-4 bg-gray-200 rounded w-1/4 mb-4"></div>
          <div className="space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-20 bg-gray-200 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow">
      <div className="px-6 py-4 border-b border-gray-200">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-semibold text-gray-900">📋 비디오 목록</h2>
          <button
            onClick={onRefresh}
            className="text-gray-500 hover:text-gray-700 transition-colors"
            title="새로고침"
          >
            🔄
          </button>
        </div>
        <p className="text-gray-600 mt-1">총 {videos.length}개의 비디오</p>
      </div>

      {error && (
        <div className="mx-6 mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <p className="text-red-700 text-sm">{error}</p>
        </div>
      )}

      <div className="divide-y divide-gray-200">
        {videos.length === 0 ? (
          <div className="p-12 text-center">
            <div className="text-gray-400 text-4xl mb-4">📁</div>
            <h3 className="text-lg font-medium text-gray-900 mb-2">비디오가 없습니다</h3>
            <p className="text-gray-600">먼저 비디오를 업로드해주세요.</p>
          </div>
        ) : (
          videos.map((video) => (
            <div key={video.id} className="p-6 hover:bg-gray-50 transition-colors">
              <div className="flex items-center justify-between">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-3">
                    <div className="text-2xl">🎬</div>
                    <div className="flex-1 min-w-0">
                      <h3 className="text-lg font-medium text-gray-900 truncate">
                        {video.original_name || video.filename}
                      </h3>
                      <div className="flex items-center space-x-4 mt-1 text-sm text-gray-500">
                        <span>⏱️ {video.duration?.toFixed(1)}초</span>
                        <span>📏 {formatFileSize(video.file_size)}</span>
                        <span>📅 {formatDate(video.uploaded_at)}</span>
                      </div>
                      {video.enhanced_analysis && (
                        <div className="mt-1 text-sm text-blue-600">
                          ✨ Enhanced 분석 (성공률: {video.success_rate}%)
                        </div>
                      )}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  {getStatusBadge(video)}
                  
                  <div className="flex space-x-2">
                    {!video.is_analyzed && video.analysis_status !== 'processing' && (
                      <>
                        <button
                          onClick={() => handleAnalyzeVideo(video, false)}
                          disabled={analyzingVideos.has(video.id)}
                          className="px-3 py-1 text-sm bg-gray-600 text-white rounded hover:bg-gray-700 disabled:bg-gray-400 transition-colors"
                        >
                          기본 분석
                        </button>
                        <button
                          onClick={() => handleAnalyzeVideo(video, true)}
                          disabled={analyzingVideos.has(video.id)}
                          className="px-3 py-1 text-sm bg-blue-600 text-white rounded hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
                        >
                          Enhanced 분석
                        </button>
                      </>
                    )}
                    
                    {video.is_analyzed && (
                      <button
                        onClick={() => onVideoSelect(video)}
                        className="px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 transition-colors"
                      >
                        💬 채팅 시작
                      </button>
                    )}
                  </div>
                </div>
              </div>
              
              {/* Progress bar for analyzing videos */}
              {analyzingVideos.has(video.id) && (
                <div className="mt-3">
                  <div className="w-full bg-gray-200 rounded-full h-1">
                    <div className="bg-blue-600 h-1 rounded-full animate-pulse" style={{width: '30%'}}></div>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">분석을 시작하고 있습니다...</p>
                </div>
              )}
            </div>
          ))
        )}
      </div>
      
      {videos.length > 0 && (
        <div className="px-6 py-4 border-t border-gray-200 bg-gray-50">
          <div className="text-sm text-gray-600">
            💡 <strong>팁:</strong> Enhanced 분석은 더 정확한 객체 인식과 상세한 캡션을 제공하지만 시간이 더 오래 걸립니다.
          </div>
        </div>
      )}
    </div>
  );
};

export default VideoList;