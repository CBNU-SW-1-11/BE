import React, { useState, useRef } from 'react';
import { videoAnalysisService } from '../../services/videoAnalysisService';

const VideoUpload = ({ onVideoUploaded }) => {
  const [uploadProgress, setUploadProgress] = useState(0);
  const [uploading, setUploading] = useState(false);
  const [dragActive, setDragActive] = useState(false);
  const [uploadStatus, setUploadStatus] = useState('');
  const fileInputRef = useRef(null);

  const handleFileSelect = (files) => {
    const file = files[0];
    if (!file) return;

    // 파일 검증
    const allowedTypes = ['video/mp4', 'video/avi', 'video/mov', 'video/mkv', 'video/webm'];
    if (!allowedTypes.includes(file.type)) {
      setUploadStatus('지원하지 않는 파일 형식입니다. MP4, AVI, MOV, MKV, WebM 파일만 업로드 가능합니다.');
      return;
    }

    // 파일 크기 검증 (100MB 제한)
    const maxSize = 100 * 1024 * 1024; // 100MB
    if (file.size > maxSize) {
      setUploadStatus('파일 크기가 너무 큽니다. 100MB 이하의 파일만 업로드 가능합니다.');
      return;
    }

    uploadVideo(file);
  };

  const uploadVideo = async (file) => {
    try {
      setUploading(true);
      setUploadProgress(0);
      setUploadStatus('업로드 중...');

      const result = await videoAnalysisService.uploadVideo(file, setUploadProgress);
      
      setUploadStatus(`✅ ${result.message}`);
      setTimeout(() => {
        setUploadStatus('');
        setUploadProgress(0);
        if (onVideoUploaded) onVideoUploaded();
      }, 2000);

    } catch (error) {
      setUploadStatus(`❌ ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files);
    }
  };

  const handleButtonClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">📤 비디오 업로드</h2>
      
      {/* Upload Area */}
      <div
        className={`relative border-2 border-dashed rounded-lg p-8 transition-colors ${
          dragActive 
            ? 'border-blue-400 bg-blue-50' 
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDragEnter={handleDrag}
        onDragLeave={handleDrag}
        onDragOver={handleDrag}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          onChange={(e) => handleFileSelect(e.target.files)}
          className="hidden"
        />
        
        <div className="text-center">
          <div className="text-4xl mb-4">🎬</div>
          <h3 className="text-lg font-medium text-gray-900 mb-2">
            {dragActive ? '파일을 여기에 놓으세요' : '비디오 파일을 업로드하세요'}
          </h3>
          <p className="text-gray-600 mb-4">
            드래그 앤 드롭하거나 클릭하여 파일을 선택하세요
          </p>
          
          <button
            onClick={handleButtonClick}
            disabled={uploading}
            className="bg-blue-600 text-white px-6 py-2 rounded-md hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
          >
            {uploading ? '업로드 중...' : '파일 선택'}
          </button>
          
          <p className="text-xs text-gray-500 mt-2">
            지원 형식: MP4, AVI, MOV, MKV, WebM (최대 100MB)
          </p>
        </div>
      </div>

      {/* Upload Progress */}
      {uploading && (
        <div className="mt-4">
          <div className="flex justify-between text-sm text-gray-600 mb-1">
            <span>업로드 진행률</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-blue-600 h-2 rounded-full transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            ></div>
          </div>
        </div>
      )}

      {/* Status Message */}
      {uploadStatus && (
        <div className={`mt-4 p-3 rounded-md ${
          uploadStatus.includes('❌') 
            ? 'bg-red-50 text-red-700 border border-red-200'
            : uploadStatus.includes('✅')
            ? 'bg-green-50 text-green-700 border border-green-200'
            : 'bg-blue-50 text-blue-700 border border-blue-200'
        }`}>
          {uploadStatus}
        </div>
      )}

      {/* Upload Tips */}
      <div className="mt-6 bg-gray-50 rounded-md p-4">
        <h4 className="font-medium text-gray-900 mb-2">💡 업로드 팁</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• 더 나은 분석을 위해 화질이 좋은 비디오를 사용하세요</li>
          <li>• 업로드 후 자동으로 기본 분석이 시작됩니다</li>
          <li>• Enhanced 분석을 원하면 비디오 목록에서 설정할 수 있습니다</li>
          <li>• 분석 완료 후 AI와 대화하여 내용을 검색할 수 있습니다</li>
        </ul>
      </div>
    </div>
  );
};

export default VideoUpload;