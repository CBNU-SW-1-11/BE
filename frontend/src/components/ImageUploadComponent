// src/components/ImageUploadComponent.jsx
import React, { useState, useRef } from 'react';
import { Upload, X, Image as ImageIcon, FileText, Check } from 'lucide-react';
import { useChat } from '../context/ChatContext';

const ImageUploadComponent = () => {
  const [dragActive, setDragActive] = useState(false);
  const [uploadedImage, setUploadedImage] = useState(null);
  const [previewUrl, setPreviewUrl] = useState('');
  const [imageAnalysisMode, setImageAnalysisMode] = useState('describe'); // 'describe', 'ocr', 'objects'
  const fileInputRef = useRef(null);
  
  const { processImageUpload } = useChat();

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
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e) => {
    e.preventDefault();
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const handleFile = (file) => {
    // 이미지 파일 유효성 검사
    if (!file.type.match('image.*')) {
      alert('이미지 파일만 업로드 가능합니다.');
      return;
    }
    
    // 이미지 크기 제한 (10MB)
    if (file.size > 10 * 1024 * 1024) {
      alert('10MB 이하의 이미지만 업로드 가능합니다.');
      return;
    }
    
    // 이미지 정보 및 미리보기 설정
    setUploadedImage(file);
    setPreviewUrl(URL.createObjectURL(file));
  };

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  const clearImage = () => {
    setUploadedImage(null);
    setPreviewUrl('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const sendImageForAnalysis = () => {
    if (!uploadedImage) return;
    
    processImageUpload(uploadedImage, imageAnalysisMode);
    clearImage();
  };

  return (
    <div className="w-full mb-4">
      {!uploadedImage ? (
        <div 
          className={`border-2 border-dashed rounded-lg p-6 text-center ${
            dragActive ? 'border-blue-500 bg-blue-50' : 'border-gray-300'
          }`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <Upload className="mx-auto h-12 w-12 text-gray-400" />
          <p className="mt-2 text-sm text-gray-600">
            이미지를 끌어다 놓거나 클릭하여 업로드하세요
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleChange}
          />
          <button
            type="button"
            onClick={handleButtonClick}
            className="mt-3 px-4 py-2 bg-blue-500 text-white text-sm rounded-md hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            이미지 선택
          </button>
        </div>
      ) : (
        <div className="border rounded-lg p-4">
          <div className="flex justify-between items-center mb-3">
            <h3 className="font-medium">업로드된 이미지</h3>
            <button
              onClick={clearImage}
              className="text-gray-500 hover:text-gray-700"
            >
              <X size={18} />
            </button>
          </div>
          
          <div className="relative">
            <img
              src={previewUrl}
              alt="미리보기"
              className="w-full h-auto max-h-64 object-contain rounded"
            />
          </div>
          
          <div className="mt-4 flex flex-col gap-3">
            <div className="flex gap-2">
              <button
                className={`flex-1 py-2 px-3 text-sm rounded-md flex items-center justify-center gap-2 ${
                  imageAnalysisMode === 'describe' 
                  ? 'bg-blue-100 text-blue-700 border border-blue-300' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                onClick={() => setImageAnalysisMode('describe')}
              >
                <ImageIcon size={16} />
                이미지 설명
              </button>
              
              <button
                className={`flex-1 py-2 px-3 text-sm rounded-md flex items-center justify-center gap-2 ${
                  imageAnalysisMode === 'ocr' 
                  ? 'bg-blue-100 text-blue-700 border border-blue-300' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                onClick={() => setImageAnalysisMode('ocr')}
              >
                <FileText size={16} />
                텍스트 추출
              </button>
              
              <button
                className={`flex-1 py-2 px-3 text-sm rounded-md flex items-center justify-center gap-2 ${
                  imageAnalysisMode === 'objects' 
                  ? 'bg-blue-100 text-blue-700 border border-blue-300' 
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
                onClick={() => setImageAnalysisMode('objects')}
              >
                <ImageIcon size={16} />
                객체 인식
              </button>
            </div>
            
            <button
              onClick={sendImageForAnalysis}
              className="w-full py-2 bg-blue-500 text-white rounded-md hover:bg-blue-600 flex items-center justify-center gap-2"
            >
              <Check size={16} />
              이미지 분석하기
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default ImageUploadComponent;