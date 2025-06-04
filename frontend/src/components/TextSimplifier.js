import React, { useState } from 'react';
import { ArrowRight, Wand2, BookOpen, User, AlertTriangle, Check } from 'lucide-react';
import axios from 'axios';

// 쉬운 표현 전환 컴포넌트
const TextSimplifier = ({ originalText, onClose }) => {
  const [simplifiedText, setSimplifiedText] = useState('');
  const [targetAudience, setTargetAudience] = useState('general');
  const [isSimplifying, setIsSimplifying] = useState(false);
  const [error, setError] = useState(null);
  const [copied, setCopied] = useState(false);

  // 대상 청중별 설명
  const audienceDescriptions = {
    general: "일반인을 위한 쉬운 표현으로 변환합니다.",
    child: "7-12세 어린이가 이해할 수 있는 쉬운 단어와 문장으로 변환합니다.",
    elderly: "고령자가 이해하기 쉬운 용어와 문장 구조로 변환합니다.",
    foreigner: "한국어 학습자를 위해 기본 어휘와 단순한 문장 구조로 변환합니다."
  };

  // 텍스트 단순화 요청 함수
  const simplifyText = async () => {
    if (!originalText.trim()) return;
    
    setIsSimplifying(true);
    setError(null);
    
    try {
      // 백엔드 API에 전송할 데이터
      const requestData = {
        message: originalText,
        targetAudience: targetAudience,
        contentType: 'simplify',
        language: 'ko'
      };
      
      // API 요청
      const response = await axios.post('http://localhost:8000/api/simplify/', requestData);
      
      // 응답 처리
      if (response.data && response.data.simplified_text) {
        setSimplifiedText(response.data.simplified_text);
      } else {
        throw new Error('응답에 단순화된 텍스트가 없습니다.');
      }
      
    } catch (err) {
      console.error('API 오류:', err);
      setError(err.response?.data?.error || err.message || '변환 중 오류가 발생했습니다.');
      
      // 백엔드 연결 실패 시 폴백으로 데모 응답 사용 (개발용)
      if (process.env.NODE_ENV === 'development') {
        setTimeout(() => {
          let demoResponse = "";
          
          if (targetAudience === 'child') {
            demoResponse = "친구들이 쉽게 이해할 수 있게 설명해 볼게요!\n\n" + 
                          originalText
                          .replace(/[.;] /g, '.\n')
                          .replace(/복잡한/g, '어려운')
                          .replace(/구현/g, '만들기')
                          .replace(/기능/g, '할 수 있는 것')
                          .replace(/요청/g, '부탁')
                          .replace(/응답/g, '대답')
                          .replace(/혁신적인/g, '새롭고 멋진');
          } else if (targetAudience === 'elderly') {
            demoResponse = "노년층 분들이 이해하기 쉽게 풀어 설명해 드리겠습니다.\n\n" + 
                          originalText
                          .replace(/[.;] /g, '.\n')
                          .replace(/혁신적인/g, '새롭고 편리한')
                          .replace(/인터페이스/g, '화면')
                          .replace(/컴포넌트/g, '화면 부품')
                          .replace(/기능/g, '기능(역할)');
          } else if (targetAudience === 'foreigner') {
            demoResponse = "한국어 학습자를 위해 간단한 표현으로 설명합니다.\n\n" + 
                          originalText
                          .replace(/[.;] /g, '.\n')
                          .replace(/혁신적인/g, '새로운')
                          .replace(/개발/g, '만들기')
                          .replace(/복잡한/g, '어려운')
                          .replace(/인터페이스/g, '사용자 화면');
          } else {
            demoResponse = "일반 독자가 쉽게 이해할 수 있도록 변환했습니다.\n\n" + 
                          originalText
                          .replace(/[.;] /g, '.\n')
                          .replace(/혁신적인/g, '새로운')
                          .replace(/인터페이스/g, '사용자 화면')
                          .replace(/컴포넌트/g, '구성 요소');
          }
          
          setSimplifiedText(demoResponse);
          setError('백엔드 연결 실패: 데모 응답 사용 중');
        }, 1000);
      }
    } finally {
      setIsSimplifying(false);
    }
  };

  // 복사 버튼 핸들러
  const handleCopy = () => {
    navigator.clipboard.writeText(simplifiedText);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg overflow-hidden shadow-xl w-full max-w-4xl">
        {/* 헤더 */}
        <div className="bg-blue-600 text-white px-6 py-4">
          <h2 className="text-xl font-semibold">쉬운 표현으로 변환하기</h2>
          <p className="text-sm opacity-90">복잡한 텍스트를 이해하기 쉬운 표현으로 바꿔드립니다.</p>
        </div>
        
        {/* 대상 선택 */}
        <div className="bg-blue-50 p-4">
          <label className="block text-sm font-medium text-blue-700 mb-2">누구를 위한 표현인가요?</label>
          <div className="flex flex-wrap gap-2">
            <button 
              className={`px-4 py-2 rounded-full text-sm flex items-center ${targetAudience === 'general' ? 'bg-blue-600 text-white' : 'bg-white text-blue-600 border border-blue-200'}`}
              onClick={() => setTargetAudience('general')}
            >
              <User className="w-4 h-4 mr-1" /> 일반인
            </button>
            <button 
              className={`px-4 py-2 rounded-full text-sm flex items-center ${targetAudience === 'child' ? 'bg-blue-600 text-white' : 'bg-white text-blue-600 border border-blue-200'}`}
              onClick={() => setTargetAudience('child')}
            >
              <BookOpen className="w-4 h-4 mr-1" /> 어린이
            </button>
            <button 
              className={`px-4 py-2 rounded-full text-sm flex items-center ${targetAudience === 'elderly' ? 'bg-blue-600 text-white' : 'bg-white text-blue-600 border border-blue-200'}`}
              onClick={() => setTargetAudience('elderly')}
            >
              <User className="w-4 h-4 mr-1" /> 고령자
            </button>
            <button 
              className={`px-4 py-2 rounded-full text-sm flex items-center ${targetAudience === 'foreigner' ? 'bg-blue-600 text-white' : 'bg-white text-blue-600 border border-blue-200'}`}
              onClick={() => setTargetAudience('foreigner')}
            >
              <User className="w-4 h-4 mr-1" /> 외국인 학습자
            </button>
          </div>
          <p className="mt-2 text-xs text-blue-600">
            {audienceDescriptions[targetAudience]}
          </p>
        </div>
        
        {/* 본문 */}
        <div className="p-6 grid grid-cols-1 md:grid-cols-2 gap-4">
          {/* 원본 텍스트 */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700">원본 텍스트</label>
            <div className="border rounded-lg p-4 h-60 overflow-y-auto bg-gray-50">
              {originalText || "원본 텍스트가 없습니다."}
            </div>
          </div>
          
          {/* 변환 결과 */}
          <div className="space-y-2">
            <div className="flex justify-between">
              <label className="block text-sm font-medium text-gray-700">변환 결과</label>
              {simplifiedText && (
                <button 
                  onClick={handleCopy} 
                  className="text-xs text-blue-600 hover:text-blue-800 flex items-center"
                >
                  {copied ? <Check className="w-3 h-3 mr-1" /> : null}
                  {copied ? "복사됨" : "결과 복사"}
                </button>
              )}
            </div>
            
            <div className="border rounded-lg p-4 h-60 overflow-y-auto bg-blue-50 relative">
              {isSimplifying ? (
                <div className="flex items-center justify-center h-full">
                  <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-700"></div>
                </div>
              ) : error ? (
                <div>
                  <div className="flex items-center text-red-500 mb-2">
                    <AlertTriangle className="w-4 h-4 mr-2" />
                    {error}
                  </div>
                  {simplifiedText && (
                    <div className="whitespace-pre-line mt-4 pt-4 border-t border-red-200">
                      {simplifiedText}
                    </div>
                  )}
                </div>
              ) : simplifiedText ? (
                <div className="whitespace-pre-line">{simplifiedText}</div>
              ) : (
                <div className="text-gray-400 flex flex-col items-center justify-center h-full">
                  <Wand2 className="w-12 h-12 mb-2 text-blue-300" />
                  <p>변환 버튼을 눌러 쉬운 표현으로 바꿔보세요.</p>
                </div>
              )}
            </div>
          </div>
        </div>
        
        {/* 하단 버튼 */}
        <div className="bg-gray-50 px-6 py-4 flex justify-end space-x-3">
          <button 
            onClick={onClose}
            className="px-4 py-2 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-100"
          >
            닫기
          </button>
          <button 
            onClick={simplifyText}
            disabled={isSimplifying || !originalText}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
          >
            <Wand2 className="w-4 h-4 mr-2" />
            {isSimplifying ? "변환 중..." : "쉬운 표현으로 변환"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default TextSimplifier;