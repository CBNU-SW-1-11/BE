import React, { useState, useEffect } from 'react';

const ObjectDetectionVisualizer = ({ imageUrl, detectionResults }) => {
  const [image, setImage] = useState(null);
  const [imageLoaded, setImageLoaded] = useState(false);
  const [parsedObjects, setParsedObjects] = useState([]);
  const [hoveredObject, setHoveredObject] = useState(null);
  
  // 임의 색상 생성
  const getRandomColor = (index) => {
    const colors = [
      'rgba(255, 99, 132, 0.7)',
      'rgba(54, 162, 235, 0.7)',
      'rgba(255, 206, 86, 0.7)',
      'rgba(75, 192, 192, 0.7)',
      'rgba(153, 102, 255, 0.7)',
      'rgba(255, 159, 64, 0.7)',
      'rgba(199, 199, 199, 0.7)',
      'rgba(83, 102, 255, 0.7)',
      'rgba(40, 159, 100, 0.7)',
      'rgba(210, 59, 130, 0.7)',
    ];
    return colors[index % colors.length];
  };
  
  // 탐지된 객체 정보 파싱
  useEffect(() => {
    if (!detectionResults || !imageLoaded) return;
    
    // 텍스트에서 객체 정보 파싱
    const objectRegex = /(\d+\.\s*)?([a-zA-Z가-힣\s]+)(?:\s*[-:]\s*([^,\n]+))?/g;
    const objectList = [];
    let match;
    
    while ((match = objectRegex.exec(detectionResults)) !== null) {
      const name = match[2]?.trim();
      const description = match[3]?.trim() || '';
      
      if (name) {
        objectList.push({
          id: objectList.length,
          name,
          description,
          color: getRandomColor(objectList.length),
          // 가상의 경계 상자 - 실제로는 AI가 제공하는 정보를 사용해야 함
          boundingBox: generateRandomBoundingBox(),
        });
      }
    }
    
    setParsedObjects(objectList);
  }, [detectionResults, imageLoaded]);
  
  // 이미지 로드 핸들러
  const handleImageLoad = (e) => {
    setImage(e.target);
    setImageLoaded(true);
  };
  
  // 임의의 경계 상자 생성 (실제로는 AI가 제공하는 정보를 사용해야 함)
  const generateRandomBoundingBox = () => {
    return {
      x: Math.random() * 0.8, // 0-0.8 사이의 x 좌표 (정규화됨)
      y: Math.random() * 0.8, // 0-0.8 사이의 y 좌표 (정규화됨)
      width: Math.random() * 0.3 + 0.1, // 0.1-0.4 사이의 너비 (정규화됨)
      height: Math.random() * 0.3 + 0.1 // 0.1-0.4 사이의 높이 (정규화됨)
    };
  };

  return (
    <div className="flex flex-col md:flex-row gap-4 mt-4">
      {/* 이미지 및 경계 상자 */}
      <div className="relative max-w-md">
        {imageUrl ? (
          <>
            <img 
              src={imageUrl}
              alt="Analyzed" 
              className="w-full rounded-lg shadow-md" 
              onLoad={handleImageLoad}
            />
            
            {/* 경계 상자 오버레이 */}
            {imageLoaded && parsedObjects.map((obj) => (
              <div
                key={obj.id}
                style={{
                  position: 'absolute',
                  left: `${obj.boundingBox.x * 100}%`,
                  top: `${obj.boundingBox.y * 100}%`,
                  width: `${obj.boundingBox.width * 100}%`,
                  height: `${obj.boundingBox.height * 100}%`,
                  border: `2px solid ${obj.color}`,
                  backgroundColor: hoveredObject === obj.id ? obj.color : 'transparent',
                  borderRadius: '4px',
                  transition: 'all 0.2s ease',
                  cursor: 'pointer',
                }}
                onMouseEnter={() => setHoveredObject(obj.id)}
                onMouseLeave={() => setHoveredObject(null)}
              >
                <div
                  style={{
                    position: 'absolute',
                    top: '-25px',
                    left: '0',
                    backgroundColor: obj.color,
                    color: 'white',
                    padding: '2px 6px',
                    borderRadius: '4px',
                    fontSize: '12px',
                    fontWeight: 'bold',
                    whiteSpace: 'nowrap',
                    opacity: hoveredObject === obj.id ? 1 : 0.8,
                  }}
                >
                  {obj.name}
                </div>
              </div>
            ))}
          </>
        ) : (
          <div className="w-full h-64 bg-gray-200 rounded-lg flex items-center justify-center">
            <p className="text-gray-500">이미지 없음</p>
          </div>
        )}
      </div>
      
      {/* 객체 목록 */}
      <div className="flex-1">
        <h3 className="font-medium text-lg mb-2">인식된 객체</h3>
        
        {parsedObjects.length > 0 ? (
          <ul className="space-y-2 max-h-64 overflow-y-auto pr-2">
            {parsedObjects.map((obj) => (
              <li 
                key={obj.id}
                className="p-2 rounded-md flex items-start gap-3 transition-colors"
                style={{
                  backgroundColor: hoveredObject === obj.id ? obj.color + '30' : 'transparent',
                  borderLeft: `4px solid ${obj.color}`
                }}
                onMouseEnter={() => setHoveredObject(obj.id)}
                onMouseLeave={() => setHoveredObject(null)}
              >
                <div 
                  className="w-4 h-4 mt-1 rounded-full flex-shrink-0"
                  style={{ backgroundColor: obj.color }}
                />
                <div>
                  <div className="font-medium">{obj.name}</div>
                  {obj.description && (
                    <div className="text-sm text-gray-600">{obj.description}</div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-500">인식된 객체가 없습니다.</p>
        )}
      </div>
    </div>
  );
};

export default ObjectDetectionVisualizer;