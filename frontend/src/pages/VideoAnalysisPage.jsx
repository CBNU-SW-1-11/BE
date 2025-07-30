// import React, { useState, useEffect } from 'react';
// import VideoUpload from '../components/VideoAnalysis/VideoUpload';
// import VideoList from '../components/VideoAnalysis/VideoList';
// import ChatInterface from '../components/VideoAnalysis/ChatInterface';
// import FrameDisplay from '../components/VideoAnalysis/FrameDisplay';
// import { videoAnalysisService } from '../services/videoAnalysisService';

// const VideoAnalysisPage = () => {
//   const [videos, setVideos] = useState([]);
//   const [selectedVideo, setSelectedVideo] = useState(null);
//   const [currentFrame, setCurrentFrame] = useState(null);
//   const [loading, setLoading] = useState(false);
//   const [error, setError] = useState('');
//   const [activeTab, setActiveTab] = useState('upload'); // upload, videos, chat

//   useEffect(() => {
//     loadVideos();
//   }, []);

//   const loadVideos = async () => {
//     try {
//       setLoading(true);
//       const data = await videoAnalysisService.getVideoList();
//       setVideos(data.videos || []);
//     } catch (err) {
//       setError(err.message);
//     } finally {
//       setLoading(false);
//     }
//   };

//   const handleVideoSelect = (video) => {
//     setSelectedVideo(video);
//     setActiveTab('chat');
//   };

//   const handleFrameReceived = (frameData) => {
//     setCurrentFrame(frameData);
//   };

//   const handleVideoUploaded = () => {
//     loadVideos();
//     setActiveTab('videos');
//   };

//   return (
//     <div className="min-h-screen bg-gray-50">
//       {/* Header */}
//       <div className="bg-white shadow-sm border-b">
//         <div className="max-w-7xl mx-auto px-4 py-4">
//           <h1 className="text-2xl font-bold text-gray-900">🎬 AI 비디오 분석</h1>
//           <p className="text-gray-600 mt-1">비디오를 업로드하고 AI로 내용을 분석하세요</p>
//         </div>
//       </div>

//       {/* Tab Navigation */}
//       <div className="max-w-7xl mx-auto px-4 py-4">
//         <div className="border-b border-gray-200">
//           <nav className="-mb-px flex space-x-8">
//             {[
//               { id: 'upload', name: '업로드', icon: '📤' },
//               { id: 'videos', name: '비디오 목록', icon: '📋' },
//               { id: 'chat', name: '분석 채팅', icon: '💬' }
//             ].map((tab) => (
//               <button
//                 key={tab.id}
//                 onClick={() => setActiveTab(tab.id)}
//                 className={`whitespace-nowrap py-2 px-1 border-b-2 font-medium text-sm ${
//                   activeTab === tab.id
//                     ? 'border-blue-500 text-blue-600'
//                     : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
//                 }`}
//               >
//                 <span className="mr-2">{tab.icon}</span>
//                 {tab.name}
//               </button>
//             ))}
//           </nav>
//         </div>
//       </div>

//       {/* Error Display */}
//       {error && (
//         <div className="max-w-7xl mx-auto px-4 mb-4">
//           <div className="bg-red-50 border border-red-200 rounded-md p-4">
//             <div className="flex">
//               <div className="flex-shrink-0">
//                 <span className="text-red-400">⚠️</span>
//               </div>
//               <div className="ml-3">
//                 <h3 className="text-sm font-medium text-red-800">오류가 발생했습니다</h3>
//                 <div className="mt-2 text-sm text-red-700">{error}</div>
//               </div>
//             </div>
//           </div>
//         </div>
//       )}

//       {/* Main Content */}
//       <div className="max-w-7xl mx-auto px-4 pb-8">
//         <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
//           {/* Left Panel - Tab Content */}
//           <div className="lg:col-span-2">
//             {activeTab === 'upload' && (
//               <VideoUpload onVideoUploaded={handleVideoUploaded} />
//             )}
            
//             {activeTab === 'videos' && (
//               <VideoList 
//                 videos={videos}
//                 loading={loading}
//                 onVideoSelect={handleVideoSelect}
//                 onRefresh={loadVideos}
//               />
//             )}
            
//             {activeTab === 'chat' && selectedVideo && (
//               <ChatInterface 
//                 selectedVideo={selectedVideo}
//                 onFrameReceived={handleFrameReceived}
//               />
//             )}
            
//             {activeTab === 'chat' && !selectedVideo && (
//               <div className="bg-white rounded-lg shadow p-8 text-center">
//                 <div className="text-gray-400 text-6xl mb-4">🎥</div>
//                 <h3 className="text-lg font-medium text-gray-900 mb-2">비디오를 선택하세요</h3>
//                 <p className="text-gray-600">
//                   분석할 비디오를 선택하여 AI와 대화를 시작하세요.
//                 </p>
//                 <button
//                   onClick={() => setActiveTab('videos')}
//                   className="mt-4 bg-blue-600 text-white px-4 py-2 rounded-md hover:bg-blue-700 transition-colors"
//                 >
//                   비디오 목록 보기
//                 </button>
//               </div>
//             )}
//           </div>

//           {/* Right Panel - Frame Display */}
//           <div className="lg:col-span-1">
//             <FrameDisplay 
//               frameData={currentFrame}
//               selectedVideo={selectedVideo}
//             />
            
//             {/* Selected Video Info */}
//             {selectedVideo && (
//               <div className="mt-4 bg-white rounded-lg shadow p-4">
//                 <h3 className="font-medium text-gray-900 mb-2">선택된 비디오</h3>
//                 <div className="text-sm text-gray-600 space-y-1">
//                   <p><strong>파일명:</strong> {selectedVideo.filename}</p>
//                   <p><strong>길이:</strong> {selectedVideo.duration?.toFixed(1)}초</p>
//                   <p><strong>상태:</strong> 
//                     <span className={`ml-1 px-2 py-1 rounded-full text-xs ${
//                       selectedVideo.is_analyzed 
//                         ? 'bg-green-100 text-green-800' 
//                         : 'bg-yellow-100 text-yellow-800'
//                     }`}>
//                       {selectedVideo.is_analyzed ? '분석 완료' : '분석 대기'}
//                     </span>
//                   </p>
//                   {selectedVideo.enhanced_analysis && (
//                     <p><strong>분석 타입:</strong> Enhanced</p>
//                   )}
//                 </div>
//               </div>
//             )}
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// export default VideoAnalysisPage;