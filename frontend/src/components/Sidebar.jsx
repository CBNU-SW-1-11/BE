

// // import React from "react";
// // import { useNavigate } from "react-router-dom";

// // const Sidebar = () => {
// //   const navigate = useNavigate();
  
// //   const quickPrompts = [
// //     { title: "코드 작성 도움", desc: "웹사이트의 스타일리시한 헤더를 위한 코드" },
// //     { title: "이미지 생성", desc: "취침 시간 이야기와 그림 만들기" },
// //     { title: "텍스트 분석", desc: "이력서를 위한 강력한 문구 생성" },
// //     { 
// //       title: "문제 해결", 
// //       desc: "OCR 및 LLM으로 이미지/PDF 분석하기",
// //       onClick: () => navigate("/ocr-tool") 
// //     },
// //   ];

// //   return (
// //     <div className="w-64 border-r bg-gray-50 p-4 absolute left-0 top-16 h-full shadow-lg">
// //       <h2 className="text-lg font-semibold mb-4">메뉴</h2>
// //       <div className="space-y-3">
// //         {quickPrompts.map((prompt, index) => (
// //           <div 
// //             key={index} 
// //             className="p-3 bg-white rounded-lg shadow-sm hover:shadow-md cursor-pointer"
// //             onClick={prompt.onClick}
// //           >
// //             <h3 className="font-medium text-sm">{prompt.title}</h3>
// //             <p className="text-xs text-gray-600 mt-1">{prompt.desc}</p>
// //           </div>
// //         ))}
// //       </div>
// //     </div>
// //   );
// // };

// // export default Sidebar;

// import React from "react";
// import { useNavigate } from "react-router-dom";

// const Sidebar = () => {
//   const navigate = useNavigate();
  
//   const quickPrompts = [
//     { 
//       title: "일정 관리", 
//       desc: "AI로 스마트한 일정 계획 및 관리하기",
//       onClick: () => navigate("/schedule-management") 
//     },
//     { title: "이미지 생성", desc: "취침 시간 이야기와 그림 만들기" },
//     { title: "텍스트 분석", desc: "이력서를 위한 강력한 문구 생성" },
//     { 
//       title: "문제 해결", 
//       desc: "OCR 및 LLM으로 이미지/PDF 분석하기",
//       onClick: () => navigate("/ocr-tool") 
//     },
//   ];

//   return (
//     <div className="w-64 border-r bg-gray-50 p-4 absolute left-0 top-16 h-full shadow-lg">
//       <h2 className="text-lg font-semibold mb-4">메뉴</h2>
//       <div className="space-y-3">
//         {quickPrompts.map((prompt, index) => (
//           <div 
//             key={index} 
//             className="p-3 bg-white rounded-lg shadow-sm hover:shadow-md cursor-pointer transition-all duration-200 hover:bg-blue-50"
//             onClick={prompt.onClick}
//           >
//             <h3 className="font-medium text-sm">{prompt.title}</h3>
//             <p className="text-xs text-gray-600 mt-1">{prompt.desc}</p>
//           </div>
//         ))}
//       </div>
//     </div>
//   );
// };

// export default Sidebar;

import React from "react";
import { useNavigate } from "react-router-dom";

const Sidebar = () => {
  const navigate = useNavigate();
  
  const quickPrompts = [
    { 
      title: "일정 관리", 
      desc: "AI로 스마트한 일정 계획 및 관리하기",
      onClick: () => navigate("/schedule-management") 
    },
    { 
      title: "비디오 분석", 
      desc: "AI로 비디오 내용 분석 및 검색하기",
      onClick: () => navigate("/video-analysis") 
    },
    { title: "이미지 생성", desc: "취침 시간 이야기와 그림 만들기" },
    { title: "텍스트 분석", desc: "이력서를 위한 강력한 문구 생성" },
    { 
      title: "문제 해결", 
      desc: "OCR 및 LLM으로 이미지/PDF 분석하기",
      onClick: () => navigate("/ocr-tool") 
    },
  ];

  return (
    <div className="w-64 border-r bg-gray-50 p-4 absolute left-0 top-16 h-full shadow-lg">
      <h2 className="text-lg font-semibold mb-4">메뉴</h2>
      <div className="space-y-3">
        {quickPrompts.map((prompt, index) => (
          <div 
            key={index} 
            className="p-3 bg-white rounded-lg shadow-sm hover:shadow-md cursor-pointer transition-all duration-200 hover:bg-blue-50"
            onClick={prompt.onClick}
          >
            <h3 className="font-medium text-sm">{prompt.title}</h3>
            <p className="text-xs text-gray-600 mt-1">{prompt.desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default Sidebar;