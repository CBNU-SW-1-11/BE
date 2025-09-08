import React from "react";
import { useNavigate } from "react-router-dom";

const Sidebar = () => {
  const navigate = useNavigate();

  // ✅ 2번 기능 그대로 유지 (라우팅 onClick)
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
  <div
  className="fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 z-50 border-r p-4 flex-shrink-0 transition-all duration-300 shadow-lg"
  style={{
    background: 'rgba(245, 242, 234, 0.4)',
    backdropFilter: 'blur(20px)',
    WebkitBackdropFilter: 'blur(20px)',
    borderRightColor: 'rgba(139, 168, 138, 0.15)',
  }}
>
      <h2 
        className="text-lg font-semibold mb-4"
        style={{
          color: '#2d3e2c',
          fontSize: '1.2rem',
          fontWeight: 600
        }}
      >
        메뉴
      </h2>

      <div className="space-y-3">
        {quickPrompts.map((prompt, index) => (
          <div 
            key={index}
            // ✅ 1번 스타일의 카드 + 샤인 효과 클래스
            className="sg-sidebar-item p-3 rounded-lg cursor-pointer transition-all duration-400 relative overflow-hidden"
            style={{
              background: 'rgba(255, 255, 255, 0.8)',
              backdropFilter: 'blur(10px)',
              WebkitBackdropFilter: 'blur(10px)',
              border: '1px solid rgba(139, 168, 138, 0.2)',
              borderRadius: '16px'
            }}
            onClick={prompt.onClick} // 2번 기능 유지
            role={prompt.onClick ? "button" : undefined}
            tabIndex={prompt.onClick ? 0 : -1}
            onKeyDown={(e) => {
              if (prompt.onClick && (e.key === 'Enter' || e.key === ' ')) {
                e.preventDefault();
                prompt.onClick();
              }
            }}
          >
            {/* 반짝이는 애니메이션 효과 */}
            <div 
              className="sg-shine-effect absolute top-0 left-0 w-full h-full pointer-events-none"
              style={{
                background: 'linear-gradient(90deg, transparent, rgba(139, 168, 138, 0.1), transparent)',
                transform: 'translateX(-100%)',
                transition: 'transform 0.6s ease'
              }}
            />

            <h3 
              className="font-medium text-sm mb-1 transition-colors duration-300"
              style={{
                color: '#5d7c5b',
                fontSize: '0.95rem',
                fontWeight: 600
              }}
            >
              {prompt.title}
            </h3>
            <p 
              className="text-xs mt-1 transition-colors duration-300"
              style={{
                color: 'rgba(45, 62, 44, 0.5)',
                fontSize: '0.8rem',
                lineHeight: 1.4
              }}
            >
              {prompt.desc}
            </p>
          </div>
        ))}
      </div>

      {/* ⚠️ CRA/Vite 환경 호환: styled-jsx 대신 일반 <style> 사용. 클래스 prefix로 범위 최소화 */}
      <style>{`
        .sg-sidebar-item:hover {
          background: rgba(255, 255, 255, 0.9) !important;
          border-color: #8ba88a !important;
          transform: translateY(-4px) scale(1.02);
          box-shadow: 0 12px 40px rgba(93, 124, 91, 0.15);
        }

        .sg-sidebar-item:hover .sg-shine-effect {
          transform: translateX(100%);
        }

        .sg-sidebar-item:hover h3 {
          color: #5d7c5b !important;
        }

        .sg-sidebar-item:hover p {
          color: rgba(45, 62, 44, 0.7) !important;
        }

        .sg-sidebar-item {
          position: relative;
        }

        .sg-sidebar-item::before {
          content: '';
          position: absolute;
          top: 0;
          left: -100%;
          width: 100%;
          height: 100%;
          background: linear-gradient(90deg, transparent, rgba(139, 168, 138, 0.1), transparent);
          transition: left 0.6s;
          pointer-events: none;
        }

        .sg-sidebar-item:hover::before {
          left: 100%;
        }
      `}</style>
    </div>
  );
};

export default Sidebar;
