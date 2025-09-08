// components/ChatMessage.jsx
import React from 'react';

export default function ChatMessage({ message, onSeek }) {
  const items = message.items || [];
  return (
    <div className="chat-msg">
      <div className="whitespace-pre-wrap text-sm leading-6">{message.response}</div>

      {items.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3 mt-3">
          {items.map((it, i) => (
            <div key={i} className="border rounded-lg p-2">
              {it.thumbUrl && (
                <a href={it.clipUrl || it.thumbUrl} target="_blank" rel="noreferrer">
                  <img src={it.thumbUrl} alt={it.desc || 'frame'} className="w-full h-auto rounded" />
                </a>
              )}
              <div className="text-xs mt-1">
                <div className="font-medium">{it.time} · {it.desc}</div>
                {typeof it.score === 'number' && (
                  <div className="text-gray-500">~{Math.round(it.score * 100)}%</div>
                )}
                <div className="flex gap-2 mt-1">
                  {it.clipUrl && (
                    <a className="text-blue-600 underline" href={it.clipUrl} target="_blank" rel="noreferrer">
                      클립 보기
                    </a>
                  )}
                  {typeof onSeek === 'function' && (
                    <button className="underline" onClick={() => onSeek?.(it.seconds)}>
                      해당 시점으로
                    </button>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
