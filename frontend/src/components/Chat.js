// components/Chat.js
import React from 'react';

const Chat = () => {
  const user = JSON.parse(localStorage.getItem('user') || '{}');

  return (
    <div className="p-4">
      <h1>채팅 페이지</h1>
      <p>환영합니다, {user.email}님!</p>
      {user.profile_image && (
        <img src={user.profile_image} alt="프로필" className="w-10 h-10 rounded-full" />
      )}
    </div>
  );
};

export default Chat;