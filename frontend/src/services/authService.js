// // src/services/authService.js
// const authService = {
//     // 소셜 로그인 초기화
//     initializeSocialLogin: async (provider) => {
//       try {
//         const response = await fetch(`http://localhost:8000/api/auth/${provider}/initialize/`, {
//           method: 'GET',
//           credentials: 'include',
//         });
        
//         if (!response.ok) throw new Error('초기화 실패');
        
//         return await response.json();
//       } catch (error) {
//         console.error('소셜 로그인 초기화 실패:', error);
//         throw error;
//       }
//     },
  
//     // 토큰 교환
//     exchangeCodeForToken: async (provider, code, state = null) => {
//       try {
//         const response = await fetch(`http://localhost:8000/api/auth/${provider}/callback/`, {
//           method: 'POST',
//           headers: {
//             'Content-Type': 'application/json',
//           },
//           credentials: 'include',
//           body: JSON.stringify({ code, state })
//         });
  
//         if (!response.ok) throw new Error('토큰 교환 실패');
        
//         return await response.json();
//       } catch (error) {
//         console.error('토큰 교환 실패:', error);
//         throw error;
//       }
//     }
//   };
  
//   export default authService;
// src/services/authService.js
const authService = {
  initializeSocialLogin: async (provider) => {
    try {
      const response = await fetch(`http://localhost:8000/api/auth/${provider}/initialize/`, {
        method: 'GET',
        credentials: 'include',
      });
      if (!response.ok) throw new Error('초기화 실패');
      return await response.json();
    } catch (error) {
      console.error('소셜 로그인 초기화 실패:', error);
      throw error;
    }
  },

  exchangeCodeForToken: async (provider, code, state = null) => {
    try {
      const response = await fetch(`http://localhost:8000/api/auth/${provider}/callback/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        credentials: 'include',
        body: JSON.stringify({ code, state })
      });
      if (!response.ok) throw new Error('토큰 교환 실패');
      return await response.json();
    } catch (error) {
      console.error('토큰 교환 실패:', error);
      throw error;
    }
  }
};

export default authService;
