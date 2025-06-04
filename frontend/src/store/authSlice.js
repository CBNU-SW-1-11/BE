// authSlice.js
import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

export const saveUserSettings = createAsyncThunk(
  'auth/saveUserSettings',
  async (settings, { getState }) => {
    try {
      const token = localStorage.getItem('accessToken');
      if (!token) {
        throw new Error('No authentication token found');
      }

      const response = await fetch("http://localhost:8000/api/user/settings/", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Token ${token}`,
        },
        body: JSON.stringify(settings),
        credentials: 'include',
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to save settings');
      }

      const data = await response.json();
      return data;
    } catch (error) {
      console.error('Error saving settings:', error);
      throw error;
    }
  }
);

// const authSlice = createSlice({
//   name: 'auth',
//   initialState: {
//     user: null,
//     token: null,
//     isAuthenticated: false,
//     error: null,
//     settings: null
//   },
//   reducers: {
//     loginSuccess: (state, action) => {
//       state.user = action.payload.user;
//       state.token = action.payload.token;
//       state.isAuthenticated = true;
//       state.error = null;
//       localStorage.setItem('accessToken', action.payload.token);
//     },
//     loginFailure: (state, action) => {
//       state.user = null;
//       state.token = null;
//       state.isAuthenticated = false;
//       state.error = action.payload;
//       localStorage.removeItem('accessToken');
//     },
//     updateUserSettings: (state, action) => {
//       if (state.user) {
//         state.user.settings = {
//           ...state.user.settings,
//           ...action.payload
//         };
//       }
//     },
//     logoutSuccess: (state) => {
//       state.user = null;
//       state.token = null;
//       state.isAuthenticated = false;
//       state.error = null;
//       localStorage.removeItem('accessToken');
//     }
//   },
  // extraReducers: (builder) => {
  //   builder
  //     .addCase(saveUserSettings.fulfilled, (state, action) => {
  //       if (state.user) {
  //         state.user.settings = action.payload.settings;
  //       }
  //     })
  //     .addCase(saveUserSettings.rejected, (state, action) => {
  //       state.error = action.error.message;
  //     });
  // }
// });

// export const { 
//   loginSuccess, 
//   loginFailure, 
//   updateUserSettings, 
//   logoutSuccess 
// } = authSlice.actions;

// export default authSlice.reducer;
// src/store/authSlice.js

const initialState = {
  user: null,
  token: null,
  error: null,
};

const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    loginSuccess: (state, action) => {
      state.user = action.payload.user;
      state.token = action.payload.token;
      state.error = null;
    },
    loginFailure: (state, action) => {
      state.user = null;
      state.token = null;
      state.error = action.payload;
    },
    logout: (state) => {
      state.user = null;
      state.token = null;
      state.error = null;
    },
    updateUserSettings: (state, action) => {
            if (state.user) {
              state.user.settings = {
                ...state.user.settings,
                ...action.payload
              };
            }
          },
    // 추가 액션이 필요한 경우 여기에 작성하세요.
  },
});

export const { loginSuccess, updateUserSettings, loginFailure, logout } = authSlice.actions;
export default authSlice.reducer;
// export const { 
//   loginSuccess, 
//   loginFailure, 
//   updateUserSettings, 
//   logoutSuccess 
// } = authSlice.actions;
