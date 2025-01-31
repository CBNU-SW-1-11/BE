// src/App.js
import React from 'react';
import { Provider } from 'react-redux';
import { GoogleOAuthProvider } from '@react-oauth/google';
import { BrowserRouter as Router } from 'react-router-dom';
import  Settingbar  from './components/Settingbar';
import { store }  from './store';

function App() {
  return (
    <Provider store={store}>
      <GoogleOAuthProvider clientId={process.env.REACT_APP_GOOGLE_CLIENT_ID}>
        <Router>
          <div className="App">
            <Settingbar />
          </div>
        </Router>
      </GoogleOAuthProvider>
    </Provider>
  );
}

export default App;