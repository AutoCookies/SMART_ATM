import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import './App.css';
import Login from './pages/Login';
import Main from './pages/Main';
import UserManagement from './component/UserManagement';
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />}/>
        <Route path="/main" element={<Main />} />
        <Route path="/users" element={<UserManagement />} />
      </Routes>
    </Router>
  );
}

export default App;
