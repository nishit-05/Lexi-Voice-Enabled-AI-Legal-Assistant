import React from 'react'
import ReactDOM from 'react-dom/client'
import './index.css'
import LexiUI from './LexiUI'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <LexiUI apiEndpoint="http://127.0.0.1:5000/api/query" />
  </React.StrictMode>
)
