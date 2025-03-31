// src/axiosInstance.js
import axios from 'axios';

// Create an axios instance with the base URL and port
const axiosInstance = axios.create({
  baseURL: 'http://localhost:5000',  // Replace with your backend URL and port
});

export default axiosInstance;
