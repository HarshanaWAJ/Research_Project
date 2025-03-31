import React, { useState } from 'react';
import '../styles/LbwClassification.css';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import axiosInstance from '../axiosInstance';  // Import the axios instance

function LbwClassification() {
  const [errorMessage, setErrorMessage] = useState('');
  const [prediction, setPrediction] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.type !== 'video/mp4') {
        const message = 'Please upload an MP4 video file.';
        setErrorMessage(message);
        toast.error(message); // Show error toast
        e.target.value = ''; // Reset file input
        console.log(errorMessage);
        
        return;
      }

      const videoElement = document.createElement('video');
      videoElement.src = URL.createObjectURL(file);

      videoElement.onloadedmetadata = () => {
        if (videoElement.duration > 30) {
          const message = 'Video duration should be less than or equal to 30 seconds.';
          setErrorMessage(message);
          toast.error(message); // Show error toast
          e.target.value = ''; // Reset file input
        } else {
          setErrorMessage(''); // Clear any existing error messages if valid
        }
      };
    }
  };

  const handleUploadClick = async () => {
    const fileInput = document.getElementById('file-upload');
    const file = fileInput.files[0];

    if (!file) {
      const message = 'Please select a video file.';
      setErrorMessage(message);
      toast.error(message); // Show error toast
      return;
    }

    // Reset previous prediction and error message
    setPrediction('');
    setErrorMessage('');
    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      // Use axios instance to make the POST request
      const response = await axiosInstance.post('/classify-lbw', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      // Handle the response from backend
      if (response.data.prediction) {
        setPrediction(response.data.prediction);
      }
    } catch (error) {
      console.error('Error uploading file:', error);
      toast.error('Error uploading file.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="lbw-classification">
      <ToastContainer />

      <div className="heading m-2 p-2">
        Leg By Wicket Classification
      </div>

      <div className="main m-5">
        <div className="left-section">
          <div className="upload-container">
            <label htmlFor="file-upload">
              <p>Click here to upload your MP4 video (max 30 sec)</p>
            </label>
            <input
              type="file"
              id="file-upload"
              onChange={handleFileChange}
              accept="video/mp4"
            />
          </div>
          <button className="upload-btn" onClick={handleUploadClick} disabled={isLoading}>
            {isLoading ? 'Uploading...' : 'Upload'}
          </button>
          <div className="instructions">
            <h4>Instructions:</h4>
            <p>Please upload MP4 short videos (max time 30 sec) only.</p>
          </div>
        </div>

        <div className="right-section">
          <div className="result">
            <p>Result: {prediction ? prediction : 'No result yet'}</p>
          </div>
        </div>
      </div>

      <div className="back ml-5">
        <a href="/">Back to Umpire Assistant</a>
      </div>
    </div>
  );
}

export default LbwClassification;
