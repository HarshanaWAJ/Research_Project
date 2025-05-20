import React, { useState, useRef } from 'react';
import '../styles/LbwClassification.css';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import axiosInstance from '../axiosInstance';

function LbwClassification() {
  const [errorMessage, setErrorMessage] = useState('');
  const [prediction, setPrediction] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [videoURL, setVideoURL] = useState(null); // 👈 New state for video URL
  const videoRef = useRef(null); // 👈 Ref to access video DOM

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      if (file.type !== 'video/mp4') {
        const message = 'Please upload an MP4 video file.';
        setErrorMessage(message);
        toast.error(message);
        e.target.value = '';
        return;
      }

      const videoElement = document.createElement('video');
      videoElement.src = URL.createObjectURL(file);

      videoElement.onloadedmetadata = () => {
        if (videoElement.duration > 30) {
          const message = 'Video duration should be less than or equal to 30 seconds.';
          setErrorMessage(message);
          toast.error(message);
          e.target.value = '';
        } else {
          setErrorMessage('');
          setVideoURL(videoElement.src); // 👈 Set the video URL for playback
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
      toast.error(message);
      return;
    }

    setPrediction('');
    setErrorMessage('');
    setIsLoading(true);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await axiosInstance.post('/classify-lbw', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

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

  // Video control handlers
  const handlePlay = () => {
    videoRef.current.play();
  };

  const handlePause = () => {
    videoRef.current.pause();
  };

  const handleSlowDown = () => {
    if (videoRef.current.playbackRate > 0.25) {
      videoRef.current.playbackRate -= 0.25;
    }
  };

  const handleSpeedReset = () => {
    videoRef.current.playbackRate = 1.0;
  };

  return (
    <div className="lbw-classification">
      <ToastContainer />

      <div className="heading m-2 p-2">Leg By Wicket Classification</div>

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

          {/* 👇 Video player with controls */}
          {videoURL && (
            <div className="video-player">
              <video ref={videoRef} src={videoURL} controls width="400" />
              <div className="video-controls">
                <button onClick={handlePlay}>Play</button>
                <button onClick={handlePause}>Pause</button>
                <button onClick={handleSlowDown}>Slow Down</button>
                <button onClick={handleSpeedReset}>Reset Speed</button>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="back ml-5">
        <a href="/">Back to Umpire Assistant</a>
      </div>
    </div>
  );
}

export default LbwClassification;
