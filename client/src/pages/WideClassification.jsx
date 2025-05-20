import React, { useState, useRef } from 'react'; // Added useRef
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import '../styles/WideBallClassification.css';
import axiosInstance from '../axiosInstance';

const handleFileChange = (e, setErrorMessage, setVideoURL) => {
  const file = e.target.files[0];
  if (file) {
    if (file.type !== 'video/mp4') {
      const message = 'Please upload an MP4 video file.';
      setErrorMessage(message);
      toast.error(message);
      e.target.value = '';
      setVideoURL(null);
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
        setVideoURL(null);
      } else {
        setErrorMessage('');
        setVideoURL(videoElement.src);
      }
    };
  } else {
    setVideoURL(null);
  }
};

const handleUploadClick = async (file, setIsLoading, setResult) => {
  if (!file) {
    const message = 'Please upload a valid video file first.';
    toast.error(message);
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  setIsLoading(true);

  try {
    const response = await axiosInstance.post('/classify-wide-ball', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    setIsLoading(false);
    toast.dismiss();

    if (response.status === 200) {
      const classificationResult = response.data.result;
      setResult(classificationResult);
    } else {
      toast.error(`Error: ${response.data.error || 'Something went wrong'}`);
    }
  } catch (error) {
    setIsLoading(false);
    toast.dismiss();
    toast.error('Upload failed, please try again.');
  }
};

function WideClassification() {
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [videoURL, setVideoURL] = useState(null); // 👈 New state
  const videoRef = useRef(null); // 👈 Ref for video control

  return (
    <div className="wide-ball-classification">
      <div className="heading m-2 p-2">
        Wide Ball Classification
      </div>
      <div className="main m-5 p-5">
        <div className="left-section">
          <div className="upload-container">
            <label htmlFor="file-upload">
              <p>Click here to upload your MP4 video (max 30 sec)</p>
            </label>
            <input
              type="file"
              id="file-upload"
              onChange={(e) => {
                handleFileChange(e, setErrorMessage, setVideoURL);
                setFile(e.target.files[0]);
              }}
              accept="video/mp4"
            />
            {errorMessage && <div className="error-message">{errorMessage}</div>}
          </div>
          <button 
            className="upload-btn" 
            onClick={() => handleUploadClick(file, setIsLoading, setResult)} 
            disabled={isLoading || !file}>
            {isLoading ? 'Uploading...' : 'Upload'}
          </button>
          <div className="instructions">
            <h4>Instructions:</h4>
            <p>Please upload MP4 short videos (max time 30 sec) only.</p>
          </div>
        </div>
        <div className="right-section">
          {result ? (
            <div className="result">
              <h4>Result:</h4>
              <h5>{result}</h5>
            </div>
          ) : (
            <p>No result yet. Please upload a video to classify.</p>
          )}

          {/* 👇 Video preview and controls */}
          {videoURL && (
            <div className="video-player">
              <video ref={videoRef} src={videoURL} controls width="400" />
              <div className="video-controls">
                <button onClick={() => videoRef.current.play()}>Play</button>
                <button onClick={() => videoRef.current.pause()}>Pause</button>
                <button
                  onClick={() => {
                    if (videoRef.current.playbackRate > 0.25) {
                      videoRef.current.playbackRate -= 0.25;
                    }
                  }}
                >
                  Slow Down
                </button>
                <button onClick={() => (videoRef.current.playbackRate = 1)}>
                  Reset Speed
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
      <div className="back ml-5">
        <a href="/">Back to Umpire Assistant</a>
      </div>
      <ToastContainer />
    </div>
  );
}

export default WideClassification;
