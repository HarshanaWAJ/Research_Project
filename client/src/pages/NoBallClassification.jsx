import React, { useState } from 'react';
import { toast, ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import '../styles/NoBallClassification.css';
import axiosInstance from '../axiosInstance';

const handleFileChange = (e, setErrorMessage) => {
  const file = e.target.files[0];
  if (file) {
    if (file.type !== 'video/mp4') {
      const message = 'Please upload an MP4 video file.';
      setErrorMessage(message);
      toast.error(message); // Show error toast
      e.target.value = ''; // Reset file input
      return;
    }

    const videoElement = document.createElement('video');
    videoElement.src = URL.createObjectURL(file);

    // Check video duration once the metadata is loaded
    videoElement.onloadedmetadata = () => {
      if (videoElement.duration > 30) {
        const message = 'Video duration should be less than or equal to 30 seconds.';
        setErrorMessage(message);
        toast.error(message); // Show error toast
        e.target.value = ''; // Reset file input
      } else {
        setErrorMessage(''); // Clear error messages if valid
      }
    };
  }
};

const handleUploadClick = async (file, setIsLoading, setResult) => {
  if (!file) {
    const message = 'Please upload a valid video file first.';
    toast.error(message); // Show error toast
    return;
  }

  const formData = new FormData();
  formData.append('file', file);

  setIsLoading(true);

  try {
    const response = await axiosInstance.post('/classify-no-ball', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    setIsLoading(false);
    toast.dismiss();

    if (response.status === 200) {
      const classificationResult = response.data.prediction; // Assuming the server returns 'prediction'
      setResult(classificationResult); // Set the result in state
    } else {
      toast.error(`Error: ${response.data.error || 'Something went wrong'}`);
    }
  } catch (error) {
    setIsLoading(false);
    toast.dismiss();
    toast.error('Upload failed, please try again.');
  }
};

function NoBallClassification() {
  const [errorMessage, setErrorMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);

  return (
    <div className="no-ball-classification">
      <div className="heading m-2 p-2">No Ball Classification</div>
      <div className="main m-5">
        <div className="left-section">
          <div className="upload-container">
            <label htmlFor="file-upload">
              <p>Click here to upload your MP4 video (max 30 sec)</p>
            </label>
            <input
              type="file"
              id="file-upload"
              onChange={(e) => {
                handleFileChange(e, setErrorMessage);
                setFile(e.target.files[0]); // Store file in state
              }}
              accept="video/mp4"
            />
            {errorMessage && <div className="error-message">{errorMessage}</div>}
          </div>
          <button
            className="upload-btn"
            onClick={() => handleUploadClick(file, setIsLoading, setResult)}
            disabled={isLoading || !file}
          >
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
              <h4>Result: </h4>
              <h4> {result}</h4>
            </div>
          ) : (
            <p>No result yet. Please upload a video to classify.</p>
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

export default NoBallClassification;
