import React, { useState } from 'react'; // Import useState
import { toast, ToastContainer } from 'react-toastify'; // Import toast and ToastContainer for error messages
import 'react-toastify/dist/ReactToastify.css'; // Import Toastify CSS
import '../styles/WideBallClassification.css';
import axiosInstance from '../axiosInstance'; // Import axios instance

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
    const response = await axiosInstance.post('/classify-wide-ball', formData, {
      headers: {
        'Content-Type': 'multipart/form-data', 
      },
    });

    setIsLoading(false);

    // Dismiss all toast notifications before showing the result
    toast.dismiss();

    if (response.status === 200) {
      const classificationResult = response.data.result;
      setResult(classificationResult); // Set the result in the state
    } else {
      toast.error(`Error: ${response.data.error || 'Something went wrong'}`);
    }
  } catch (error) {
    setIsLoading(false);
    toast.dismiss(); // Dismiss any error toast before showing the failure message
    toast.error('Upload failed, please try again.');
  }
};

function WideClassification() {
  const [errorMessage, setErrorMessage] = useState(''); // Define state for error messages
  const [isLoading, setIsLoading] = useState(false); // State to handle loading status
  const [file, setFile] = useState(null); // Store the uploaded file
  const [result, setResult] = useState(null); // Store the result from the API response

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
            disabled={isLoading || !file}>
            {isLoading ? 'Uploading...' : 'Upload'}
          </button>
          <div className="instructions">
            <h4>Instructions:</h4>
            <p>Please upload MP4 short videos (max time 30 sec) only.</p>
          </div>
        </div>
        <div className="right-section">
          {/* Right section content */}
          {result ? (
            <div className="result">
              <h4>Result:</h4>
              <h5>{result}</h5> {/* Display only the classification result */}
            </div>
          ) : (
            <p>No result yet. Please upload a video to classify.</p>
          )}
        </div>
      </div>
      <div className="back ml-5">
        <a href="/">Back to Umpire Assistant</a>
      </div>
      {/* ToastContainer to display the toasts */}
      <ToastContainer />
    </div>
  );
}

export default WideClassification;
