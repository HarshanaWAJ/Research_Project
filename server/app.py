from flask import Flask, request, jsonify
import cv2
import numpy as np
from keras.models import load_model
import os
from flask_cors import CORS

# Suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0 = All, 1 = Warnings, 2 = Errors, 3 = None

app = Flask(__name__)
CORS(app)

# Load the pre-trained models
lbw_model = load_model('lbw_model.keras')  #  LBW model
no_ball_model = load_model('no_ball_model.keras')  #  No Ball model
wide_ball_model = load_model('wide_model.keras') # Wide Ball Model

def preprocess_video(video_path, frame_size=(224, 224), frame_rate=30):
    """
    Preprocess the input video to extract frames and resize them for the model.
    """
    cap = cv2.VideoCapture(video_path)

    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame to the size expected by the model (frame_size)
        frame_resized = cv2.resize(frame, frame_size)

        # Optionally normalize the frame (if required by the model)
        frame_normalized = frame_resized / 255.0  # Example for normalization

        frames.append(frame_normalized)

    cap.release()

    # Convert the list of frames to a numpy array
    frames = np.array(frames)

    return frames

# LBW classification endpoint
from flask import request, jsonify
import os
import numpy as np

@app.route('/classify-lbw', methods=['POST'])
def classify_lbw_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to a temporary directory
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    try:
        # Preprocess the video and predict using the LBW model
        video_frames = preprocess_video(file_path)
        prediction = lbw_model.predict(np.expand_dims(video_frames, axis=0))  # Assuming 'lbw_model' is used for LBW classification

        # If the model outputs probabilities, decide on the threshold for LBW classification
        threshold = 0.5
        lbw_out = prediction[0][1] > threshold  # Prediction for LBW out

        # Result based on prediction
        if lbw_out:
            result = {'prediction': 'LBW Out'}
        else:
            result = {'prediction': 'Not LBW Out'}

    except Exception as e:
        # If there's an error with classification (e.g., video processing or prediction failure)
        result = {'prediction': 'Not Out', 'error': str(e)}

    # Clean up the saved file after prediction
    os.remove(file_path)

    return jsonify(result)


# No Ball classification endpoint
@app.route('/classify-no-ball', methods=['POST'])
def classify_no_ball_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to a temporary directory
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Preprocess the video and predict using the No Ball model
    video_frames = preprocess_video(file_path)
    prediction = no_ball_model.predict(np.expand_dims(video_frames, axis=0))  # Use no_ball_model for no-ball prediction

    # If the model outputs probabilities, decide on the threshold for No Ball classification
    threshold = 0.5
    no_ball = prediction[0][0] > threshold

    result = {
        'prediction': 'No Ball' if no_ball else 'Legal Ball'
    }

    # Clean up the saved file after prediction
    os.remove(file_path)

    return jsonify(result)

""" ---------------------------  Predict Wide Ball ------------------------------------------------"""
def predict_ball_type(video_path, threshold=0.7):
    """
    Predicts whether the ball is a 'wide ball' or 'legal ball' from the given video.
    """
    frames = preprocess_video(video_path)

    if frames.shape[0] == 0:
        return "Cannot identify: No frames detected."

    # Reshaping the input frames to match the model's input shape
    frames = np.expand_dims(frames, axis=0)  # Add batch dimension (1, frame_count, height, width, channels)

    # Perform prediction on the frames
    predictions = wide_ball_model.predict(frames)

    # If you have binary classification or two classes
    class_probabilities = predictions[0]  # Assuming this is a softmax output
    class_labels = ['Legal Ball', 'Wide Ball']

    # Find the predicted class and probability
    predicted_class = np.argmax(class_probabilities)
    predicted_probability = class_probabilities[predicted_class]

    # Check if the predicted probability is above the threshold
    if predicted_probability >= threshold:
        return f" {class_labels[predicted_class]}"
    else:
        return "Cannot identify: Confidence too low."

""" ---------------------------  Predict Wide Ball ------------------------------------------------"""

@app.route('/classify-wide-ball', methods=['POST'])
def classify_wide_ball_video():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file to a temporary directory
    file_path = os.path.join('uploads', file.filename)
    file.save(file_path)

    # Use the predict_ball_type function for classification
    result = predict_ball_type(file_path, threshold=0.7)  # You can adjust the threshold if needed

    # Clean up the saved file after prediction
    os.remove(file_path)

    return jsonify({'result': result})

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=False, host='0.0.0.0', port=5000)
