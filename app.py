from flask import Flask, request, jsonify, send_file
import os
import uuid
import time
from werkzeug.utils import secure_filename
import threading
import json

# Import your existing batsman analyzer - this stays unchanged
from batsman_analyzer import CricketShotAnalyzer
from gemini_analyzer import GeminiCricketAnalyzer

# Initialize Flask app
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
RESULTS_FOLDER = 'results'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}


# Model paths for each player type
BATSMAN_MODEL_PATH = 'models/batsman/best.pt'  # Using custom model - unchanged

# Enable/disable Gemini integration
ENABLE_GEMINI = os.environ.get('ENABLE_GEMINI', 'True').lower() in ('true', '1', 't')
os.environ['GEMINI_API_KEY'] = 'AIzaSyD8MuYmCZDZk9KGypJUbtdvZSR9hbw461A'
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Create necessary directories
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)
os.makedirs(os.path.join(RESULTS_FOLDER, 'videos'), exist_ok=True)
os.makedirs(os.path.join(RESULTS_FOLDER, 'data'), exist_ok=True)

# Store job status
job_status = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# Define specialized prompts for different player types
GEMINI_PROMPTS = {
    'batsman': """Analyze this cricket batting video with precise technical detail and provide the following comprehensive assessment:
1. Classify the exact type of cricket shot being played (e.g., cover drive, pull shot, flick).
2. Evaluate the timing quality (shot_timing: early,  good, late).
3. Estimate the approximate timing of ball and bat impacting after the bat movement started, being as precise as possible. 
   - For timestamps, use mili seconds from the start of the bat movement.
4. Provide detailed technical breakdown including:
   - Foot movement and positioning analysis (front foot vs back foot)
   - Head position relative to the ball
   - Bat path and angle at impact
   - Weight transfer mechanics
   - Balance throughout the shot
   - Follow-through completeness
5. Suggest specific, actionable improvements prioritized by impact:
   - Primary technical correction needed
   - Secondary technique refinements
   - Advanced optimization suggestions

Format your response as JSON with the following structure:
{
"shot_classification": string,
"ball_bat_impact_time": string,
"shot_timing": string,
"technical_analysis": string,
"suggested_improvements": string
}"""
}

# ==================== BATSMAN ANALYSIS ENDPOINTS ====================

def process_batsman_video_thread(video_path, job_id, output_path, pitch_length_pixels=None, progress_callback=None):
    """Background thread for batsman video processing with combined CV and Gemini analysis"""
    try:
        # Update status
        job_status[job_id] = {
            'status': 'processing',
            'progress': 0,
            'message': 'Starting combined batsman analysis...',
            'type': 'batsman',
            'cv_complete': False,
            'gemini_complete': False
        }
        
        # Progress callback function for CV analysis
        def update_cv_progress(progress, message):
            if job_id in job_status:
                job_status[job_id]['cv_progress'] = progress
                job_status[job_id]['cv_message'] = message
                # Update overall progress - weight CV as 60% of total progress
                _update_combined_progress(job_id)
        
        # Progress callback function for Gemini analysis
        def update_gemini_progress(progress, message):
            if job_id in job_status:
                job_status[job_id]['gemini_progress'] = progress
                job_status[job_id]['gemini_message'] = message
                # Update overall progress - weight Gemini as 40% of total progress
                _update_combined_progress(job_id)
        
        # Helper function to update combined progress
        def _update_combined_progress(job_id):
            cv_progress = job_status[job_id].get('cv_progress', 0)
            gemini_progress = job_status[job_id].get('gemini_progress', 0)
            # Weight CV as 60% and Gemini as 40% of total progress
            combined_progress = int((cv_progress * 0.6) + (gemini_progress * 0.4))
            job_status[job_id]['progress'] = combined_progress
            
            # Update message based on what's currently happening
            if cv_progress < 100 and gemini_progress < 100:
                job_status[job_id]['message'] = f"CV: {job_status[job_id].get('cv_message', '')} | Gemini: {job_status[job_id].get('gemini_message', '')}"
            elif cv_progress < 100:
                job_status[job_id]['message'] = f"CV: {job_status[job_id].get('cv_message', '')} | Gemini complete"
            elif gemini_progress < 100:
                job_status[job_id]['message'] = f"CV complete | Gemini: {job_status[job_id].get('gemini_message', '')}"
            else:
                job_status[job_id]['message'] = "Both analyses complete, finalizing results"
        
        # Initialize tracking for results
        cv_results = None
        gemini_results = None
        
        # Create a thread for Computer Vision analysis
        def run_cv_analysis():
            nonlocal cv_results
            try:
                # Initialize batsman analyzer with our custom model
                analyzer = CricketShotAnalyzer(
                    model_path=BATSMAN_MODEL_PATH,
                    pitch_length_pixels=pitch_length_pixels,
                    real_pitch_length=20.12  # Cricket pitch length in meters
                )
                
                # Run computer vision analysis
                cv_results = analyzer.process_video(
                    video_path=video_path,
                    output_path=output_path,
                    progress_callback=update_cv_progress
                )
                
                # Save metadata/results
                result_data_path = os.path.join(RESULTS_FOLDER, 'data', f"{job_id}_cv.json")
                analyzer.save_results(cv_results, result_data_path)
                
                # Mark CV analysis as complete
                if job_id in job_status:
                    job_status[job_id]['cv_complete'] = True
                    _check_completion()
                    
            except Exception as e:
                print(f"Error in CV analysis: {str(e)}")
                if job_id in job_status:
                    job_status[job_id]['cv_error'] = str(e)
                    job_status[job_id]['cv_complete'] = True
                    _check_completion()
        
        # Create a thread for Gemini analysis
        def run_gemini_analysis():
            nonlocal gemini_results
            try:
                # Check if Gemini is enabled
                if not ENABLE_GEMINI:
                    print("Gemini integration is disabled, skipping")
                    if job_id in job_status:
                        job_status[job_id]['gemini_complete'] = True
                        job_status[job_id]['gemini_skipped'] = True
                        _check_completion()
                    return
                
                # Check for API key
                if not GEMINI_API_KEY:
                    print("No Gemini API key provided, skipping")
                    if job_id in job_status:
                        job_status[job_id]['gemini_complete'] = True
                        job_status[job_id]['gemini_skipped'] = True
                        _check_completion()
                    return
                
                # Initialize Gemini analyzer
                gemini = GeminiCricketAnalyzer(api_key=GEMINI_API_KEY)
                
                # Use the batsman-specific prompt
                prompt = GEMINI_PROMPTS.get('batsman')
                
                # Prepare output path for Gemini results
                gemini_output_path = os.path.join(RESULTS_FOLDER, 'data', f"{job_id}_gemini.json")
                
                # Run Gemini analysis
                gemini_results = gemini.process_video(
                    video_path=video_path,
                    output_path=gemini_output_path,
                    custom_prompt=prompt,
                    progress_callback=update_gemini_progress
                )
                
                # Mark Gemini analysis as complete
                if job_id in job_status:
                    job_status[job_id]['gemini_complete'] = True
                    _check_completion()
                    
            except Exception as e:
                print(f"Error in Gemini analysis: {str(e)}")
                if job_id in job_status:
                    job_status[job_id]['gemini_error'] = str(e)
                    job_status[job_id]['gemini_complete'] = True
                    _check_completion()
        
        # Function to check if both analyses are complete and update status
        def _check_completion():
            if job_id in job_status:
                if job_status[job_id]['cv_complete'] and job_status[job_id]['gemini_complete']:
                    # Both analyses are complete, combine results
                    combined_results = _combine_results()
                    
                    # Save combined results
                    combined_path = os.path.join(RESULTS_FOLDER, 'data', f"{job_id}.json")
                    with open(combined_path, 'w') as f:
                        json.dump(combined_results, f, indent=4)
                    
                    # Update final status
                    job_status[job_id]['status'] = 'completed'
                    job_status[job_id]['progress'] = 100
                    job_status[job_id]['message'] = 'Combined batsman analysis complete'
                    job_status[job_id]['results'] = combined_results
        
        # Function to combine CV and Gemini results
        def _combine_results():
            combined = {}
            
            # Add CV metrics if available
            if cv_results:
                combined.update(cv_results)
            
            # Add Gemini analysis if available
            if gemini_results:
                combined['gemini_analysis'] = gemini_results.get('gemini_analysis', {})
            
            # Add any errors that occurred
            errors = {}
            if job_id in job_status:
                if 'cv_error' in job_status[job_id]:
                    errors['cv_error'] = job_status[job_id]['cv_error']
                if 'gemini_error' in job_status[job_id]:
                    errors['gemini_error'] = job_status[job_id]['gemini_error']
                if 'gemini_skipped' in job_status[job_id] and job_status[job_id]['gemini_skipped']:
                    errors['gemini_skipped'] = 'Gemini analysis was skipped (disabled or no API key)'
            
            if errors:
                combined['errors'] = errors
            
            return combined
        
        # Start both analyses in parallel threads
        import threading
        cv_thread = threading.Thread(target=run_cv_analysis)
        gemini_thread = threading.Thread(target=run_gemini_analysis)
        
        cv_thread.start()
        gemini_thread.start()
        
        # Don't wait for threads to complete here - they will update the job status as they progress
        
    except Exception as e:
        # Handle any exceptions in the main thread
        job_status[job_id] = {
            'status': 'failed',
            'progress': 0,
            'message': f"Error: {str(e)}",
            'type': 'batsman'
        }
        print(f"Error starting combined batsman analysis: {str(e)}")

@app.route('/api/batsman/upload', methods=['POST'])
def upload_batsman_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Generate unique ID for this job
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    video_filename = f"{timestamp}_{filename}"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    file.save(video_path)
    
    # Get optional pitch length parameter
    pitch_length_pixels = None
    if 'pitch_length_pixels' in request.form:
        try:
            pitch_length_pixels = float(request.form['pitch_length_pixels'])
        except ValueError:
            pass
    
    # Prepare output path
    output_filename = f"{job_id}_batsman_analyzed.mp4"
    output_path = os.path.join(RESULTS_FOLDER, 'videos', output_filename)
    
    # Start processing in background
    job_status[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Queued for batsman analysis',
        'type': 'batsman'
    }
    
    processing_thread = threading.Thread(
        target=process_batsman_video_thread,
        args=(video_path, job_id, output_path, pitch_length_pixels)
    )
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'message': 'Video uploaded and queued for batsman analysis',
        'type': 'batsman'
    }), 202

@app.route('/api/batsman/status/<job_id>', methods=['GET'])
def get_batsman_job_status(job_id):
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    if job_status[job_id].get('type') != 'batsman':
        return jsonify({'error': 'This is not a batsman analysis job'}), 400
    
    return jsonify(job_status[job_id]), 200

@app.route('/api/batsman/results/<job_id>', methods=['GET'])
def get_batsman_results(job_id):
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    if job_status[job_id].get('type') != 'batsman':
        return jsonify({'error': 'This is not a batsman analysis job'}), 400
    
    if job_status[job_id]['status'] != 'completed':
        return jsonify({'error': 'Analysis not complete', 'status': job_status[job_id]}), 400
    
    # Return analysis results
    return jsonify(job_status[job_id]['results']), 200

@app.route('/api/batsman/video/<job_id>', methods=['GET'])
def get_batsman_video(job_id):
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    if job_status[job_id].get('type') != 'batsman':
        return jsonify({'error': 'This is not a batsman analysis job'}), 400
    
    if job_status[job_id]['status'] != 'completed':
        return jsonify({'error': 'Analysis not complete', 'status': job_status[job_id]}), 400
    
    video_path = os.path.join(RESULTS_FOLDER, 'videos', f"{job_id}_batsman_analyzed.mp4")
    if not os.path.exists(video_path):
        return jsonify({'error': 'Video file not found'}), 404
    
    return send_file(video_path, mimetype='video/mp4', as_attachment=True)

@app.route('/api/batsman/metrics/<job_id>', methods=['GET'])
def get_batsman_metrics(job_id):
    """Get specific metrics from the batsman analysis"""
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    if job_status[job_id].get('type') != 'batsman':
        return jsonify({'error': 'This is not a batsman analysis job'}), 400
    
    if job_status[job_id]['status'] != 'completed':
        return jsonify({'error': 'Analysis not complete', 'status': job_status[job_id]}), 400
    
    # Return just the metrics subset of the results
    return jsonify(job_status[job_id]['results']['metrics']), 200

# ==================== GEMINI AI ANALYSIS ENDPOINTS ====================

def process_gemini_video_thread(video_path, job_id, output_path, player_type, custom_prompt=None):
    """Background thread for Gemini video processing with player type"""
    try:
        # Update status
        job_status[job_id] = {
            'status': 'processing',
            'progress': 0,
            'message': f'Starting Gemini {player_type} analysis...',
            'type': f'gemini_{player_type}'
        }
        
        # Progress callback function
        def update_progress(progress, message):
            job_status[job_id]['progress'] = progress
            job_status[job_id]['message'] = message
        
        # Check for API key
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API key not configured. Set GEMINI_API_KEY environment variable.")
        
        # Initialize Gemini analyzer
        analyzer = GeminiCricketAnalyzer(api_key=GEMINI_API_KEY)
        
        # Get the appropriate prompt for the player type
        prompt = custom_prompt or GEMINI_PROMPTS.get(player_type, GEMINI_PROMPTS['batsman'])
        
        # Run Gemini analysis
        results = analyzer.process_video(
            video_path=video_path,
            output_path=output_path,
            custom_prompt=prompt,
            progress_callback=update_progress
        )
        
        # Enhance results with player type
        results['player_type'] = player_type
        
        # Save metadata/results
        result_data_path = os.path.join(RESULTS_FOLDER, 'data', f"{job_id}.json")
        with open(result_data_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        # Update final status
        job_status[job_id] = {
            'status': 'completed',
            'progress': 100,
            'message': f'Gemini {player_type} analysis complete',
            'results': results,
            'type': f'gemini_{player_type}'
        }
        
    except Exception as e:
        job_status[job_id] = {
            'status': 'failed',
            'progress': 0,
            'message': f"Error: {str(e)}",
            'type': f'gemini_{player_type}'
        }
        print(f"Error processing {player_type} video with Gemini: {str(e)}")

# General Gemini analysis endpoint (original)
@app.route('/api/gemini/analyze', methods=['POST'])
def gemini_analyze():
    """Analyze video with Gemini AI using default batting analysis"""
    if not ENABLE_GEMINI:
        return jsonify({'error': 'Gemini integration is not enabled'}), 400
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Generate unique ID for this job
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    video_filename = f"{timestamp}_{filename}"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    file.save(video_path)
    
    # Get custom prompt if provided
    custom_prompt = request.form.get('prompt')
    
    # Prepare output path
    output_filename = f"{job_id}_gemini_analysis.json"
    output_path = os.path.join(RESULTS_FOLDER, 'data', output_filename)
    
    # Start processing in background
    job_status[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Queued for Gemini analysis',
        'type': 'gemini_batsman'
    }
    
    processing_thread = threading.Thread(
        target=process_gemini_video_thread,
        args=(video_path, job_id, output_path, 'batsman', custom_prompt)
    )
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'message': 'Video uploaded and queued for Gemini analysis',
        'type': 'gemini_batsman'
    }), 202

# Batsman-specific Gemini analysis
@app.route('/api/gemini/batsman/analyze', methods=['POST'])
def gemini_batsman_analyze():
    """Analyze batsman video with Gemini AI"""
    if not ENABLE_GEMINI:
        return jsonify({'error': 'Gemini integration is not enabled'}), 400
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'error': 'No video selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    # Generate unique ID for this job
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    video_filename = f"{timestamp}_{filename}"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    file.save(video_path)
    
    # Get custom prompt if provided (override default batsman prompt)
    custom_prompt = request.form.get('prompt')
    
    # Prepare output path
    output_filename = f"{job_id}_gemini_batsman_analysis.json"
    output_path = os.path.join(RESULTS_FOLDER, 'data', output_filename)
    
    # Start processing in background
    job_status[job_id] = {
        'status': 'queued',
        'progress': 0,
        'message': 'Queued for Gemini batsman analysis',
        'type': 'gemini_batsman'
    }
    
    processing_thread = threading.Thread(
        target=process_gemini_video_thread,
        args=(video_path, job_id, output_path, 'batsman', custom_prompt)
    )
    processing_thread.daemon = True
    processing_thread.start()
    
    return jsonify({
        'job_id': job_id,
        'status': 'queued',
        'message': 'Video uploaded and queued for Gemini batsman analysis',
        'type': 'gemini_batsman'
    }), 202

# Common status endpoint for all Gemini analyses
@app.route('/api/gemini/status/<job_id>', methods=['GET'])
def get_gemini_job_status(job_id):
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    if not job_status[job_id].get('type', '').startswith('gemini_'):
        return jsonify({'error': 'This is not a Gemini analysis job'}), 400
    
    return jsonify(job_status[job_id]), 200

# Common results endpoint for all Gemini analyses
@app.route('/api/gemini/results/<job_id>', methods=['GET'])
def get_gemini_results(job_id):
    """Get full Gemini analysis results including technical breakdown"""
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    if not job_status[job_id].get('type', '').startswith('gemini_'):
        return jsonify({'error': 'This is not a Gemini analysis job'}), 400
    
    if job_status[job_id]['status'] != 'completed':
        return jsonify({'error': 'Analysis not complete', 'status': job_status[job_id]}), 400
    
    # Check if we have results
    if 'results' not in job_status[job_id]:
        return jsonify({'error': 'No results available'}), 400
        
    # Return the Gemini analysis part
    gemini_analysis = job_status[job_id]['results'].get('gemini_analysis', {})
    return jsonify(gemini_analysis), 200

# Raw response endpoint for all Gemini analyses
@app.route('/api/gemini/raw/<job_id>', methods=['GET'])
def get_gemini_raw_response(job_id):
    """Get raw Gemini response text"""
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    if not job_status[job_id].get('type', '').startswith('gemini_'):
        return jsonify({'error': 'This is not a Gemini analysis job'}), 400
    
    if job_status[job_id]['status'] != 'completed':
        return jsonify({'error': 'Analysis not complete', 'status': job_status[job_id]}), 400
    
    # Check if we have results with Gemini analysis
    if 'results' not in job_status[job_id] or 'gemini_analysis' not in job_status[job_id]['results']:
        return jsonify({'error': 'No Gemini results available'}), 400
        
    # Return just the raw response
    raw_response = job_status[job_id]['results']['gemini_analysis'].get('raw_response', '')
    return jsonify({'raw_response': raw_response}), 200

# Specific player type results endpoints
@app.route('/api/gemini/batsman/results/<job_id>', methods=['GET'])
def get_gemini_batsman_results(job_id):
    """Get specific batsman Gemini analysis results"""
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    if job_status[job_id].get('type') != 'gemini_batsman':
        return jsonify({'error': 'This is not a Gemini batsman analysis job'}), 400
    
    # Forward to the common results endpoint
    return get_gemini_results(job_id)


# ==================== COMMON ENDPOINTS ====================

@app.route('/api/status/<job_id>', methods=['GET'])
def get_job_status(job_id):
    """Generic endpoint to check status for any job type"""
    if job_id not in job_status:
        return jsonify({'error': 'Job not found'}), 404
    
    return jsonify(job_status[job_id]), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)