import streamlit as st
import requests
import json
import time
import os
import tempfile
from PIL import Image
import pandas as pd

# Set page configuration
st.set_page_config(
    page_title="Cricket Technique Analyzer",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API endpoint base URL
API_BASE_URL = "http://localhost:5000/api"

# Function to upload video and start analysis
def upload_and_analyze(file, player_type, additional_params=None):
    url = f"{API_BASE_URL}/{player_type}/upload"
    
    files = {
        'video': (file.name, file.getvalue(), 'video/mp4')
    }
    
    params = {}
    if additional_params:
        params.update(additional_params)
    
    try:
        response = requests.post(url, files=files, data=params)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error uploading video: {str(e)}")
        return None

# Function to check job status
def check_job_status(job_id, player_type):
    url = f"{API_BASE_URL}/{player_type}/status/{job_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error checking job status: {str(e)}")
        return None

# Function to get job results
def get_job_results(job_id, player_type):
    url = f"{API_BASE_URL}/{player_type}/results/{job_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting job results: {str(e)}")
        return None

# Function to get analyzed video and convert to web-compatible format
def get_analyzed_video(job_id, player_type):
    url = f"{API_BASE_URL}/{player_type}/video/{job_id}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Save video to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(response.content)
            input_path = tmp_file.name
        
        # Create output path for converted video
        output_path = input_path.replace('.mp4', '_converted.mp4')
        
        # Convert video to web-compatible format using ffmpeg
        import subprocess
        try:
            subprocess.run([
                'ffmpeg', 
                '-y',  # Overwrite output files without asking
                '-i', input_path,  # Input file
                '-c:v', 'libx264',  # Video codec
                '-c:a', 'aac',  # Audio codec
                '-movflags', '+faststart',  # Optimize for web playback
                output_path  # Output file
            ], check=True, capture_output=True)
            
            # Clean up the original temporary file
            try:
                os.unlink(input_path)
            except:
                pass
                
            return output_path
        except subprocess.CalledProcessError as e:
            st.warning(f"Video conversion failed: {e.stderr.decode() if e.stderr else str(e)}. Using original format.")
            return input_path
        except FileNotFoundError:
            st.warning("ffmpeg not found. Please install ffmpeg to enable video conversion. Using original format.")
            return input_path
            
    except requests.exceptions.RequestException as e:
        st.error(f"Error getting analyzed video: {str(e)}")
        return None

# Function to display metrics based on player type
def display_metrics(results, player_type):
    if 'metrics' not in results:
        st.warning("No metrics available in the results.")
        return
    
    metrics = results['metrics']
    
    # Common metrics display
    st.subheader("Key Metrics")
    cols = st.columns(3)
    
    if player_type == 'batsman':
        # Display batsman-specific metrics
        with cols[0]:
            if 'max_bat_speed' in metrics:
                st.metric("Max Bat Speed", f"{metrics['max_bat_speed']:.2f} {metrics.get('speed_unit', 'km/h')}")
            if 'max_ball_speed' in metrics:
                st.metric("Max Ball Speed", f"{metrics['max_ball_speed']:.2f} {metrics.get('speed_unit', 'km/h')}")
        
        with cols[1]:
            if 'impact_speed' in metrics and metrics['impact_speed'] is not None:
                st.metric("Impact Speed", f"{metrics['impact_speed']:.2f} {metrics.get('speed_unit', 'km/h')}")
            if 'ball_speed_after_impact' in metrics and metrics['ball_speed_after_impact'] is not None:
                st.metric("Ball Speed After Impact", f"{metrics['ball_speed_after_impact']:.2f} {metrics.get('speed_unit', 'km/h')}")
        
        with cols[2]:
            if 'swing_to_impact_time' in metrics and metrics['swing_to_impact_time'] is not None:
                st.metric("Swing to Impact Time", f"{metrics['swing_to_impact_time']:.3f} s")
            if 'video_length_seconds' in metrics:
                st.metric("Video Duration", f"{metrics['video_length_seconds']:.2f} s")
    
    # Create expandable section for all metrics
    with st.expander("View All Metrics"):
        # Convert metrics to DataFrame with proper string conversion to avoid Arrow errors
        metrics_items = [(k, str(v) if v is not None else "None") for k, v in metrics.items()]
        metrics_df = pd.DataFrame(metrics_items, columns=['Metric', 'Value'])
        st.dataframe(metrics_df, use_container_width=True)

# Function to display Gemini analysis
def display_gemini_analysis(results):
    if 'gemini_analysis' not in results:
        st.warning("No Gemini analysis available in the results.")
        return
    
    gemini_analysis = results['gemini_analysis']
    
    # Only display raw_response in JSON format
    if 'raw_response' in gemini_analysis:
        st.subheader("Gemini Analysis (Raw JSON Response)")
        
        # Extract clean JSON from the raw response by removing markdown code block markers
        raw_response = gemini_analysis['raw_response']
        import re
        
        # Look for JSON content inside code blocks
        json_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', raw_response, re.DOTALL)
        if json_match:
            # Extract just the JSON content without the markers
            clean_json = json_match.group(1).strip()
            st.code(clean_json, language="json")
        else:
            # If no code block markers found, just display the raw response as is
            st.code(raw_response, language="json")
    else:
        st.warning("No raw JSON response available from Gemini.")
        
        # As a fallback, display the existing gemini_analysis fields
        st.subheader("Gemini Analysis (Parsed Fields)")
        for key, value in gemini_analysis.items():
            if value is not None and value.strip() != "":
                # Format the key as a readable title
                title = " ".join(word.capitalize() for word in key.split('_'))
                st.write(f"**{title}**: {value}")
        st.info("Note: Raw JSON response not available. Showing parsed fields instead.")

# Main app
def main():
    # Add a title and introduction
    st.title("🏏 Cricket Technique Analyzer")
    st.markdown("""
    Upload cricket videos for AI-powered technique analysis. Get detailed metrics and personalized improvement suggestions.
    """)
    
    # Sidebar with options
    st.sidebar.header("Settings")
    player_type = st.sidebar.radio(
        "Player Type",
        options=["batsman", "bowler", "keeper"],
        format_func=lambda x: {"batsman": "Batsman", "bowler": "Bowler", "keeper": "Wicket Keeper"}[x]
    )
    
    # File uploader
    uploaded_file = st.sidebar.file_uploader("Upload Cricket Video", type=["mp4", "avi", "mov", "mkv"])
    
    # Additional parameters based on player type
    additional_params = {}
    
    if player_type == "batsman":
        pitch_length_pixels = st.sidebar.number_input(
            "Pitch Length (pixels, optional)", 
            min_value=100.0, 
            max_value=2000.0, 
            value=None,
            help="If known, provide the cricket pitch length in pixels for speed calibration"
        )
        if pitch_length_pixels:
            additional_params["pitch_length_pixels"] = pitch_length_pixels
    
    # Submit button
    submit_button = st.sidebar.button("Analyze Video", type="primary", disabled=uploaded_file is None)
    
    # Main content area
    if 'job_id' not in st.session_state:
        st.session_state.job_id = None
        st.session_state.player_type = None
        st.session_state.job_completed = False
        st.session_state.results = None
    
    # Process the uploaded file when the submit button is clicked
    if submit_button and uploaded_file is not None:
        with st.spinner("Uploading video..."):
            response = upload_and_analyze(uploaded_file, player_type, additional_params)
            
            if response and 'job_id' in response:
                st.session_state.job_id = response['job_id']
                st.session_state.player_type = player_type
                st.session_state.job_completed = False
                st.session_state.results = None
                st.success(f"Video uploaded successfully! Job ID: {response['job_id']}")
            else:
                st.error("Failed to upload and analyze video.")
    
    # Display progress if a job is in progress
    if st.session_state.job_id and not st.session_state.job_completed:
        job_id = st.session_state.job_id
        player_type = st.session_state.player_type
        
        status_container = st.container()
        
        with status_container:
            status_placeholder = st.empty()
            progress_bar = st.progress(0)
            message_placeholder = st.empty()
            
            # Poll for job status until completed
            while True:
                status = check_job_status(job_id, player_type)
                
                if not status:
                    status_placeholder.error("Failed to retrieve job status")
                    break
                
                progress = status.get('progress', 0)
                message = status.get('message', '')
                job_status = status.get('status', '')
                
                progress_bar.progress(progress / 100)
                status_placeholder.info(f"Status: {job_status.upper()}")
                message_placeholder.text(message)
                
                if job_status == 'completed':
                    # Get results
                    results = get_job_results(job_id, player_type)
                    if results:
                        st.session_state.results = results
                        st.session_state.job_completed = True
                        status_placeholder.success("Analysis completed successfully!")
                        progress_bar.progress(100)
                    else:
                        status_placeholder.error("Failed to retrieve results")
                    break
                elif job_status == 'failed':
                    status_placeholder.error(f"Analysis failed: {message}")
                    break
                
                # Poll every 2 seconds
                time.sleep(2)
    
    # Display results if job is completed
    if st.session_state.job_completed and st.session_state.results:
        results = st.session_state.results
        player_type = st.session_state.player_type
        job_id = st.session_state.job_id
        
        # Create tabs for different types of results
        tabs = st.tabs(["Analyzed Video", "Metrics", "Gemini Analysis"])
        
        # Tab 1: Analyzed Video
        with tabs[0]:
            st.subheader("Analyzed Video")
            video_path = get_analyzed_video(job_id, player_type)
            if video_path:
                video_file = open(video_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                
                # Download button for video
                st.download_button(
                    label="Download Analyzed Video",
                    data=video_bytes,
                    file_name=f"{player_type}_analyzed.mp4",
                    mime="video/mp4"
                )
                
                # Clean up temporary file
                video_file.close()
                try:
                    os.unlink(video_path)
                except:
                    pass
            else:
                st.error("Failed to retrieve analyzed video")
        
        # Tab 2: Metrics
        with tabs[1]:
            display_metrics(results, player_type)
        
        # Tab 3: Gemini Analysis
        with tabs[2]:
            display_gemini_analysis(results)
        
        # Option to start a new analysis
        if st.button("Start New Analysis", type="primary"):
            st.session_state.job_id = None
            st.session_state.player_type = None
            st.session_state.job_completed = False
            st.session_state.results = None
            st.rerun()

if __name__ == "__main__":
    main()