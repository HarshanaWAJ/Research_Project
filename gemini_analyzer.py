import os
import json
import time
import re
from google import genai
from google.genai import types

class GeminiCricketAnalyzer:
    """Cricket video analysis using Google Gemini multimodal capabilities"""
    
    def __init__(self, api_key=None):
        """
        Initialize the Gemini analyzer
        
        Args:
            api_key (str, optional): Gemini API key. If not provided, tries to get from env var.
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Gemini API key must be provided or set as GEMINI_API_KEY environment variable")
        
        # Initialize Gemini client
        self.client = genai.Client(api_key=self.api_key)
        
        # Default Gemini model
        self.model_name = "gemini-2.0-flash-thinking-exp-01-21"
        
        # Default prompts for analysis
        self.standard_prompts = {
            "shot_classification": """Analyze this cricket batting video and provide following details:
1. Classify the type of shot being played.
2. Provide a detailed technical analysis of the shot.
3. Suggest improvements for this player's execution of the shot.

Format your response as JSON with the following structure:
{
"shot_classification": string,
"technical_analysis": string,
"suggested_improvements": string
}"""
        }
    
    def upload_video(self, video_path):
        """
        Upload a video file to Gemini
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            File object with URI
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        max_retries = 3
        retry_delay = 2  # seconds
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"Retrying file upload (attempt {attempt+1}/{max_retries})...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                
                uploaded_file = self.client.files.upload(file=video_path)
                
                # Add a brief delay after successful upload to allow Gemini to process
                time.sleep(1)
                
                return uploaded_file
            except Exception as e:
                last_error = e
                # If it's not a temporary error, don't retry
                if "timeout" not in str(e).lower() and "rate limit" not in str(e).lower():
                    break
        
        # If we get here, all retries failed
        raise Exception(f"Failed to upload video to Gemini after {max_retries} attempts: {str(last_error)}")
    
    def analyze_video(self, video_path, analysis_type="shot_classification", custom_prompt=None):
        """
        Analyze cricket video using Gemini
        
        Args:
            video_path (str): Path to the video file
            analysis_type (str): Type of analysis to perform
            custom_prompt (str, optional): Custom prompt to use instead of defaults
            
        Returns:
            dict: Structured analysis results
        """
        try:
            # Upload the video file
            uploaded_file = self.upload_video(video_path)
            
            # Get the appropriate prompt
            prompt = custom_prompt or self.standard_prompts.get(analysis_type, self.standard_prompts["shot_classification"])
            
            # Prepare the content for Gemini
            contents = [
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_uri(
                            file_uri=uploaded_file.uri,
                            mime_type=uploaded_file.mime_type,
                        ),
                    ],
                ),
                types.Content(
                    role="user", 
                    parts=[types.Part.from_text(text=prompt)]
                )
            ]
            
            # Configure the generation parameters
            generate_content_config = types.GenerateContentConfig(
                temperature=0.2,  # Lower temperature for more deterministic/analytical results
                top_p=0.95,
                top_k=64,
                max_output_tokens=8192,
            )
            
            # Generate content from Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=generate_content_config,
            )
            
            # Parse the response - handle both JSON and text
            result_text = response.text
            
            try:
                # Try to parse as JSON first
                return json.loads(result_text)
            except json.JSONDecodeError:
                # If not valid JSON, extract structured information from text
                return self._parse_text_response(result_text)
                
        except Exception as e:
            return {"error": f"Gemini analysis failed: {str(e)}"}
    
    def _parse_text_response(self, text):
        """
        Parse text response into structured data with improved code block handling
        
        Args:
            text (str): Text response from Gemini
            
        Returns:
            dict: Structured data extracted from text
        """
        import json
        import re
        
        # Initialize result structure with raw response
        result = {
            "raw_response": text
        }

        
        # Check if response contains a code block with JSON
        json_block_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL)
        if json_block_match:
            # Extract JSON from code block
            json_str = json_block_match.group(1).strip()
            try:
                # Parse the JSON
                parsed_json = json.loads(json_str)

                # Add any additional fields that might be in the response
                for key, value in parsed_json.items():
                    result[key] = value
                        
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, fall back to regex extraction
                print(f"Failed to parse JSON from Gemini response: {e}")
        
        # If we couldn't parse the JSON block or there wasn't one,
        # fall back to the existing regex extraction method
        
        # Extract shot classification
        shot_match = re.search(r"Shot (?:Type|Classification):\s*\*?\*?(.*?)\*?\*?(?:\n|$)", text, re.IGNORECASE)
        if shot_match:
            result["shot_classification"] = shot_match.group(1).strip()
        
        # Extract technical analysis
        technical_section = re.search(r"(?:Technical Analysis|Key Observations|Characteristics|Technical Breakdown):\s*\*?\*?(.*?)(?:\n\n|\n(?:[A-Z]|$)|\Z)", text, re.DOTALL | re.IGNORECASE)
        if technical_section:
            result["technical_analysis"] = technical_section.group(1).strip()
        
        # Extract suggested improvements
        improvements_section = re.search(r"(?:Suggested Improvements|Improvements|Areas for Improvement|Could Improve):\s*\*?\*?(.*?)(?:\n\n|\Z)", text, re.DOTALL | re.IGNORECASE)
        if improvements_section:
            result["suggested_improvements"] = improvements_section.group(1).strip()
        
        return result
    
    def save_results(self, results, output_path):
        """Save analysis results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=4)
            
    def process_video(self, video_path, output_path, analysis_type="shot_classification", custom_prompt=None, progress_callback=None):
        """
        Process video with progress updates (compatible with existing analyzer interface)
        
        Args:
            video_path (str): Path to the video file
            output_path (str): Path to save the analysis results
            analysis_type (str): Type of analysis to perform
            custom_prompt (str, optional): Custom prompt to use
            progress_callback (function, optional): Callback for progress updates
            
        Returns:
            dict: Structured analysis results
        """
        try:
            # Update progress
            if progress_callback:
                progress_callback(10, "Uploading video to Gemini...")
            
            # Process the video
            results = self.analyze_video(video_path, analysis_type, custom_prompt)
            
            if progress_callback:
                progress_callback(90, "Processing complete, formatting results...")
            
            # Format the results for our API structure
            formatted_results = self._format_results_for_api(results, video_path, output_path)
            
            # Save results
            results_filename = os.path.splitext(output_path)[0] + ".json"
            self.save_results(formatted_results, results_filename)
            
            if progress_callback:
                progress_callback(100, "Analysis complete")
            
            return formatted_results
            
        except Exception as e:
            if progress_callback:
                progress_callback(0, f"Error: {str(e)}")
            raise
    
    def _format_results_for_api(self, gemini_results, video_path, output_path):
        """Format Gemini results to match our API structure"""
        
        # Handle case where the result might be a list
        if isinstance(gemini_results, list):
            if len(gemini_results) > 0:
                gemini_results = gemini_results[0]
            else:
                # Empty list, create an empty dict
                gemini_results = {}
                
        # Check for error
        if isinstance(gemini_results, dict) and "error" in gemini_results:
            return {
                "metrics": {},
                "file_info": {
                    "input_video": os.path.basename(video_path),
                    "output_file": os.path.basename(output_path),
                    "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "analysis_method": "gemini-ai"
                },
                "gemini_analysis": {
                    "error": gemini_results.get("error"),
                    "raw_response": gemini_results.get("raw_response", "")
                }
            }
        
        # Create standard format results for any type of response structure
        results = {
            "metrics": {},  # No numerical metrics from text analysis
            "file_info": {
                "input_video": os.path.basename(video_path),
                "output_file": os.path.basename(output_path),
                "processed_date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "analysis_method": "gemini-ai"
            },
            "gemini_analysis": {
                "shot_classification": gemini_results.get("shot_classification"),
                "technical_analysis": gemini_results.get("technical_analysis"),
                "suggested_improvements": gemini_results.get("suggested_improvements"),
                "raw_response": gemini_results.get("raw_response", "")
            }
        }
        
        return results