from flask import Flask, request, jsonify
import io
import json
import re
import mimetypes
import tempfile
import os
from google import genai
from google.genai import types

app = Flask(__name__)

client = genai.Client(api_key="AIzaSyDuoaSb4koA4JBsY9g5gXUSYAFIMbVd_wA")
system_instruction = """
You are an AI expert in Northeast Indian languages and cultural heritage preservation.
Your job:
1. Carefully analyze the text in the image to identify ALL languages present.
2. For each language detected, determine:
   - The specific language name
   - Writing script used
   - Confidence level
   - Percentage of text in that language
   - Linguistic family
3. For each detected language, provide relevant references:
   - History of the language
   - Cultural significance
   - Links to credible resources or further reading
4. Translate all text in the image into English and include it in the JSON. If translation is not possible, return "none".

Return only valid JSON with this exact structure:
{
  "languages": {
    "primary_language": "",
    "detected_languages": [
      {
        "name": "",
        "script": "",
        "confidence": 0.0,
        "percentage": "",
        "linguistic_family": "",
        "additional_info": {
          "history": "",
          "cultural_significance": "",
          "resources": ["link1", "link2"]
        }
      }
    ]
  },
  "script": "",
  "confidence": 0.0,
  "text_direction": "",
  "notes": "",
  "additional_info": "",
  "english_translation": ""
}
"""

def get_mime_type(filename, file_bytes):
    """Get MIME type from filename and file content"""
    # First try to get from filename
    mime_type, _ = mimetypes.guess_type(filename)
    
    # If that fails, try to determine from file header
    if not mime_type:
        # Check common image file signatures
        if file_bytes.startswith(b'\xFF\xD8\xFF'):
            mime_type = 'image/jpeg'
        elif file_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            mime_type = 'image/png'
        elif file_bytes.startswith(b'GIF87a') or file_bytes.startswith(b'GIF89a'):
            mime_type = 'image/gif'
        elif file_bytes.startswith(b'RIFF') and b'WEBP' in file_bytes[:12]:
            mime_type = 'image/webp'
        else:
            # Default fallback
            mime_type = 'image/png'
    
    return mime_type

def extract_json_from_response(response_text):
    """Extract JSON from response text"""
    try:
        # Try to find JSON block in the response
        match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if match:
            return match.group(0)
        else:
            return response_text
    except Exception:
        return response_text

def analyze_image(file_bytes, filename):
    """Analyze an image from bytes and return JSON"""
    try:
        # Determine MIME type
        mime_type = get_mime_type(filename, file_bytes)
        print(f"Detected MIME type: {mime_type}")
        
        # Create temporary file for upload
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_bytes)
            temp_file_path = temp_file.name
        
        try:
            # Upload file to Gemini using the correct method
            uploaded_file = client.files.upload(file=temp_file_path)
            print(f"File uploaded successfully: {uploaded_file.uri}")
            
            # Create content parts
            contents = [
                types.Part.from_text(text=system_instruction),
                types.Part.from_uri(
                    file_uri=uploaded_file.uri,
                    mime_type=mime_type
                )
            ]
            
            # Generate content with structured output
            response = client.models.generate_content(
                model="gemini-2.5-flash-lite",
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.1,  # Lower temperature for more consistent JSON
                    response_mime_type="application/json"  # Request JSON response
                )
            )
            
            print(f"Raw response: {response.text}")
            
            # Try to parse JSON
            try:
                result = json.loads(response.text)
            except json.JSONDecodeError:
                # If direct parsing fails, try to extract JSON
                json_text = extract_json_from_response(response.text)
                result = json.loads(json_text)
            
            # Ensure all required keys exist
            default_structure = {
                "languages": {
                    "primary_language": "unknown",
                    "detected_languages": []
                },
                "script": "unknown",
                "confidence": 0.0,
                "text_direction": "ltr",
                "notes": "",
                "additional_info": ""
            }
            
            # Fill in missing keys
            for key, default_value in default_structure.items():
                if key not in result:
                    result[key] = default_value
            
            return result
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file_path)
            except OSError:
                pass
                
    except Exception as e:
        print(f"Error in analyze_image: {str(e)}")
        return {
            "languages": {
                "primary_language": "unknown", 
                "detected_languages": []
            },
            "script": "unknown",
            "confidence": 0.0,
            "text_direction": "ltr",
            "notes": f"Processing failed: {str(e)}",
            "additional_info": f"Error: {str(e)}"
        }

@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze uploaded image for language detection"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if not file.filename:
        return jsonify({'error': 'No image selected'}), 400
    
    # Read file bytes
    file_bytes = file.read()
    
    if len(file_bytes) == 0:
        return jsonify({'error': 'Empty file uploaded'}), 400
    
    # Analyze the image
    result = analyze_image(file_bytes, file.filename)
    
    return jsonify(result)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'Northeast Indian Language Analyzer'})

if __name__ == '__main__':
    print("Starting Northeast Indian Language Analyzer...")
    print("Make sure to set your GEMINI_API_KEY in the code!")
    app.run(debug=True, host='0.0.0.0', port=5000)
