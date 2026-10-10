from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import easyocr
import spacy
import numpy as np
import urllib.request
import os
import tempfile
from PIL import Image
import base64
import io

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize models
reader = None
nlp = None
face_net = None
models_initialized = False  # Flag to track initialization status

def initialize_models():
    """Initialize the NLP and face detection models"""
    global reader, nlp, face_net
    
    try:
        # Initialize EasyOCR reader
        try:
            reader = easyocr.Reader(['en'])
            print("OCR model loaded successfully")
        except Exception as e:
            print(f"Error loading OCR model: {str(e)}")
            reader = None
        
        # Initialize spaCy NLP model
        try:
            nlp = spacy.load("en_core_web_sm")
            print("NLP model loaded successfully")
        except Exception as e:
            print(f"Error loading NLP model: {str(e)}")
            nlp = None
        
        # Initialize face detector
        try:
            download_face_model_if_needed()
            
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
            prototxt_path = os.path.join(model_dir, "deploy.prototxt")
            weights_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
            
            if os.path.exists(prototxt_path) and os.path.exists(weights_path):
                face_net = cv2.dnn.readNet(prototxt_path, weights_path)
                print("Face detection model loaded successfully")
            else:
                print(f"Face detection model files not found at expected locations: {prototxt_path}, {weights_path}")
                face_net = None
        except Exception as e:
            print(f"Error loading face detection model: {str(e)}")
            face_net = None
        
        print("All available models loaded")
        
    except Exception as e:
        print(f"Error initializing models: {str(e)}")

def download_face_model_if_needed():
    """Download the face detection model files if they don't exist"""
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Define model files
    prototxt_path = os.path.join(model_dir, "deploy.prototxt")
    weights_path = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
    
    # Download prototxt if needed
    if not os.path.exists(prototxt_path):
        print("Downloading face detection model configuration...")
        try:
            prototxt_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
            urllib.request.urlretrieve(prototxt_url, prototxt_path)
        except Exception as e:
            print(f"Error downloading face detection model configuration: {str(e)}")
            return False
    
    # Download weights if needed
    if not os.path.exists(weights_path):
        print("Downloading face detection model weights... (This may take a while)")
        try:
            # This is the OpenCV DNN sample face detector model
            weights_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
            urllib.request.urlretrieve(weights_url, weights_path)
        except Exception as e:
            print(f"Error downloading face detection model weights: {str(e)}")
            return False
    
    return True

def anonymize_face_pixelate(face, blocks=10):
    """Anonymize a face region using pixelation"""
    # Get the dimensions of the face
    (h, w) = face.shape[:2]
    
    # Calculate the size of each block in pixels
    x_steps = np.linspace(0, w, blocks + 1, dtype="int")
    y_steps = np.linspace(0, h, blocks + 1, dtype="int")
    
    # Make a copy of the face to avoid modifying the original
    pixelated_face = face.copy()
    
    # Loop over each block in the face
    for i in range(1, len(y_steps)):
        for j in range(1, len(x_steps)):
            # Compute the starting and ending (x, y)-coordinates of the current block
            start_x = x_steps[j - 1]
            start_y = y_steps[i - 1]
            end_x = x_steps[j]
            end_y = y_steps[i]
            
            # Extract the ROI of the block from the face
            roi = face[start_y:end_y, start_x:end_x]
            
            # Skip empty ROIs
            if roi.size == 0:
                continue
            
            # Fill the block with the mean pixel intensity value of the ROI
            B = roi[:, :, 0].mean()
            G = roi[:, :, 1].mean()
            R = roi[:, :, 2].mean()
            pixelated_face[start_y:end_y, start_x:end_x] = (B, G, R)
    
    # Return the pixelated face
    return pixelated_face

def apply_blurring_to_frame(frame, pii_bounding_boxes, confidence_threshold):
    """Apply blurring to detected faces and PII areas in a frame"""
    # Make a copy of the frame to avoid modifying the original
    processed_frame = frame.copy()
    
    # Detect faces in the frame if face detector is available
    if face_net is not None:
        try:
            height, width = processed_frame.shape[:2]
            blob = cv2.dnn.blobFromImage(processed_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()
            
            # Apply blurring to detected faces and draw confidence percentage
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    # Get coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * [width, height, width, height]
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure coordinates are within frame boundaries
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(width, endX), min(height, endY)
                    
                    # Skip if box dimensions are invalid
                    if startX >= endX or startY >= endY:
                        continue
                    
                    # Extract the face region
                    face = processed_frame[startY:endY, startX:endX]
                    
                    # Apply Gaussian blur to anonymize the face
                    blurred_face = cv2.GaussianBlur(face, (99, 99), 30)
                    
                    # Replace the face region with the blurred face
                    processed_frame[startY:endY, startX:endX] = blurred_face
                    
                    # Draw a bounding box and confidence percentage
                    label = f"{confidence * 100:.2f}%"
                    processed_frame = cv2.putText(processed_frame, label, (startX, startY-10 if startY-10>10 else startY+10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    processed_frame = cv2.rectangle(processed_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        except Exception as e:
            print(f"Face detection error: {str(e)}")
    
    # Apply blurring to areas containing PII
    height, width = processed_frame.shape[:2]
    for top_left, bottom_right in pii_bounding_boxes:
        # Ensure coordinates are within frame boundaries
        x1, y1 = max(0, top_left[0]), max(0, top_left[1])
        x2, y2 = min(width, bottom_right[0]), min(height, bottom_right[1])
        
        # Skip if box dimensions are invalid
        if x1 >= x2 or y1 >= y2:
            continue
        
        # Blur the area containing PII
        blurred_area = cv2.GaussianBlur(processed_frame[y1:y2, x1:x2], (99, 99), 30)
        
        # Replace the area containing PII with the blurred area
        processed_frame[y1:y2, x1:x2] = blurred_area
        
        # Draw a bounding box around the PII area
        processed_frame = cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return processed_frame

def apply_pixelation_to_frame(frame, pii_bounding_boxes, confidence_threshold):
    """Apply pixelation to detected faces and PII areas in a frame"""
    # Make a copy of the frame to avoid modifying the original
    processed_frame = frame.copy()
    
    # Detect faces in the frame if face detector is available
    if face_net is not None:
        try:
            height, width = processed_frame.shape[:2]
            blob = cv2.dnn.blobFromImage(processed_frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
            face_net.setInput(blob)
            detections = face_net.forward()
            
            # Apply pixelation to detected faces and draw confidence percentage
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                if confidence > confidence_threshold:
                    # Get coordinates of the bounding box
                    box = detections[0, 0, i, 3:7] * [width, height, width, height]
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Ensure coordinates are within frame boundaries
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(width, endX), min(height, endY)
                    
                    # Skip if box dimensions are invalid
                    if startX >= endX or startY >= endY:
                        continue
                    
                    # Extract the face region
                    face = processed_frame[startY:endY, startX:endX]
                    
                    # Apply pixelation to anonymize the face
                    pixelated_face = anonymize_face_pixelate(face, blocks=10)
                    
                    # Replace the face region with the pixelated face
                    processed_frame[startY:endY, startX:endX] = pixelated_face
                    
                    # Draw a bounding box and confidence percentage
                    label = f"{confidence * 100:.2f}%"
                    processed_frame = cv2.putText(processed_frame, label, (startX, startY-10 if startY-10>10 else startY+10),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
                    processed_frame = cv2.rectangle(processed_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        except Exception as e:
            print(f"Face detection error: {str(e)}")
    
    # Apply pixelation to areas containing PII
    height, width = processed_frame.shape[:2]
    for top_left, bottom_right in pii_bounding_boxes:
        # Ensure coordinates are within frame boundaries
        x1, y1 = max(0, top_left[0]), max(0, top_left[1])
        x2, y2 = min(width, bottom_right[0]), min(height, bottom_right[1])
        
        # Skip if box dimensions are invalid
        if x1 >= x2 or y1 >= y2:
            continue
        
        # Extract the PII region
        pii_area = processed_frame[y1:y2, x1:x2]
        
        # Apply pixelation to anonymize the PII
        pixelated_pii = anonymize_face_pixelate(pii_area, blocks=10)
        
        # Replace the PII region with the pixelated PII
        processed_frame[y1:y2, x1:x2] = pixelated_pii
        
        # Draw a bounding box around the PII area
        processed_frame = cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    return processed_frame

def detect_pii_in_image(image_array):
    """Detect PII in an image using OCR and NLP"""
    pii_bounding_boxes = []
    
    # Perform OCR to detect text if OCR reader is available
    if reader is not None and nlp is not None:
        try:
            # Create a temporary file to store the image
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name
                
            # Save the image to the temporary file
            cv2.imwrite(temp_path, image_array)
            
            # Perform OCR
            results = reader.readtext(temp_path)
            
            # Clean up the temporary file
            os.unlink(temp_path)
            
            for bbox, text, prob in results:
                # Use spaCy to analyze the detected text
                doc = nlp(text)
                for ent in doc.ents:
                    # Check if the detected entity is PII
                    if ent.label_ in ['PERSON', 'ORG', 'GPE', 'LOC', 'DATE', 'TIME', 'MONEY', 'QUANTITY', 
                                     'ORDINAL', 'CARDINAL', 'NORP', 'FAC', 'EVENT', 'WORK_OF_ART', 
                                     'LAW', 'LANGUAGE']:
                        # Calculate bounding box coordinates
                        top_left = (int(bbox[0][0]), int(bbox[0][1]))
                        bottom_right = (int(bbox[2][0]), int(bbox[2][1]))
                        
                        # Store the bounding box of the PII area
                        pii_bounding_boxes.append([top_left, bottom_right])
        except Exception as e:
            print(f"OCR processing error: {str(e)}")
    
    return pii_bounding_boxes

@app.route('/api/status', methods=['GET'])
def status():
    """API endpoint to check the status of the server and models"""
    return jsonify({
        'status': 'online',
        'models': {
            'ocr': reader is not None,
            'nlp': nlp is not None,
            'face_detection': face_net is not None
        }
    })

@app.route('/api/anonymize', methods=['POST'])
def anonymize_image():
    """API endpoint to anonymize an image"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get anonymization method and confidence threshold
    anonymization_method = request.form.get('method', 'blur')
    confidence_threshold = float(request.form.get('confidence', '0.5'))
    
    # Read the image file
    image_file = request.files['image']
    image_data = image_file.read()
    
    # Convert to OpenCV format
    nparr = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect PII in the image
    pii_bounding_boxes = detect_pii_in_image(img)
    
    # Apply anonymization
    if anonymization_method.lower() == 'blur':
        processed_image = apply_blurring_to_frame(img, pii_bounding_boxes, confidence_threshold)
    else:
        processed_image = apply_pixelation_to_frame(img, pii_bounding_boxes, confidence_threshold)
    
    # Convert the processed image to base64
    _, buffer = cv2.imencode('.jpg', processed_image)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{img_base64}',
        'pii_count': len(pii_bounding_boxes)
    })

@app.before_request
def before_request():
    """Initialize models before the first request"""
    global models_initialized
    if not models_initialized:
        initialize_models()
        models_initialized = True

if __name__ == '__main__':
    # Initialize models immediately when running directly
    initialize_models()
    
    # Start the Flask app
    app.run(debug=True, host='0.0.0.0', port=5001)