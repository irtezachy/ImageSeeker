import os
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
from skimage.metrics import structural_similarity
from skimage.feature import hog
from skimage.color import rgb2gray
from scipy.spatial.distance import cosine
from utils.image_validator import validate_image
from utils.web_search import search_similar_images
from dotenv import load_dotenv
import cv2
import numpy.typing as npt
from typing import Optional, Union
import torch
from torchvision import models, transforms
from collections import Counter

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Load pre-trained ResNet model for image classification
model = models.resnet50(pretrained=True)
model.eval()

# ImageNet class labels
with open('imagenet_classes.txt', 'r') as f:
    IMAGENET_CLASSES = [line.strip() for line in f.readlines()]

# Image preprocessing for the model
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image_path: str) -> list:
    """
    Classify the image and return top categories
    """
    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')
        input_tensor = preprocess(image)
        input_batch = input_tensor.unsqueeze(0)

        # Get model predictions
        with torch.no_grad():
            output = model(input_batch)

        # Get top 5 predictions
        _, indices = torch.topk(output[0], 5)
        categories = [IMAGENET_CLASSES[idx] for idx in indices]
        
        # Extract meaningful keywords from categories
        keywords = []
        for category in categories:
            # Split category into words and remove common words
            words = category.replace('_', ' ').split()
            keywords.extend([word.lower() for word in words if len(word) > 3])
        
        # Get most common keywords
        common_keywords = Counter(keywords).most_common(3)
        return [keyword for keyword, _ in common_keywords]

    except Exception as e:
        print(f"Error classifying image: {str(e)}")
        return []

def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image: Union[Image.Image, npt.NDArray]) -> Optional[npt.NDArray]:
    """
    Extract multiple features from the image for better comparison
    """
    try:
        # Convert to numpy array if not already
        if isinstance(image, Image.Image):
            image_array = np.array(image)
        else:
            image_array = image

        # Ensure image is RGB
        if len(image_array.shape) == 2:  # Grayscale
            image_array = cv2.cvtColor(image_array, cv2.COLOR_GRAY2RGB)
        elif image_array.shape[2] == 4:  # RGBA
            image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2RGB)

        # Resize for consistent processing
        image_array = cv2.resize(image_array, (200, 200))

        # 1. Color histogram features
        color_hist = cv2.calcHist([image_array], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        color_hist = cv2.normalize(color_hist, color_hist).flatten()

        # 2. HOG features
        gray_image = rgb2gray(image_array)
        hog_features = hog(gray_image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1))

        # 3. Edge features using Canny
        edges = cv2.Canny(image_array, 100, 200)
        edge_features = edges.flatten() / 255.0

        # 4. Basic statistical features
        stats_features = np.array([
            np.mean(image_array, axis=(0, 1)),
            np.std(image_array, axis=(0, 1)),
            np.max(image_array, axis=(0, 1)),
            np.min(image_array, axis=(0, 1))
        ]).flatten()

        # Combine all features
        return np.concatenate([color_hist, hog_features, edge_features, stats_features])

    except (ValueError, cv2.error) as e:
        print(f"Error extracting features: {str(e)}")
        return None
    except Exception as e:
        print(f"Unexpected error extracting features: {str(e)}")
        return None

def calculate_similarity(img1: Image.Image, img2: Image.Image) -> float:
    """
    Calculate similarity between two images using multiple metrics
    """
    try:
        # Extract features for both images
        features1 = extract_features(img1)
        features2 = extract_features(img2)

        if features1 is None or features2 is None:
            return 0.0

        # Calculate multiple similarity metrics
        try:
            # 1. Cosine similarity between feature vectors
            cosine_sim = 1 - cosine(features1, features2)
        except (ValueError, np.linalg.LinAlgError) as e:
            print(f"Error calculating cosine similarity: {str(e)}")
            cosine_sim = 0.0

        try:
            # 2. Structural similarity on grayscale images
            img1_gray = cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY)
            img2_gray = cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY)
            img1_gray = cv2.resize(img1_gray, (200, 200))
            img2_gray = cv2.resize(img2_gray, (200, 200))
            ssim = structural_similarity(img1_gray, img2_gray)
        except (ValueError, cv2.error) as e:
            print(f"Error calculating structural similarity: {str(e)}")
            ssim = 0.0

        # Combine similarities with weights
        final_similarity = (0.6 * cosine_sim + 0.4 * ssim)

        # Apply a more sophisticated threshold
        if final_similarity > 0.4:  # Base threshold
            # Boost score if both metrics agree
            if cosine_sim > 0.4 and ssim > 0.4:
                final_similarity = final_similarity * 1.2
            # Cap at 1.0
            final_similarity = min(final_similarity, 1.0)
            return final_similarity
        return 0.0

    except Exception as e:
        print(f"Error calculating similarity: {str(e)}")
        return 0.0

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if file and allowed_file(file.filename):
            print(f"Processing uploaded file: {file.filename}")
            # Validate the image before processing
            is_valid, error_message = validate_image(file)
            if not is_valid:
                print(f"Image validation failed: {error_message}")
                return jsonify({'error': error_message}), 400
            
            # Reset file pointer after validation
            file.seek(0)
            
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f"File saved to: {filepath}")
            
            # Classify the image
            categories = classify_image(filepath)
            print(f"Detected categories: {categories}")
            
            # Process the uploaded image
            uploaded_image = Image.open(filepath)
            print(f"Uploaded image opened: mode={uploaded_image.mode}, size={uploaded_image.size}")
            
            # Compare with existing images in the upload folder
            local_similarities = []
            existing_files = os.listdir(app.config['UPLOAD_FOLDER'])
            print(f"Found {len(existing_files)} files in uploads directory")
            
            for existing_file in existing_files:
                if existing_file != filename:
                    try:
                        existing_path = os.path.join(app.config['UPLOAD_FOLDER'], existing_file)
                        print(f"Comparing with: {existing_file}")
                        existing_image = Image.open(existing_path)
                        similarity = calculate_similarity(uploaded_image, existing_image)
                        if similarity > 0:  # Only add if similarity is above threshold
                            local_similarities.append({
                                'filename': existing_file,
                                'similarity': float(similarity),
                                'path': f'/uploads/{existing_file}',
                                'source': 'local'
                            })
                            print(f"Similarity with {existing_file}: {similarity}")
                    except (IOError, OSError) as e:
                        print(f"Error processing {existing_file}: {str(e)}")
                        continue
            
            # Search for similar images on the web using detected categories
            print("Searching for similar images on the web...")
            web_images = []
            web_errors = []  # List to store errors from each category search
            
            # Search for each category separately
            for category in categories:
                category_images, category_error = search_similar_images(filepath, category)
                if category_error:
                    print(f"Web search error for category {category}: {category_error}")
                    web_errors.append(f"{category}: {category_error}")
                    continue
                web_images.extend(category_images)
            
            # Combine local and web results
            all_results = local_similarities + [
                {
                    'filename': img['title'],
                    'similarity': 0.5,  # Default similarity score for web results
                    'path': img['url'],
                    'thumbnail': img['thumbnail'],
                    'source': 'web',
                    'source_url': img['source'],
                    'category': img.get('category', 'Unknown')
                }
                for img in web_images
            ]
            
            # Sort by similarity (highest first)
            all_results.sort(key=lambda x: x['similarity'], reverse=True)
            print(f"Found {len(all_results)} similar images in total")
            
            response_data = {
                'message': 'File uploaded successfully',
                'filename': filename,
                'path': f'/uploads/{filename}',
                'categories': categories,
                'similarities': all_results
            }
            
            # Add errors to response if any occurred
            if web_errors:
                response_data['web_search_errors'] = web_errors
            
            return jsonify(response_data)
        
        return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        print(f"Error in upload_file: {str(e)}")
        return jsonify({'error': 'An error occurred while processing the file'}), 500

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except (IOError, OSError) as e:
        print(f"Error serving file {filename}: {str(e)}")
        return jsonify({'error': 'File not found'}), 404

if __name__ == '__main__':
    app.run(debug=True) 