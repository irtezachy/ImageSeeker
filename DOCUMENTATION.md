# ImageSeeker Documentation

## Overview
ImageSeeker is a sophisticated visual search system that helps users discover visually similar images through a combination of local and web-based search capabilities. The system employs advanced image processing techniques, machine learning models, and multiple search services to provide comprehensive and relevant results.

## System Architecture

### 1. Core Components

#### 1.1 Image Processing Pipeline
- **Upload Handler**: Manages image uploads and validation
- **Image Classifier**: Uses ResNet50 for content understanding
- **Feature Extractor**: Analyzes visual characteristics
- **Search Engine**: Multi-source image search system

#### 1.2 Search Services
- Google Custom Search API (Primary)
- Pexels API (Secondary)
- Unsplash (Tertiary)
- Lorem Picsum (Fallback)

### 2. Technical Stack
- **Backend**: Python/Flask
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: OpenCV, Pillow
- **Machine Learning**: PyTorch, torchvision
- **APIs**: Google Custom Search, Pexels
- **Dependencies**: See requirements.txt

## Detailed Process Flow

### 1. Image Upload and Validation
```python
def upload_file():
    # Validate file presence
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    # Process uploaded file
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
```
- Accepts image uploads
- Validates file format and size
- Stores temporarily in uploads directory

### 2. Image Classification
```python
def classify_image(image_path):
    # Load pre-trained model
    model = models.resnet50(pretrained=True)
    model.eval()
    
    # Process image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
```
- Uses ResNet50 model
- Preprocesses image for classification
- Returns top 3 categories

### 3. Feature Extraction
```python
def extract_image_features(image):
    features = []
    
    # Dimension analysis
    width, height = image.size
    if width > height:
        features.append("landscape")
    elif height > width:
        features.append("portrait")
    else:
        features.append("square")
    
    # Color analysis
    image = image.convert('RGB')
    colors = image.getcolors(maxcolors=256)
```
- Analyzes image dimensions
- Extracts color information
- Identifies composition type

### 4. Search Process
```python
def search_similar_images(image_path, category=None, max_results=10):
    # Generate search terms
    search_terms = [category] if category else []
    features = extract_image_features(image)
    search_terms.extend(features)
    
    # Multi-source search
    results = []
    google_results = get_google_search_results(search_terms, max_results)
    if google_results:
        results.extend(google_results)
```
- Generates search queries
- Implements cascading search
- Combines results from multiple sources

## API Integration

### 1. Google Custom Search
```python
def get_google_search_results(search_terms, max_results=10):
    api_key = os.getenv('GOOGLE_API_KEY')
    cse_id = os.getenv('GOOGLE_CSE_ID')
    
    service = build('customsearch', 'v1', developerKey=api_key)
    result = service.cse().list(
        q=' '.join(search_terms),
        cx=cse_id,
        searchType='image',
        num=max_results
    ).execute()
```
- Primary search source
- Requires API key and Search Engine ID
- Returns structured results

### 2. Pexels API
```python
response = requests.get(
    'https://api.pexels.com/v1/search',
    headers={'Authorization': pexels_key},
    params={'query': ' '.join(search_terms), 'per_page': max_results}
)
```
- Secondary search source
- Requires API key
- Returns high-quality stock photos

## Frontend Interface

### 1. Main Components
```html
<div class="results-container">
    <div class="result-card">
        <img src="{{ result.url }}" alt="{{ result.title }}">
        <div class="result-info">
            <h3>{{ result.title }}</h3>
            <span class="source-badge">{{ result.source }}</span>
        </div>
    </div>
</div>
```
- Responsive grid layout
- Image cards with metadata
- Source indicators

### 2. User Interaction
- Drag-and-drop upload
- Real-time feedback
- Progress indicators
- Error messages

## Error Handling

### 1. API Failures
```python
try:
    google_results = get_google_search_results(search_terms, max_results)
except Exception as e:
    print(f"Error in Google search: {str(e)}")
    # Fall back to alternative sources
```
- Graceful degradation
- Multiple fallback options
- Detailed error logging

### 2. Image Processing
```python
try:
    image = Image.open(image_path)
    features = extract_image_features(image)
except Exception as e:
    print(f"Error analyzing image: {str(e)}")
```
- Format validation
- Size limits
- Processing errors

## Performance Optimization

### 1. Image Processing
- Efficient resizing
- Caching of processed images
- Optimized feature extraction

### 2. Search Optimization
- Parallel API calls
- Result caching
- Pagination support

## Security Considerations

### 1. File Upload
- Format validation
- Size limits
- Path traversal prevention

### 2. API Security
- Environment variable storage
- Rate limiting
- Error message sanitization

## Deployment

### 1. Requirements
- Python 3.8+
- Virtual environment
- Required API keys

### 2. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate environment
.\.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Unix/MacOS

# Install dependencies
pip install -r requirements.txt
```

### 3. Configuration
```env
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_search_engine_id
PEXELS_API_KEY=your_pexels_api_key
```

## Maintenance

### 1. Regular Tasks
- Update dependencies
- Monitor API usage
- Clean temporary files

### 2. Troubleshooting
- Check API credentials
- Verify file permissions
- Monitor error logs

## Future Enhancements

### 1. Planned Features
- Advanced image similarity
- Face detection
- Object detection
- Custom model training

### 2. Performance Improvements
- Caching system
- Load balancing
- CDN integration

## Support

### 1. Getting Help
- GitHub Issues
- Documentation updates
- Community support

### 2. Contributing
- Fork repository
- Create feature branch
- Submit pull request

## License
This project is licensed under the MIT License - see the LICENSE file for details. 