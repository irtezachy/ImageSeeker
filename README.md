# ImageSeeker

ImageSeeker is a powerful image search application that finds similar images both locally and on the web. It uses advanced image processing techniques and multiple search services to provide comprehensive results.

## Features

- Upload and analyze images
- Find similar images using multiple search services:
  - Google Custom Search
  - Pexels
  - Unsplash
  - Lorem Picsum (fallback)
- Image classification using ResNet50
- Color and composition analysis
- Local image similarity comparison
- Web-based user interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ImageSeeker.git
cd ImageSeeker
```

2. Create a virtual environment and activate it:
```bash
python -m venv .venv
# On Windows:
.\.venv\Scripts\activate
# On Unix or MacOS:
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the root directory with the following content:
```
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID=your_search_engine_id
PEXELS_API_KEY=your_pexels_api_key
```

## API Keys Setup

1. Google Custom Search:
   - Get API key from [Google Cloud Console](https://console.cloud.google.com/)
   - Create Search Engine ID at [Programmable Search Engine](https://programmablesearchengine.google.com/)

2. Pexels:
   - Get API key from [Pexels API](https://www.pexels.com/api/)

## Usage

1. Start the application:
```bash
python app.py
```

2. Open your browser and navigate to `http://127.0.0.1:5000`

3. Upload an image and view similar images from both local and web sources

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 