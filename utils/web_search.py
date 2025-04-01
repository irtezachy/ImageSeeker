import os
import requests
from PIL import Image
from io import BytesIO
import base64
from bs4 import BeautifulSoup
import time
import random
import json
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

def get_google_search_results(search_terms, max_results=10):
    """
    Search for images using Google Custom Search API
    """
    try:
        api_key = os.getenv('GOOGLE_API_KEY')
        cse_id = os.getenv('GOOGLE_CSE_ID')
        
        if not api_key or not cse_id:
            print("Google API credentials not found")
            return []
            
        service = build('customsearch', 'v1', developerKey=api_key)
        
        # Perform the search
        result = service.cse().list(
            q=' '.join(search_terms),
            cx=cse_id,
            searchType='image',
            num=max_results,
            imgType='photo',
            safe='active'
        ).execute()
        
        # Extract image results
        if 'items' in result:
            return [{
                'title': item.get('title', 'Similar Image'),
                'url': item.get('link', ''),
                'thumbnail': item.get('image', {}).get('thumbnailLink', item.get('link', '')),
                'source': 'Google',
                'category': 'search_result'
            } for item in result['items']]
    except Exception as e:
        print(f"Error in Google search: {str(e)}")
    return []

def search_similar_images(image_path, category=None, max_results=10):
    """
    Search for similar images using multiple search services
    """
    try:
        print(f"Searching for images in category: {category}")
        search_terms = [category] if category else []
        
        # Add image characteristics
        try:
            image = Image.open(image_path)
            # Extract image features
            features = extract_image_features(image)
            search_terms.extend(features)
        except Exception as e:
            print(f"Error analyzing image: {str(e)}")

        # Ensure we have at least some search terms
        if not search_terms:
            search_terms = ["image", "photo"]

        print(f"Using search terms: {search_terms}")

        results = []
        
        # 1. Try Google Custom Search first
        google_results = get_google_search_results(search_terms, max_results)
        if google_results:
            results.extend(google_results)
            print(f"Found {len(google_results)} results from Google")

        # 2. Try Pexels API
        try:
            pexels_key = os.getenv('PEXELS_API_KEY')
            if pexels_key:
                response = requests.get(
                    'https://api.pexels.com/v1/search',
                    headers={'Authorization': pexels_key},
                    params={'query': ' '.join(search_terms), 'per_page': max_results}
                )
                if response.status_code == 200:
                    data = response.json()
                    for photo in data.get('photos', []):
                        results.append({
                            'title': photo.get('alt', 'Similar Image'),
                            'url': photo['src']['large'],
                            'thumbnail': photo['src']['small'],
                            'source': 'Pexels',
                            'category': category or 'general'
                        })
        except Exception as e:
            print(f"Error with Pexels API: {str(e)}")

        # 3. Try Unsplash direct URLs
        try:
            for _ in range(max(2, max_results - len(results))):
                random_param = str(random.randint(1, 1000000))
                url = f"https://source.unsplash.com/featured/?{','.join(search_terms)}&random={random_param}"
                results.append({
                    'title': f'Similar {" ".join(search_terms)} Image',
                    'url': url,
                    'thumbnail': url,
                    'source': 'Unsplash',
                    'category': category or 'general'
                })
                time.sleep(0.1)  # Small delay to avoid rate limits
        except Exception as e:
            print(f"Error with Unsplash: {str(e)}")

        # If we still don't have enough results, add some AI-generated placeholder images
        while len(results) < max_results:
            seed = random.randint(1, 1000000)
            results.append({
                'title': f'Similar {" ".join(search_terms)} Image',
                'url': f'https://picsum.photos/seed/{seed}/800/600',
                'thumbnail': f'https://picsum.photos/seed/{seed}/200/150',
                'source': 'Lorem Picsum',
                'category': category or 'general'
            })

        print(f"Found total of {len(results)} results from all sources")
        return results[:max_results], None

    except Exception as e:
        print(f"Error in web search: {str(e)}")
        return [], str(e)

def extract_image_features(image):
    """
    Extract meaningful features from the image for better search
    """
    features = []
    
    # Analyze image dimensions
    width, height = image.size
    if width > height:
        features.append("landscape")
    elif height > width:
        features.append("portrait")
    else:
        features.append("square")
    
    # Analyze colors
    try:
        image = image.convert('RGB')
        colors = image.getcolors(maxcolors=256)
        if colors:
            colors.sort(key=lambda x: x[0], reverse=True)
            dominant_color = colors[0][1]
            
            # Add color information
            r, g, b = dominant_color
            if r > max(g, b) + 50:
                features.append("red")
            elif g > max(r, b) + 50:
                features.append("green")
            elif b > max(r, g) + 50:
                features.append("blue")
            elif r > 200 and g > 200 and b > 200:
                features.append("bright")
            elif r < 50 and g < 50 and b < 50:
                features.append("dark")
            
            # Check for black and white images
            if abs(r - g) < 30 and abs(g - b) < 30:
                if r < 50:
                    features.append("black")
                elif r > 200:
                    features.append("white")
                else:
                    features.append("grayscale")
    except Exception as e:
        print(f"Error analyzing colors: {str(e)}")
    
    return features

def download_image(url):
    """
    Download an image from a URL and return it as a PIL Image object
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except Exception as e:
        raise Exception(f"Failed to download image: {str(e)}") 