from PIL import Image

def validate_image(file):
    """
    Validate an uploaded image file.
    Returns (is_valid, error_message)
    """
    try:
        # Try to open the image
        image = Image.open(file)
        
        # Check if it's a valid image format
        if image.format not in ['JPEG', 'PNG', 'GIF']:
            return False, "Unsupported image format. Please upload JPEG, PNG, or GIF images."
        
        # Check image dimensions
        width, height = image.size
        if width < 10 or height < 10:
            return False, "Image dimensions are too small."
        if width > 5000 or height > 5000:
            return False, "Image dimensions are too large."
        
        # Check file size (should be checked before this function is called)
        
        # Try to verify the image data
        image.verify()
        
        return True, None
    except Exception as e:
        return False, f"Invalid image file: {str(e)}" 