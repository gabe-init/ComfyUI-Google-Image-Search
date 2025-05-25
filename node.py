from googleapiclient.discovery import build
import requests
from PIL import Image
from io import BytesIO
import torch
import numpy as np
import json
import os

class GoogleImageSearchNode:
    def __init__(self):
        self.config = self.load_config()
    
    def load_config(self):
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading config.json: {e}")
                return {}
        else:
            print("config.json not found. Please copy config.json.example to config.json and add your credentials.")
            return {}
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "search_query": ("STRING", {"default": "cat"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "search_image"
    CATEGORY = "Custom Nodes/Google"

    def search_image(self, search_query):
        api_key = self.config.get('api_key', '')
        search_engine_id = self.config.get('search_engine_id', '')
        
        if not api_key or not search_engine_id:
            print("Error: API key and Search Engine ID are required. Please configure config.json")
            # Return a red error image
            error_tensor = torch.zeros(1, 64, 64, 3)
            error_tensor[0, :, :, 0] = 1
            return (error_tensor,)
            
        try:
            print(f"Starting search for: {search_query}")
            
            # Initialize the Custom Search API service
            service = build("customsearch", "v1", developerKey=api_key)

            # Perform the search with the provided Search Engine ID
            result = service.cse().list(
                q=search_query,
                cx=search_engine_id,
                searchType='image',
                num=1,
                safe='off'  # Disable SafeSearch to get more results
            ).execute()

            print(f"Search API response received")
            
            # Get the first image URL
            if 'items' in result and len(result['items']) > 0:
                image_url = result['items'][0]['link']
                print(f"Found image URL: {image_url}")
                
                # Download the image
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()  # Raise an exception for bad status codes
                
                print(f"Image downloaded successfully")

                # Convert to PIL Image
                img = Image.open(BytesIO(response.content))
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                print(f"Image converted to RGB format: {img.size}")
                
                # Convert PIL image to numpy array
                img_array = np.array(img).astype(np.float32) / 255.0
                
                # Convert numpy array to tensor with correct shape [B, H, W, C]
                img_tensor = torch.from_numpy(img_array)
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension
                
                print(f"Successfully created tensor with shape: {img_tensor.shape}")
                return (img_tensor,)
            else:
                print(f"No images found in API response: {result}")
                raise Exception("No images found in search results")

        except Exception as e:
            print(f"Detailed error in Google Image Search: {type(e).__name__}: {str(e)}")
            # Return a small red tensor to indicate error
            error_tensor = torch.zeros(1, 64, 64, 3)  # [B, H, W, C] format
            error_tensor[0, :, :, 0] = 1  # Red channel
            return (error_tensor,)

NODE_CLASS_MAPPINGS = {
    "GoogleImageSearchNode": GoogleImageSearchNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GoogleImageSearchNode": "Google Image Search"
}