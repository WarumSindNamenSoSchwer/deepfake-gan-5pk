import os
import requests
from serpapi import GoogleSearch
from PIL import Image
from io import BytesIO

# Get your free SerpAPI key from https://serpapi.com/
SERPAPI_KEY = "2ce52c82049a918463f66e2e0c24f80d1c5d596c7ee2226c46ca5489d16107b2"

def fetch_google_images(query, max_images=5):
    """Fetch image URLs from Google Images using SerpAPI."""
    params = {
        "engine": "google_images",
        "q": query,
        "num": max_images,
        "api_key": SERPAPI_KEY
    }
    search = GoogleSearch(params)
    results = search.get_dict()
    return [img["original"] for img in results.get("images_results", [])[:max_images]]

def save_images(image_urls, save_dir):
    """Download and save images as .png files."""
    os.makedirs(save_dir, exist_ok=True)

    for i, url in enumerate(image_urls):
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB")  # Convert to RGB for PNG compatibility
            save_path = os.path.join(save_dir, f"image_{i+1}.png")
            image.save(save_path, "PNG")

            print(f"Saved: {save_path}")
        except Exception as e:
            print(f"Failed to save {url}: {e}")

if __name__ == "__main__":
    politician = input("Enter the politician's name: ")
    save_directory = "Bilder"

    print(f"Searching for images of {politician}...")
    urls = fetch_google_images(politician, max_images=50)

    if urls:
        print("Downloading images...")
        save_images(urls, save_directory)
        print("Download complete.")
    else:
        print("No images found.")
