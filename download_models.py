import os
import requests
from tqdm import tqdm  # Optional, for progress bar

MODEL_URLS = {
    'gender_net.caffemodel': 'https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/gender_net.caffemodel',
    'age_net.caffemodel': 'https://github.com/GilLevi/AgeGenderDeepLearning/raw/master/models/age_net.caffemodel',
}

def download_file(url, path):
    print(f"Downloading {os.path.basename(path)}...")
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    with open(path, 'wb') as f, tqdm(desc=os.path.basename(path), total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))

if __name__ == '__main__':
    for filename, url in MODEL_URLS.items():
        path = os.path.join(os.path.dirname(__file__), filename)
        if os.path.exists(path):
            print(f"{filename} already exists. Skipping.")
            continue
        download_file(url, path)
        print(f"Downloaded {filename}")
    print("Models ready! Run: python detect_age_gender.py")
