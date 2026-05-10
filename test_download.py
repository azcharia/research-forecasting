import requests
import zipfile
import io
import os

token = os.environ.get("KAGGLE_API_TOKEN")
if not token:
    raise RuntimeError("KAGGLE_API_TOKEN is required to download from Kaggle API.")
headers = {"Authorization": f"Bearer {token}"}
url = "https://www.kaggle.com/api/v1/datasets/download/robikscube/hourly-energy-consumption"
print("Downloading...")
response = requests.get(url, headers=headers, stream=True)
if response.status_code == 200:
    print("Download successful.")
    # Assuming it's a zip file
    try:
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall("d:/WORK/research/dataset")
            print("Extracted files:", z.namelist())
    except Exception as e:
        print("Failed to extract:", e)
else:
    print(f"Failed to download: {response.status_code}")
    print(response.text)
