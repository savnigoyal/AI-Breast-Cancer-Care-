
import requests
import os
import sys

def verify_api():
    url = "http://localhost:5000/predict"
    
    # Check if we have sample images
    # Uses dummy images for now if real ones don't exist
    
    from PIL import Image
    import numpy as np
    
    # Create dummy images
    dummy_cc = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    dummy_mlo = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
    
    dummy_cc.save("test_cc.jpg")
    dummy_mlo.save("test_mlo.jpg")
    
    files = {
        'cc': open("test_cc.jpg", 'rb'),
        'mlo': open("test_mlo.jpg", 'rb')
    }
    
    try:
        print(f"Sending request to {url}...")
        response = requests.post(url, files=files)
        
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print("Response:", response.json())
            print("SUCCESS: API is working correctly!")
        else:
            print("Error:", response.text)
            print("FAILURE: API returned error.")
            
    except requests.exceptions.ConnectionError:
        print("FAILURE: Could not connect to server. Is it running?")
    except Exception as e:
        print(f"FAILURE: An error occurred: {e}")
    finally:
        files['cc'].close()
        files['mlo'].close()
        if os.path.exists("test_cc.jpg"):
            os.remove("test_cc.jpg")
        if os.path.exists("test_mlo.jpg"):
            os.remove("test_mlo.jpg")

if __name__ == "__main__":
    verify_api()
