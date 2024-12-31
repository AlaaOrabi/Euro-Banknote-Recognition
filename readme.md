Banknote Recognition API
This project implements a Flask-based API for recognizing the denomination of banknotes using SIFT feature matching. The API processes images uploaded via HTTP requests, compares them with reference images, and returns the recognized denomination.

Features
Preprocess Image: Apply various image preprocessing techniques such as Gaussian blur, bilateral filtering, and histogram equalization.
Recognize Denomination: Use SIFT feature extraction and matching to recognize the denomination of a banknote.
Metadata Stripping: Strip metadata from images to avoid time errors during processing.
Image Normalization: Resize and normalize images to a standard format for consistent processing.
Prerequisites
Python 3.7+
Flask
OpenCV
Pillow
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-username/banknote-recognition-api.git
cd banknote-recognition-api
Create a virtual environment and activate it:

bash
Copy code
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Usage
Start the Flask server:

bash
Copy code
python app.py
The API will be available at http://0.0.0.0:8000.

API Endpoints
Preprocess Image
URL: /preprocess
Method: POST
Description: Preprocess the uploaded image using various techniques and return the paths to the processed images.
Parameters:
image (file): The image file to preprocess.
Response: JSON object containing URLs to the processed images.
Recognize Denomination
URL: /recognize
Method: POST
Description: Recognize the denomination of the uploaded banknote image.
Parameters:
test_image (file): The image file of the banknote to recognize.
Response: JSON object containing the recognized denomination and the URL to the result image with matches drawn.
Get Result Image
URL: /result/<filename>
Method: GET
Description: Get the recognized denomination for a specific result image.
Parameters:
filename (string): The filename of the result image.
Response: JSON object containing the recognized denomination.
Example Requests
Preprocess Image
bash
Copy code
curl -X POST -F 'image=@path_to_your_image.jpg' http://0.0.0.0:8000/preprocess
Recognize Denomination
bash
Copy code
curl -X POST -F 'test_image=@path_to_your_banknote_image.jpg' http://0.0.0.0:8000/recognize
Get Result Image
bash
Copy code
curl -X GET http://0.0.0.0:8000/result/result.jpg
References
Flask Documentation
OpenCV Documentation
Pillow Documentation