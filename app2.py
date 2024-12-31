from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0  # Disable caching of the static files

REFERENCE_IMAGES_DIR = 'reference_images'
REFERENCE_IMAGES_PATHS = {
  '10 Euro': os.path.join(REFERENCE_IMAGES_DIR, '10_back.jpg'),
  
    '20 Euro': os.path.join(REFERENCE_IMAGES_DIR, '20_back.png'),
   
    '50 Euro': os.path.join(REFERENCE_IMAGES_DIR, '50_back.jpg'),
    
    '100 Euro': os.path.join(REFERENCE_IMAGES_DIR, '100_back.jpg'),
    
    '200 Euro': os.path.join(REFERENCE_IMAGES_DIR, '200_back.png'),
   
    '500 Euro': os.path.join(REFERENCE_IMAGES_DIR, '500_back.jpg')
}

recognized_results = {}

def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Image not found or unable to load: {image_path}")
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)
    bilateral_filtered_image = cv2.bilateralFilter(image, 9, 75, 75)
    equalized_image = cv2.equalizeHist(blurred_image)
    return image, blurred_image, bilateral_filtered_image, equalized_image

def extract_features(image):
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(image, None)
    return keypoints, descriptors

def match_features(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def recognize_denomination(test_image_path):
    test_image = preprocess_image(test_image_path)[0]
    kp_test, des_test = extract_features(test_image)

    best_score = float('inf')
    best_denomination = None
    best_matches = None
    best_kp_ref = None
    best_ref_image = None

    for denomination, reference_image_path in REFERENCE_IMAGES_PATHS.items():
        ref_image = preprocess_image(reference_image_path)[0]
        kp_ref, des_ref = extract_features(ref_image)
        matches = match_features(des_ref, des_test)
        score = sum(match.distance for match in matches[:10])
        if score < best_score:
            best_score = score
            best_denomination = denomination
            best_matches = matches
            best_kp_ref = kp_ref
            best_ref_image = ref_image

    return best_denomination, best_matches, kp_test, best_kp_ref, test_image, best_ref_image

@app.route('/preprocess', methods=['POST'])
def preprocess():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image_path = image_file.filename
    image_file.save(image_path)

    try:
        original, blurred_image, bilateral_filtered_image, equalized_image = preprocess_image(image_path)
        result_paths = {
            'original': 'original.jpg',
            'blurred': 'blurred.jpg',
            'bilateral_filtered': 'bilateral_filtered.jpg',
            'equalized': 'equalized.jpg'
        }
        cv2.imwrite(result_paths['original'], original)
        cv2.imwrite(result_paths['blurred'], blurred_image)
        cv2.imwrite(result_paths['bilateral_filtered'], bilateral_filtered_image)
        cv2.imwrite(result_paths['equalized'], equalized_image)

        response = {key: f"/result/{path}" for key, path in result_paths.items()}
        return jsonify(response)
    finally:
        os.remove(image_path)

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'test_image' not in request.files:
        return jsonify({"error": "No test image provided"}), 400

    test_image_file = request.files['test_image']
    test_image_path = test_image_file.filename
    test_image_file.save(test_image_path)

    try:
        denomination, matches, kp_test, kp_ref, test_image, ref_image = recognize_denomination(test_image_path)
        
        # Draw matches
        result_image = cv2.drawMatches(ref_image, kp_ref, test_image, kp_test, matches[:10], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Resize the image to reduce size
        height, width = result_image.shape[:2]
        new_dimensions = (width // 2, height // 2)
        result_image_resized = cv2.resize(result_image, new_dimensions, interpolation=cv2.INTER_AREA)
        
        # Save the resized result image
        result_image_path = 'result.jpg'
        cv2.imwrite(result_image_path, result_image_resized)
        
        # Store the recognized denomination in the dictionary
        recognized_results[result_image_path] = denomination
        
        return jsonify({"recognized_denomination": denomination, "result_image_url": f"/result/{result_image_path}"}), 200
    finally:
        os.remove(test_image_path)

@app.route('/result/<filename>', methods=['GET'])
def get_result(filename):
    denomination = recognized_results.get(filename, "Unknown")
    return jsonify({"recognized_denomination": denomination})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True)
