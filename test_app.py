import os
import unittest
import tempfile
import shutil # For tearDown
from io import BytesIO # For testing empty filename
from app import app # Your Flask app instance

class AppTestCase(unittest.TestCase):

    def setUp(self):
        # Configure the app for testing
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False # Disable CSRF for testing forms if any; not strictly needed here but good practice
        self.client = app.test_client()

        # Create a temporary directory for uploads during tests
        self.test_uploads_dir = tempfile.mkdtemp()
        app.config['UPLOAD_FOLDER'] = self.test_uploads_dir

        # It's good practice to ensure the 'Models' directory and model file exist or are mocked.
        # For this test, we'll assume 'Models/COVID19_VGG19.h5' is accessible relative to 'app.py'
        # as corrected in the previous plan. If model loading is slow or problematic for unit tests,
        # mocking app.model would be the next step.

    def tearDown(self):
        # Remove the temporary directory and its contents
        shutil.rmtree(self.test_uploads_dir)

    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Upload Chest X-Ray Image", response.data) # Check for a byte string

    def test_predict_route_success(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        source_image_path = os.path.join(current_dir, 'uploads', '094.png') # Assumes 094.png is in an 'uploads' folder at project root

        if not os.path.exists(source_image_path):
            # If 094.png doesn't exist at root/uploads, try checking if it's at root (if uploads dir was flattened)
            source_image_path_alt = os.path.join(os.path.dirname(current_dir), 'uploads', '094.png')
            if os.path.exists(source_image_path_alt):
                source_image_path = source_image_path_alt
            else:
                # Fallback: try to find any .png file in the original 'uploads' directory if structure is preserved
                original_uploads_dir = os.path.join(current_dir, 'uploads')
                if os.path.exists(original_uploads_dir) and os.path.isdir(original_uploads_dir):
                    png_files = [f for f in os.listdir(original_uploads_dir) if f.endswith('.png')]
                    if png_files:
                        source_image_path = os.path.join(original_uploads_dir, png_files[0])
                    else:
                        self.skipTest(f"Test image (e.g., 094.png) not found in {original_uploads_dir} or project root's uploads folder. Skipping success test.")
                        return
                else:
                    self.skipTest(f"Test image (e.g., 094.png) not found. Skipping success test.")
                    return

        with open(source_image_path, 'rb') as img_file:
            data = {'file': (img_file, 'test_image.png')} # Filename in tuple is 'test_image.png'
            response = self.client.post('/predict', data=data, content_type='multipart/form-data')

        self.assertEqual(response.status_code, 200, f"Prediction failed with status {response.status_code}, data: {response.data.decode()}")
        
        possible_predictions = [
            b"The Person is Infected With COVID-19 disease",
            b"The Person is Normal",
            b"The Person is Infected With Viral Pneumonia"
        ]
        self.assertTrue(any(prediction in response.data for prediction in possible_predictions),
                        f"Unexpected prediction result: {response.data.decode()}")

    def test_predict_route_no_file_part(self):
        # Simulate a POST request with no file part
        response = self.client.post('/predict', data={})
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"No file part in the request.", response.data)

    def test_predict_route_empty_filename(self):
        # Simulate a POST request with a file part but an empty filename
        # This requires creating a dummy file-like object with an empty filename
        data = {'file': (BytesIO(b"some dummy content"), '')}
        response = self.client.post('/predict', data=data, content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        self.assertIn(b"No selected file.", response.data)

if __name__ == '__main__':
    unittest.main()

# To run these tests, navigate to the project root directory in your terminal and execute:
# python -m unittest test_app.py
