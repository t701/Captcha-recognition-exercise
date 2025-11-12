import cv2
import numpy as np
import os

# --- Configuration Constants ---
CHAR_SET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

# Given the image size is 60 (height) x 30 (width).
# Assuming standard Captcha display: Width=60, Height=30 pixels. 

# 5 characters in each captcha. The character width is around 9 pixels (45 / 5).
# Adjust segmentation for a 30 (H) x 60 (W) image.
# Fixed positions for 5-character segmentation (determined empirically/proportional)

IMAGE_HEIGHT = 30
IMAGE_WIDTH = 60
CHAR_WIDTH = 9
CHAR_HEIGHT = 10 
OFFSET_L = 5
OFFSET_T = 11
# Segmentation boundaries (x_start, x_end) for each character slot:
# Slot 1: 5-14, Slot 2: 14-23, Slot 3: 23-32, Slot 4: 32-41, Slot 5: 41-50
SEGMENTATION_BOUNDS = [
    (OFFSET_L, OFFSET_L + CHAR_WIDTH),
    (OFFSET_L + CHAR_WIDTH, OFFSET_L + 2 * CHAR_WIDTH),
    (OFFSET_L + 2 * CHAR_WIDTH, OFFSET_L + 3 * CHAR_WIDTH),
    (OFFSET_L + 3 * CHAR_WIDTH, OFFSET_L + 4 * CHAR_WIDTH),
    (OFFSET_L + 4 * CHAR_WIDTH, OFFSET_L + 5 * CHAR_WIDTH)
]
# Character height crop
CAPTCHA_HEIGHT_CROP = (OFFSET_T, OFFSET_T + CHAR_HEIGHT)

class Captcha(object):
    def __init__(self, templates_dir='templates'):
        """
        Initializes the Captcha solver and loads character templates.
        """
        self.templates_dir = templates_dir
        self.templates = {}
        # Ensure templates directory exists for later saving
        os.makedirs(self.templates_dir, exist_ok=True)
        self._load_templates()

    def _load_templates(self):
        """
        Loads normalized character templates from the templates directory.
        """
        self.templates = {}
        for char in CHAR_SET:
            template_path = os.path.join(self.templates_dir, f'{char}.jpg')
            if os.path.exists(template_path):
                # Load the template in grayscale
                template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
                if template is not None:
                    self.templates[char] = template
        
        if not self.templates and os.listdir(self.templates_dir):
             print("Warning: Templates directory is not empty but no valid templates loaded. Check paths and format.")

    def _preprocess_image(self, image):
        """
        Common image preprocessing steps: Grayscale, Thresholding.
        """
        # 1. Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 2. Apply a binary threshold (Otsu's method)
        # Assuming dark characters on a light background, use BINARY_INV
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        return thresh

    def _segment_character(self, preprocessed_image, index):
        """
        Segments one character based on fixed coordinates (determined by index).
        """
        x_start, x_end = SEGMENTATION_BOUNDS[index]
        y_start, y_end = CAPTCHA_HEIGHT_CROP
        
        # Crop the character area
        char_image = preprocessed_image[y_start:y_end, x_start:x_end]
        
        # For other character patch sizes, can resize it to the standard template size. Fro example, let TEMPLATE_SIZE= (20, 30).
        # char_image_normalized = cv2.resize(char_image, TEMPLATE_SIZE, interpolation=cv2.INTER_AREA)

        return char_image

    def _classify_character(self, segmented_char_img):
        """
        Compares the segmented character image against all templates 
        and finds the best match using Sum of Squared Differences (SSD).
        """
        best_match_char = '?'
        min_distance = float('inf') 

        for char, template in self.templates.items():
            if template is None or template.shape != segmented_char_img.shape:
                continue

            # Calculate the difference (Sum of Squared Differences / L2 Norm)
            difference = segmented_char_img.astype(float) - template.astype(float)
            distance = np.sum(difference**2) # Smaller is better
            
            if distance < min_distance:
                min_distance = distance
                best_match_char = char
            
        return best_match_char
        
    def train(self, samples_base_dir='sampleCaptchas'):
        """
        Trains the system using the provided sample structure (input/output folders).
        Creates and saves normalized templates for all 36 characters.
        """
        print("--- Starting Template Generation (Training) ---")
        input_dir = os.path.join(samples_base_dir, 'input')
        output_dir = os.path.join(samples_base_dir, 'output')
        
        if not os.path.exists(input_dir) or not os.path.exists(output_dir):
            print(f"Error: Required directories '{input_dir}' or '{output_dir}' not found.")
            return

        template_map = {} # Store the best template found for each character

        # Iterate through the 25 samples (index 00 to 24)
        for i in range(25):
            index = f'{i:02d}' # Format as 2-digit index
            im_path = os.path.join(input_dir, f'input{index}.jpg')
            label_path = os.path.join(output_dir, f'output{index}.txt')

            # 1. Load Ground Truth Label
            try:
                with open(label_path, 'r') as f:
                    label = f.read().strip()
                if len(label) != 5:
                    print(f"Warning: Label for {index} is not 5 characters long. Skipping.")
                    continue
            except FileNotFoundError:
                print(f"Error: Label file {label_path} not found. Skipping sample.")
                continue

            # 2. Load and Preprocess Image
            image = cv2.imread(im_path)
            if image is None:
                print(f"Error: Could not load image from {im_path}. Skipping sample.")
                continue

            preprocessed_img = self._preprocess_image(image)

            # 3. Segment and Store Template
            for char_index in range(5):
                char_label = label[char_index]
                
                # We only need one good template for each of the 36 unique characters
                if char_label not in template_map:
                    segmented_img = self._segment_character(preprocessed_img, char_index)
                    template_map[char_label] = segmented_img
                    
        # 4. Save all generated templates
        print(f"Saving {len(template_map)} templates to '{self.templates_dir}'...")
        for char, template in template_map.items():
            save_path = os.path.join(self.templates_dir, f'{char}.jpg')
            cv2.imwrite(save_path, template)
            
        self._load_templates() # Reload templates after training
        print(f"--- Training Complete. {len(self.templates)} unique templates loaded. ---")


    def __call__(self, im_path, save_path):
        """
        Algo for inference: Load, Preprocess, Segment, Classify, Save.
        args:
            im_path: .jpg image path to load and to infer
            save_path: output file path to save the one-line outcome
        """
        if not self.templates:
            print("Inference failed: Templates not loaded. Please run 'train' method first.")
            return

        # 1. Load Image
        image = cv2.imread(im_path)
        if image is None:
            print(f"Error: Could not load image from {im_path}")
            return

        # 2. Preprocess
        preprocessed_img = self._preprocess_image(image)
        
        # 3. Segment and Classify
        captcha_text = ""
        for i in range(5): # 5 characters in the Captcha
            segmented_img = self._segment_character(preprocessed_img, i)
            char_prediction = self._classify_character(segmented_img)
            captcha_text += char_prediction
            
        # 4. Save Outcome
        try:
            with open(save_path, 'w') as f:
                f.write(captcha_text)
            print(f"Inferred CAPTCHA: **{captcha_text}**. Result saved to {save_path}")
        except Exception as e:
            print(f"Error saving output to {save_path}: {e}")

# Example Usage (for testing the implementation structure, assumes data is available)
if __name__ == '__main__':
    # You must have the 'sampleCaptchas/input' and 'sampleCaptchas/output' folders ready
    # and install: pip install opencv-python numpy
    
    # 1. Instantiate the solver
    solver = Captcha()
    
    # 2. Run the training phase once to generate the templates
    # This assumes the folder structure is:
    # .
    # ├── sampleCaptchas
    # │   ├── input
    # │   │   ├── input01.jpg, ...
    # │   └── output
    # │       ├── output01.txt, ...
    # └── templates (will be created)
    # └── captcha_solver.py 
    
    print("\n--- Running Training ---")
    solver.train(samples_base_dir='sampleCaptchas')
    
    # 3. Inference Example (You'd replace 'input01.jpg' with an unseen captcha image)
    print("\n--- Running Inference ---")
    test_image_path = 'sampleCaptchas/input/input100.jpg'
    output_file_path = 'prediction_test.txt'
    solver(im_path=test_image_path, save_path=output_file_path)
    