import os 
import cv2

def process_image(image_folder):
    
    for filename in os.listdir(image_folder):
        image_path = os.path.join(image_folder, filename)
        image = cv2.imread(image_path)
        processed_image = cv2.resize(image, (256, 256))  
        print(f"Processing image: {filename}")      
        cv2.imwrite(image_path, processed_image)
    print("Image processing complete.")

process_image('assets/flicker/images')
    

