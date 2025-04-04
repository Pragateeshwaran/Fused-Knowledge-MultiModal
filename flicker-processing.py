import json
import os

def process_flickr_data(input_file, output_file):
    captions = []
    with open(input_file, 'r') as f:
        for line in f:
            parts = line.strip().split(",")
            image_id = parts[0]
            caption = ' '.join(parts[1:])
            captions.append((image_id, caption))
    caption = captions[1:]
    return caption

def image_with_captions(caption):
    image_path = os.path.join(os.getcwd(), 'assets/flicker/images')
    image_captions = {}
    for image_id, caption in caption:
        image_file = os.path.join(image_path, image_id)
        if os.path.exists(image_file):
            image_captions[image_file] = caption
    return image_captions


def json_create(image_captions, json_file):
    with open(json_file, 'w') as f:
        json.dump(image_captions, f)

captions = process_flickr_data('assets/flicker/captions.txt', 'assets/flicker/processed_captions.json')
image_captions = image_with_captions(captions)
json_file = 'assets/processed_captions.json'
json_create(image_captions, json_file)