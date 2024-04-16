import os
from PIL import Image

current_script_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_script_path))

input_dir = os.path.join(root_path, "data", "boneage-test-dataset", "boneage-test-dataset")
output_dir = os.path.join(root_path, "data", "test")
target_size = 300

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over all files in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        # Open the image
        image_path = os.path.join(input_dir, filename)
        image = Image.open(image_path)

        # Calculate the new dimensions while preserving aspect ratio
        width, height = image.size
        
        new_width = target_size
        new_height = target_size
        # if width > height:
        #     new_width = target_size
        #     new_height = int(height * (target_size / width))
        # else:
        #     new_width = int(width * (target_size / height))
        #     new_height = target_size

        print("Resizing", filename, "from", image.size, "to", (new_width, new_height))
        # Resize the image
        resized_image = image.resize((new_width, new_height))

        # Save the resized image to the output directory
        output_path = os.path.join(output_dir, filename)
        resized_image.save(output_path)


