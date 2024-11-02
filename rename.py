import os
from shutil import copy2

def rename_images(src_folder, dest_folder, prefix="image"):
    # Ensure the destination folder exists
    os.makedirs(dest_folder, exist_ok=True)
    
    # Loop through each file in the source folder
    for idx, filename in enumerate(os.listdir(src_folder)):
        # Check if the file is an image (you may adjust the extensions as needed)
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            # Define the source and destination file paths
            src_path = os.path.join(src_folder, filename)
            new_name = f"{prefix}_{idx + 1}.jpg"
            dest_path = os.path.join(dest_folder, new_name)
            
            # Copy the file to the destination folder with the new name
            copy2(src_path, dest_path)
            print(f"Renamed {filename} to {new_name}")


if __name__ == "__main__":
    rename_images("data/random", "renamed_data/random", prefix="random")