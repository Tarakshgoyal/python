import os

# Get a list of files in your target directory
file_dir = "C:/Users/Taraksh Goyal/Desktop/coding/python/tensorflow/maestro-v2.0.0-midi/maestro-v2.0.0"  # replace with your actual path
filenames = os.listdir(file_dir)

# Check if the list has enough items
if len(filenames) > 1:
    sample_file = filenames[1]
    print(f"Sample file: {sample_file}")
else:
    print("No files found or not enough files in the directory.")
