from PIL import Image
import matplotlib.pyplot as plt
import os
from datetime import datetime

def show_data(data_sample, loop, class_name):
    for i in range(loop):
        image = Image.open(data_sample[i])
        plt.imshow(image)
        plt.title(f"y = {class_name}")
        plt.show()

# Specify your local path to the dataset
directory = "C://Wildfire-Prediction-from-Satellite-Imagery-main//data//train"
wildfire = 'wildfire'
nowildfire = 'nowildfire'

# Paths to wildfire and non-wildfire images
wildfire_file_path = os.path.join(directory, wildfire)
wildfire_files = [os.path.join(wildfire_file_path, file) for file in os.listdir(wildfire_file_path) if file.endswith(".jpg")]
wildfire_files.sort()

nowildfire_file_path = os.path.join(directory, nowildfire)
nowildfire_files = [os.path.join(nowildfire_file_path, file) for file in os.listdir(nowildfire_file_path) if file.endswith(".jpg")]
nowildfire_files.sort()

# Display some images from each category
show_data(nowildfire_files, 3, 'No Wildfire')
show_data(wildfire_files, 3, 'Wildfire')

# Display timestamp of changes
print(f"Changes have been made to the project on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
