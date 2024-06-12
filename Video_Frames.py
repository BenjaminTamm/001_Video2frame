import os
import csv
import datetime
import cv2
import torch
from glob import glob
from transformers import AutoModel, AutoTokenizer
from PIL import Image
from tqdm import tqdm

### Folder Path Config ###
# Specify the main folder containing the video folders

# main_folder = '../../Desktop/Seekr_Data/Video_Sources/'  
main_folder = './Data/Video_Sources/'  

# Specify the CSV file to store the processed information
# csv_file = '../../Desktop/Seekr_Data/processed_folders_log.csv'  
csv_file = './Data/processed_folders_log.csv'  
# csv_file_obj = './Data/processed_cleaned_images_log.csv'  

# Get the current date
current_date = datetime.date.today().strftime('%Y-%m-%d')

# Output folder path for extracted images to be saved
output_folder = './Data/Cleaned_Images/' + current_date
output_folder_obj = './Data/Object_Detection_Data/' + current_date


### Loading the VLM ###

# Download the VLM for the first time.

# model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
# model = model.to(device='cuda')

# tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
# model.eval()

# 
# output_folder = "./model"
# model.save_pretrained(output_folder)
# tokenizer.save_pretrained(output_folder)

model = AutoModel.from_pretrained('./model', 
                                  trust_remote_code=True, 
                                  torch_dtype=torch.float16,
                                  device_map="auto")

tokenizer = AutoTokenizer.from_pretrained('./model', trust_remote_code=True)
model.eval()
print("VLM loaded.")

promt = f"""

Please tell me if the following items are in the picture? Please answer: "Yes" or "No".

# RESPONSE #
Please answer in the following format: 1. Object: Yes or No

<List>
1. Dustbins:
2. Staircases:
3. Elevators:
4. Elevator request buttons:
5. Elevator signs:
6. Escalators:
7. MTR signs:
8. Handrails:
9. Bus stops:
10. Minibus stop poles:
11. Trams:
12. Toilet signs (Male, Female, Disabled):
13. Poles on the road:
14. Traffic Cones:
15. Telephone booths:
16. Tactile paving:
17. Ramp access:
18. Switches/ Switch Boards:

</list>

"""


# Please only answer one of the follwing words: "None", "Contain", "Blur".
### Helper Function ###
def save_target_frame(image, frame_path):
    msgs = [{'role': 'user', 'content': promt}]
    res = model.chat(image = image, msgs = msgs, tokenizer=tokenizer)
    # print("save_target_frame: ", res)

    if not os.path.exists(output_folder_obj):
        os.makedirs(output_folder_obj)

    if "Yes" in res:
        save_path = f'{output_folder_obj}/obj-' + frame_path.split('\\')[-1]
        print("Object detection save path: ", save_path)
        image.save(save_path)  # Save the image

def extract_frames(video_path, output_folder):
    """
    All the saved frames can be use for image captioning.

    Further filtering can be done to extract only the frames that are needed 
    for object detection.
    """
    if not os.path.exists(output_folder):
        # Create the directory
        os.makedirs(output_folder)
    print(output_folder)
    # Open the video file
    video = cv2.VideoCapture(video_path)
    video_name = video_path.split("\\")[-1]
    # Check if the video is valid
    if not video.isOpened():
        print("Error opening video")
        exit()
        
    # Define the frame rate and interval for saving frames
    # fps = 30
    interval = 30 * 3 # 30 stands for 1 s
    # Initialize the frame counter and the image saver
    frame_count = 0
    image_save_count = 0
    
    success, frame = video.read()
    while video.isOpened():
        success, frame = video.read()
        
        if not success:
            break
            
        # Increment the frame counter and save the frame to the image saver
        frame_count += 1
        # Check if it's time to save a new frame
        if frame_count % interval == 0:

            image_save_count +=1
            # Save the frame as an image
            frame_path = os.path.join(output_folder, f"{video_name}-{frame_count}-{image_save_count}.jpg")
            print(f"Saving to {frame_path}")
            cv2.imwrite(frame_path, frame)
            # print("type(frame)", type(frame))
            # break
            frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(frame2, mode='RGB')
            # print("type(image)", type(image))

            save_target_frame(image, frame_path)

        # Read the next frame
        success, frame = video.read()

    # Release the video object
    video.release()




### Check if the folder is processed before ###
processed_folders = []
if os.path.exists(csv_file):
    with open(csv_file, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            processed_folders.append(row['Folder_name'])

### Iterate through the folders in the main folder (Video folder) ###
for folder in os.listdir(main_folder):
    folder_path = os.path.join(main_folder, folder)

    # Check if the item in the main folder is a video folder and not processed before
    if os.path.isdir(folder_path) and folder not in processed_folders:
        print(f"Processing video folder: {folder}")
        
        # Initialize the list to store the processed video files
        processed_files = []

        # Iterate through the video files in the video folder
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)

            # Check if the file is a video
            if file.endswith(('.mp4', '.avi', '.mkv', '.mov', '.MP4')):
                print(f"Processing video file: {file}")

                # Extract frames from the video file
                extract_frames(file_path, output_folder)

                # Add the processed video file to the list
                processed_files.append(file)

        # Update the CSV file with the processed information
        with open(csv_file, 'a', newline='') as file:
            writer = csv.writer(file)
            
            # Check if the file is empty and write the header row
            if file.tell() == 0:
                writer.writerow(['Folder_name', 'Video_files_processed', 'Folder_save_as'])
            
            # Write the processed rows
            for file in processed_files:
                
                writer.writerow([folder, file, current_date])
        print(f"Video folder {folder} processed and CSV file updated.")
