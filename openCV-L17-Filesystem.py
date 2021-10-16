import cv2
import os

print(cv2.__version__)
imageDir = '.\demoImages'

for root, dirs, files in os.walk(imageDir):
    print(f'Working folder: {root}')
    print(f'Directories under root: {dirs}')
    print(f'Files in root: {files}')
    for file in files:
        print(f"File is {os.path.join(root,file)}")
        print(f"File (no ext) is {os.path.join(root,os.path.splitext(file)[0])}")
