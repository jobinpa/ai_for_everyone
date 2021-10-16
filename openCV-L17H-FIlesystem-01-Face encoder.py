import os
import face_recognition as fr
from pathlib import Path

IMG_DIR = './demoImages/known'
IMG_EXT = '*.jpg'
ENCODING_DIR = './faceenc'
ENCODING_JITTERS = 1

FILEMODE_WRITE_BINARY = 'wb'

print('Work started')

if not os.path.isdir(IMG_DIR):
    raise ValueError(f'Directory "{IMG_DIR}"" was not found.')

if not os.path.isdir(ENCODING_DIR):
    os.makedirs(ENCODING_DIR)

imgDirResolved = Path(IMG_DIR).resolve()
encDirResolved = Path(ENCODING_DIR).resolve()

print(f'   Image directory is "{imgDirResolved}"')
print(f'   Face encoding directory is "{encDirResolved}"')

for filePath in imgDirResolved.glob(IMG_EXT):
    filePath = filePath.resolve()
    print(f'   > Processing image "{filePath}""...')

    fileName = os.path.basename(filePath)

    fileNameExtIdx = fileName.rfind('.')
    if fileNameExtIdx >= 0:
        fileName = fileName[:fileNameExtIdx]

    imgRgb = fr.load_image_file(filePath)
    imgFaceEncodings = fr.face_encodings(imgRgb, num_jitters=ENCODING_JITTERS)

    if len(imgFaceEncodings) != 1:
        raise ValueError(f'Image "{filePath}" contains {len(imgFaceEncodings)} face(s).')

    file = open(os.path.join(encDirResolved, fileName + '.bin'), FILEMODE_WRITE_BINARY)
    file.write(imgFaceEncodings[0].tobytes())
    file.close()

print('Work done!')
