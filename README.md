# AI for Everyone

This my personal repository for the most excellent
[AI for Everyone](https://www.youtube.com/watch?v=gD_HWj_hvbo&list=PLGs0VKk2DiYyXlbJVaE8y1qr24YldYNDm)
lessons published by Paul McWhorter.

__IMPORTANT:__
The device ID of my camera is 1. If you get a runtime error, try to set the
camera ID to 0. Most of the files have a CAM_ID parameter on the top however,
for those without this parameter, look for `cv2.VideoCapture` and update the
first argument.

The lessons have been designed around [Python 3.6.6](https://www.python.org/downloads/release/python-366/).

| Package          | Version  | Notes                     |
| :--------------- | -------: | :------------------------ |
| opencv-python    | 4.5.3.56 |                           | 
| cmake            | 3.21.3   | Required to compile dlib. |
| dlib             | 19.22.1  |                           |
| face-recognition | 1.3.0    |                           |
| mediapipe        | 0.8.3    |                           |

Visual Studio may be required to compile `dlib` on Windows. The
[Community edition](https://visualstudio.microsoft.com/vs/community/) is free.
Please make sure you select the `Desktop development with C++` workload during the
installation.