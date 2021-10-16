import cv2
import face_recognition as fr
font = cv2.FONT_HERSHEY_SIMPLEX

# Known faces
donFaceRgb = fr.load_image_file('./demoImages/known/Donald Trump.jpg')
donFaceEncoding = fr.face_encodings(donFaceRgb)[0]

nancyFaceRgb = fr.load_image_file('./demoImages/known/Nancy Pelosi.jpg')
nancyFaceEncoding = fr.face_encodings(nancyFaceRgb)[0]

penceFaceRgb = fr.load_image_file('./demoImages/known/Mike Pence.jpg')
penceFaceEncoding = fr.face_encodings(penceFaceRgb)[0]

knownFaces = [donFaceEncoding, nancyFaceEncoding, penceFaceEncoding]
knownFaceNames = ['Donald Trump', 'Nancy Pelosi', 'Mike Pence']

# Unkown faces
unknownFaceRgb = fr.load_image_file('demoImages/unknown/u1.jpg')
unknownFaceBgr = cv2.cvtColor(unknownFaceRgb, cv2.COLOR_RGB2BGR)

unknownFaceLocations = fr.face_locations(unknownFaceRgb)
unknownFaceEncodings = fr.face_encodings(unknownFaceRgb, unknownFaceLocations)

for loc, encoding in zip(unknownFaceLocations, unknownFaceEncodings):
    top, right, bottom, left = loc
    cv2.rectangle(unknownFaceBgr, (left, top), (right, bottom), (255, 0, 0), 3)
    name = 'Unknown Person'
    matches = fr.compare_faces(knownFaces, encoding)
    if True in matches:
        idx = matches.index(True)
        name = knownFaceNames[idx]
    cv2.putText(unknownFaceBgr, name, (left, top - 10), font, .5, (255, 0, 0), 2)

cv2.imshow('Faces', unknownFaceBgr)

while True:
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
