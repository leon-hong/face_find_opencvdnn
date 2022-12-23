import cv2
import cvlib
import os

dirout = 'd:/temp/result'
img = cv2.imread('d:/temp/819531.jpg')
h, w, c = img.shape

os.makedirs(dirout, exist_ok=True)

# threshold : 정확도 컷트라인(0.5는 검출 실패하는 경우가 있어서 0.3으로 하양)
faces, confidences = cvlib.detect_face(img, threshold=0.3)

idx = 0
margin = 10
for (x, y, x2, y2), conf in zip(faces, confidences):
    x = max(0, x - margin)
    y = max(0, y - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    roi = img[y:y2, x:x2]
    imgOut = roi.copy()
    cv2.imwrite(dirout + '/' + str(idx) + '.png', imgOut)
    idx += 1

    # cv2.putText(img, str(conf), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1)
    #cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)

# cv2.imshow('org', img)
# key = cv2.waitKey(0)
