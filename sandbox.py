import cv2
import json
import numpy
import math

def lengthSquare(X, Y):
    xDiff = X[0] - Y[0]
    yDiff = X[1] - Y[1]
    return xDiff * xDiff + yDiff * yDiff

def getAngles(A, B, C):
     # Square of lengths be a2, b2, c2
    a2 = lengthSquare(B, C)
    b2 = lengthSquare(A, C)
    c2 = lengthSquare(A, B)
 
    # length of sides be a, b, c
    a = math.sqrt(a2);
    b = math.sqrt(b2);
    c = math.sqrt(c2);
 
    # From Cosine law
    alpha = math.acos((b2 + c2 - a2) / (2 * b * c));
    betta = math.acos((a2 + c2 - b2) / (2 * a * c));
    gamma = math.acos((a2 + b2 - c2) / (2 * a * b));
 
    # Converting to degree
    alpha = alpha * 180 / math.pi;
    betta = betta * 180 / math.pi;
    gamma = gamma * 180 / math.pi;

    return [round(alpha, 1), round(betta, 1), round(gamma, 1)]

#inputs
img = cv2.imread("png_image.png")
camera_intrinsics = open("camera_intrinsics.json")

imgData = json.load(camera_intrinsics)
width = imgData['width']
height = imgData['height']
ffx = imgData['ffx']
ffy = imgData['ffy']
ppx = imgData['ppx']
ppy = imgData['ppy']
distortion_coeffs = imgData['distortion_coeffs']

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

edges = cv2.Canny(gray, 200,400)

contours, hierarchy= cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01*cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (255,0,0), 5)
    x = approx.ravel()[0]
    y = approx.ravel()[1] + 70
    if len(approx) == 3:
        cv2.putText(img, f"Triangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,0))
        print('Triangle angles ' + ', '.join(str(x) for x in getAngles(approx[0,0], approx[1,0], approx[2,0])))        
    elif len(approx) == 4:
        x1, y1, w, h = cv2.boundingRect(approx)
        aspectRatio = float(w)/h
        if aspectRatio >= 0.95 and aspectRatio <= 1.05:
            cv2.putText(img, f"Square, aspect ratio {round(aspectRatio,3)} side length = {h} {w}", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0))            
        else:
            cv2.putText(img, f"Rectangle, aspect ratio {round(aspectRatio,3)}", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255))
    elif len(approx) > 15:
        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))        
    else:
        cv2.putText(img, "Other", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))    

#cv2.imshow("result", img)
print(f"number of shapes: {len(contours)}")

print("The size of the original image is", img.size)
