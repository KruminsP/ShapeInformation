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

def getSquareSideLength(input):
    x1 = input[0,0,0]
    y1 = input[0,0,1]
    x2 = input[1,0,0]
    y2 = input[1,0,1]
    x3 = input[2,0,0]
    y3 = input[2,0,1]
    x4 = input[3,0,0]
    y4 = input[3,0,1]

    A = math.sqrt((x2-x1)**2 + (y2-y1)**2)
    B = math.sqrt((x3-x2)**2 + (y3-y2)**2)
    C = math.sqrt((x4-x3)**2 + (y4-y3)**2)
    D = math.sqrt((x1-x4)**2 + (y1-y4)**2)
    return round((A+B+C+D)/4, 2)

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

print(f"number of shapes: {len(contours)}")

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
            cv2.putText(img, f"Square", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255,255,0))
            print(f"Square side length {getSquareSideLength(approx)}")
        else:
            cv2.putText(img, f"Rectangle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,255))
    elif len(approx) > 15:
        cv2.putText(img, "Circle", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))
        print('Circle')
        #print(approx)
        #print(cv2.minEnclosingCircle(approx))
    else:
        cv2.putText(img, "Other", (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,255,255))    

#cv2.imshow("result", img)
print("The size of the original image is", img.size)
