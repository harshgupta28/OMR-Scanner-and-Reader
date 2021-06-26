import cv2
import numpy as np

## TO DISPLAY AN ARRAY OF IMAGES IN A SINGLE WINDOW
def DisplayImg(arr,scl,lbl=[]):
    r = len(arr)
    c = len(arr[0])
    rAvailable = isinstance(arr[0], list)
    w = arr[0][0].shape[1]
    h = arr[0][0].shape[0]
    if rAvailable:
        for x in range ( 0, r):
            for y in range(0, c):
                arr[x][y] = cv2.resize(arr[x][y], (0, 0), None, scl, scl)
                if len(arr[x][y].shape) == 2: arr[x][y]= cv2.cvtColor( arr[x][y], cv2.COLOR_GRAY2BGR)
        emptyImg = np.zeros((h, w, 3), np.uint8)
        hor = [emptyImg]*r
        hor_con = [emptyImg]*r
        for x in range(0, r):
            hor[x] = np.hstack(arr[x])
            hor_con[x] = np.concatenate(arr[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, r):
            arr[x] = cv2.resize(arr[x], (0, 0), None, scl, scl)
            if len(arr[x].shape) == 2: arr[x] = cv2.cvtColor(arr[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(arr)
        hor_con= np.concatenate(arr)
        ver = hor
    if len(lbl) != 0:
        eachImgw= int(ver.shape[1] / c)
        eachImgh = int(ver.shape[0] / r)
        for d in range(0, r):
            for c in range (0,c):
                cv2.rectangle(ver,(c*eachImgw,eachImgh*d),(c*eachImgw+len(lbl[d][c])*13+27,30+eachImgh*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,lbl[d][c],(eachImgw*c+10,eachImgh*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    return ver

def reorder(points):

    points = points.reshape((4, 2)) # REMOVE EXTRA BRACKET
    # print(points)
    pointsNew = np.zeros((4, 1, 2), np.int32) # NEW MATRIX WITH ARRANGED POINTS
    add = points.sum(1)
    print(add)
    print(np.argmax(add))
    pointsNew[0] = points[np.argmin(add)]  #[0,0]
    pointsNew[3] =points[np.argmax(add)]   #[w,h]
    diff = np.diff(points, axis=1)
    pointsNew[1] =points[np.argmin(diff)]  #[w,0]
    pointsNew[2] = points[np.argmax(diff)] #[h,0]

    return pointsNew

def rectContour(boundary):

    rectContour = []
    max_area = 0
    for i in boundary:
        area = cv2.contourArea(i)
        if area > 50:
            perimeter = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
            if len(approx) == 4:
                rectContour.append(i)
    rectContour = sorted(rectContour, key=cv2.contourArea,reverse=True)
    #print(len(rectCon))
    return rectContour

def getCornerPoints(points):
    perimeter = cv2.arcLength(points, True) # LENGTH OF CONTOUR
    approx = cv2.approxPolyDP(points, 0.02 * perimeter, True) # APPROXIMATE THE POLY TO GET CORNER POINTS
    return approx

def splitBoxes(img,ques,options):
    r = np.vsplit(img,ques)
    # cv2.imshow("Split",r[0])
    boxes=[]
    for r in r:
        c= np.hsplit(r,options)
        for box in c:
            # cv2.imshow("Img",box)
            boxes.append(box)
    return boxes

def drawGrid(img,ques,options):
    secW = int(img.shape[1]/ques)
    secH = int(img.shape[0]/options)
    for i in range (0,9):
        pt1 = (0,secW*i)
        pt2 = (img.shape[1],secW*i)
        pt3 = (secH * i, 0)
        pt4 = (secH*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)

    return img

def showAnswers(img,myIndex,grading,ans,ques,options):
    secW = int(img.shape[1]/ques)
    secH = int(img.shape[0]/options)

    for x in range(0,ques):
        myAns= myIndex[x]
        cX = (myAns * secH) + secH // 2
        cY = (x * secW) + secW // 2
        if grading[x]==1:
            myColor = (0,255,0)
            #cv2.rectangle(img,(myAns*secW,x*secH),((myAns*secW)+secW,(x*secH)+secH),myColor,cv2.FILLED)
            cv2.circle(img,(cX,cY),50,myColor,cv2.FILLED)
        else:
            myColor = (0,0,255)
            #cv2.rectangle(img, (myAns * secW, x * secH), ((myAns * secW) + secW, (x * secH) + secH), myColor, cv2.FILLED)
            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)

            # CORRECT ANSWER
            myColor = (0, 100, 0)
            correctAns = ans[x]
            cv2.circle(img,((correctAns * secW)+secW//2, (x * secH)+secH//2),
            50,myColor,cv2.FILLED)
