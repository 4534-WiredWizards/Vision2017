import numpy as np
import cv2
import math
import random
from networktables import NetworkTable
import socket

logfile = open('log.txt', 'w')

addr = "10.45.34.12"
try:
    hostname = "roborio-4534-frc.local"
    addr = socket.gethostbyname(hostname)
    logfile.write(addr)
except:
    pass

NetworkTable.setIPAddress(addr)
NetworkTable.setClientMode()
NetworkTable.initialize()

logfile.write("Initialized")

table = NetworkTable.getTable('vision')

### GLOBALS ###

displayThreshold = False
### END GLOBALS ###
### CONSTANTS ###
# Camera's physical height (inches) above ground
boilerCameraAltitude = 24
gearCameraAltitude = 5

# Camera's Angle (degrees) above horizontal
gearCameraAngle = 1
boilerCameraAngle = 30

# Height of center of target (inches) above ground
boilerTargetAltitude = 83
gearTargetAltitude = 13.25

# Difference between camera hieght and target height, useful for the trig.
boilerDeltaAltitude = boilerTargetAltitude - boilerCameraAltitude
boilerDeltaAltitude = gearTargetAltitude - gearCameraAltitude

# vertical size (inches) of vision target.
boilerTargetHeight = 6
gearTargetHeight = 5

gearTargetDist = 10.25
gearTargetWidth = 2

# Horizontal distance (inches) between the camera and the center of the robot.
gearCameraOffset = 9.7


# angle function values
#angleFunc1A = -90.535724570955
#angleFunc1B = 45.247456281206
#angleFunc2A = -5.511621418651
#angleFunc2B = 24.318446167847
# distance function values
estimateDistanceSlope = 0.78818915774652
estimateDistanceIntercept = 4.637193403987
#distFunc2A = -0.1414458147376
#distFunc2B = -0.0675841969269

### END CONSTANTS ###
### CALIBRATION ###
# camera calibration
cameraRMS = 0.283286598231
cameraMatrix = np.float32([[1.12033194e+03, 0.0, 6.49786694e+02],
                           [0.0, 1.11455896e+03, 3.80918277e+02],
                           [0.0, 0.0, 1.0]])
cameraDistortion = np.float32([0.15190902, -0.78835469, 0.00402702, -0.00291226, -1.00032999])
# color calibration
gearCalibrationTuple = ((64, 213, 66), (74, 255, 105), (14, 66, 0), (42, 105, 13))
gearCalLowHSV, gearCalHighHSV, gearCalLowBGR, gearCalHighBGR = gearCalibrationTuple
boilerCalibrationTuple = ((75, 73, 98), (81, 164, 175), (75, 98, 35), (148, 175, 95))
boilerCalLowHSV, boilerCalHighHSV, boilerCalLowBGR, boilerCalHighBGR = boilerCalibrationTuple

# exposure
exposure = -9

cameraWidth = 580
### END CALIBRATION ###
### FUNCTIONS ###

# a clicking function, simply toggles the binary displayThreshold variable
def clickFunc(evt,x,y,flags,param):
    global displayThreshold

    if evt == cv2.EVENT_LBUTTONDOWN:
        displayThreshold = not displayThreshold

def findCorners2(contour):
    global w,h

    topLeftDistance = 9999999
    topRightDistance = 9999999
    bottomRightDistance = 9999999
    bottomLeftDistance = 9999999
    topLeft = topLeftOrigin = (0,0)
    topRight = topRightOrigin = (w,0)
    bottomRight = bottomRightOrigin = (w,h)
    bottomLeft = bottomLeftOrigin = (0,h)

    for pt2 in contour:
        x,y = pt2
        pt = (x,y)

        if distance(topLeftOrigin,pt) < topLeftDistance:
            topLeftDistance = distance(topLeftOrigin,pt)
            topLeft = pt

        if distance(topRightOrigin,pt) < topRightDistance:
            topRightDistance = distance(topRightOrigin,pt)
            topRight = pt

        if distance(bottomRightOrigin,pt) < bottomRightDistance:
            bottomRightDistance = distance(bottomRightOrigin,pt)
            bottomRight = pt

        if distance(bottomLeftOrigin,pt) < bottomLeftDistance:
            bottomLeftDistance = distance(bottomLeftOrigin,pt)
            bottomLeft = pt

    return (topLeft,topRight,bottomRight,bottomLeft)

def findExtremePoints(contour):
    leftmost = tuple(contour[contour[:,:,0].argmin()][0])
    rightmost = tuple(contour[contour[:,:,0].argmax()][0])
    topmost = tuple(contour[contour[:,:,1].argmin()][0])
    bottommost = tuple(contour[contour[:,:,1].argmax()][0])

    return (leftmost,topmost,bottommost,rightmost)

def calculateAngle(before, point, after):
    a = distance(point,after)
    b = distance(before,after)
    c = distance(before,point)

    a2 = a**2
    b2 = b**2
    c2 = c**2

    n2ac = -2 * a * c

    b2ma2mc2 = b2 - a2 - c2

    B = 1/math.cos(b2ma2mc2/n2ac)

    return math.degrees(B)

def calculateCenter(centerX, width):
    widthCenter = width / 2.0
    centerX = centerX * 1.0
    return (centerX - widthCenter) / widthCenter

def pixelsOffCenter(location, dimension):
    #determines center of dimension, then returns the number of pixels location is away from that center.
    dimensionCenter = dimension / 2
    return location - (dimensionCenter) # Negative values mean left/upwards of center.

def pixelsToInches(inputPixels, refPixels, refInches):
    #converts pixels to inches linear to refInches / refPixels.
    inputPixels = inputPixels * 1.0
    refPixels = refPixels * 1.0
    refInches = refInches * 1.0
    return inputPixels * refInches / refPixels

def simplifyContour(contour):

    # out = [None] * len(contour)
    #
    # for i in xrange(len(contour)):
    #     before = (contour[len(contour)-1][0] if i==0 else contour[i-1][0])
    #     point = contour[i][0]
    #     after = (contour[0][0] if i == len(contour)-1 else contour[i+1][0])
    #
    #     angle = calculateAngle(before, point, after)
    #     out[i] = contour[i][0].tolist() if angle < 90 else None
    #
    # def remFunc(item):
    #     return not item is None
    #
    # out = filter(remFunc, out)
    #
    # # copy out
    # cout = list(out)
    #
    # # remove similar points
    # # TODO: rewrite this to be more intelligent
    # for i in xrange(len(out)):
    #     if i == 0:
    #         continue;
    #
    #     oy,ox = out[i-1]
    #     y,x = out[i]
    #
    #     if(abs(oy-y) < 40 and abs(ox-x) < 40):
    #         cout[i] = None
    #
    # return np.array(filter(remFunc, cout))
    rect = cv2.minAreaRect(contour)
    box = cv2.cv.BoxPoints(rect)
    box = np.int0(box)
    return box

# draws a point on screen
def drawPoint(img,pt,color=(0,255,0)):
    # pt is (y,x)
    cv2.circle(img,pt,4,color,-1)

# mathematical distance
def distance(p0,p1):
    return math.sqrt((p0[0] - p1[0])**2 + (p0[1] - p1[1])**2)

# computes the 2d transform matrix
def findTransform(contour,corners):
    global w,h

    # now that we have our rectangle of points, let's compute
    # the width of our new image
    (tl, tr, br, bl) = corners
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

    # ...and now for the height of our new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

    # take the maximum of the width and height values to reach
    # our final dimensions
    maxWidth = max(int(widthA), int(widthB))
    maxHeight = max(int(heightA), int(heightB))

    # construct our destination points
    dst = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype = "float32")

    # calculate the perspective transform matrix and warp
    # the perspective to grab the screen
    M = cv2.getPerspectiveTransform(np.array(corners, dtype="float32"), dst)

    return (M,maxWidth,maxHeight)

# gives a vector of x,y,z rotations from a 3x3 rotation matrix
def mat2euler(M, cy_thresh=None):
    ''' Discover Euler angle vector from 3x3 matrix

    Uses the conventions above.

    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
       threshold below which to give up on straightforward arctan for
       estimating x rotation.  If None (default), estimate from
       precision of input.

    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
       Rotations in radians around z, y, x axes, respectively

    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::

      [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
      [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
      [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]

    with the obvious derivations for z, y, and x

       z = atan2(-r12, r11)
       y = asin(r13)
       x = atan2(-r23, r33)

    Problems arise when cos(y) is close to zero, because both of::

       z = atan2(cos(y)*sin(z), cos(y)*cos(z))
       x = atan2(cos(y)*sin(x), cos(x)*cos(y))

    will be close to atan2(0, 0), and highly unstable.

    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:

    See: http://www.graphicsgems.org/

    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    M = np.asarray(M)
    if cy_thresh is None:
        try:
            cy_thresh = np.finfo(M.dtype).eps * 4
        except ValueError:
            cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.flat
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = math.sqrt(r33*r33 + r23*r23)
    if cy > cy_thresh: # cos(y) not close to zero, standard form
        z = math.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = math.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
    else: # cos(y) (close to) zero, so x -> 0.0 (see above)
        # so r21 -> sin(z), r22 -> cos(z) and
        z = math.atan2(r21,  r22)
        y = math.atan2(r13,  cy) # atan2(sin(y), cy)
        x = 0.0
    return z, y, x

# # estimates angle when board tilted far end left
# def estimateAngleFunction1(thetaY):
#     result = 0
#
#     result += angleFunc1A
#
#     try:
#         result += angleFunc1B*math.log(thetaY)
#     except ValueError:
#         result = -999
#
#     return result
#
# # estimates angle when board tilted far end right
# def estimateAngleFunction2(thetaY):
#     thetaY = -thetaY
#     thetaY += 7
#
#     result = 0
#
#     result += angleFunc2A
#
#     try:
#         result += angleFunc2B*math.log(thetaY)
#     except ValueError:
#         result = -999
#
#     # subtract for correction
#     try:
#         result -= math.log(thetaY)
#     except ValueError:
#         result = -999
#
#     return result
def calculateLateralAngle(offCenterX, distance):
    return np.arcsin(offCenterX / distance)

def estimateDistance(pixelHeight):
    return (estimateDistanceSlope * pixelHeight) + estimateDistanceIntercept

def calculateDistanceToTarget(inchesOffCenterY):
    return (deltaAltitude + inchesOffCenterY) / np.tan(cameraAngle)

# def estimateDistanceFunction1(translationZ):
#     result = 0
#     result += distFunc1A * translationZ
#     result += distFunc1B
#     return result
#
# def estimateDistanceFunction2(translationY):
#     result = 0
#     result += distFunc2A * translationY
#     result += distFunc2B
#     return result


def countCameras():
    ret = 5
    for i in range(0,5):
        tempCam = cv2.VideoCapture(i)
        res = tempCam.isOpened()
        tempCam.release()
        print i
        if res is True:
            ret = i-1
    print ret
    return ret

def arbitrateValue(v1,v2):
    if v1 == -999:
        return v2
    if v2 == -999:
        return -v1
    if v2 > v1:
        return -v1
    if v1 > v2:
        return v2
    return -999

def boilerLocate(frame):
    dist = -777
    centerValue = -777
    angle = -777
    try:
    # Our operations on the frame come here
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    except cv2.error:
        print "Camera not found. Exiting..."
        exit(1)
    cv2.imwrite('frame.jpg', frame)
    cv2.imwrite('hsv.jpg', hsv)

    # get the calibration values in
    hsvLow = np.array(list(boilerCalLowHSV))
    hsvHigh = np.array(list(boilerCalHighHSV))
    bgrLow = np.array(list(boilerCalLowBGR))
    bgrHigh = np.array(list(boilerCalHighBGR))

    # use the calibration values to mask out what we want
    mask = cv2.inRange(hsv, hsvLow, hsvHigh)
    mask2 = cv2.inRange(frame, bgrLow, bgrHigh)

    # combine the masks
    bw = cv2.bitwise_and(mask,mask2)

    # dilate the image to simplify small black bits
    bw = cv2.dilate(bw, None, None, None, 3)

    # contour it
    contours, hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # find the largest area contour
    largestArea = 0
    secondLargestArea = 0
    largestAreaIndex = -1
    secondLargestAreaIndex = -1
    for i in xrange(len(contours)):
        contours[i] = cv2.convexHull(contours[i])
        area = cv2.contourArea(contours[i])
        if area > largestArea:
            secondLargestArea = largestArea
            secondLargestAreaIndex = largestAreaIndex
            largestArea = area
            largestAreaIndex = i
        elif area > secondLargestArea:
            secondLargestArea = area
            secondLargestAreaIndex = i

    # if we have a largest area
    if(largestAreaIndex > -1):
        # get the contour
        topContour = contours[largestAreaIndex]
        bottomContour = contours[secondLargestAreaIndex]
        # simplify it (try to get it to 4 corners)
        topBox = simplifyContour(topContour)
        bottomBox = simplifyContour(bottomContour)

        # draw it on screen
        cv2.drawContours(frame, [topBox], -1, (16,16,223), 3)
        cv2.drawContours(frame, [bottomBox], -1, (0,0,255), 3)

        # if it has 4 corners
        if len(topBox) == 4 and len(bottomBox) == 4:
            # find the corners
            # fc2 was just the most reliable IMO -sam
            #aTop = topBox[0]
            #bTop = topBox[1]
            #cTop = topBox[2]
            #dTop = topBox[3]
            #aTop, bTop, cTop, dTop = cv2.cv.BoxPoints(topBox)

            aTop,bTop,cTop,dTop = findCorners2(topBox)
            aBottom,bBottom,cBottom,dBottom = findCorners2(bottomBox)

            #print (a,b,c,d)
            # draw the corners
            drawPoint(frame, aTop, (0,255,0))
            drawPoint(frame, bTop, (0,255,0))
            drawPoint(frame, cTop, (255,255,255))
            drawPoint(frame, dTop, (0,255,255))
            drawPoint(frame, aBottom, (255,0,0))
            drawPoint(frame, bBottom, (255,0,0))
            drawPoint(frame, cBottom, (255,255,255))
            drawPoint(frame, dBottom, (255,0,255))


            # # find the 2D transform
            # MTop,mwTop,mhTop = findTransform(topBox,(aTop,bTop,cTop,dTop))
            # MBottom,mwBottom,mhBottom = findTransform(bottomBox,(aBottom,bBottom,cBottom,dBottom))
            #
            # # transform and get the transformed image
            # bwTop = cv2.warpPerspective(frame,MTop,(mwTop,mhTop))#(int(w),int(h)))
            # bwBottom = cv2.warpPerspective(frame,MBottom,(mwBottom,mhBottom))#(int(w),int(h)))

            #centerXTop = (aTop[0] + bTop[0] + cTop[0] + dTop[0]) / 4
            #centerXBottom = (aBottom[0] + bBottom[0] + cBottom[0] + dBottom[0]) / 4
            #centerYTop = (aTop[1] + bTop[1] + cTop[1] + dTop[1]) / 4
            #centerYBottom = (aBottom[1] + bBottom[1] + cBottom[1] + dBottom[1]) / 4
            centerXTarget = (cTop[0] + dTop[0] + cBottom[0] + dBottom[0]) / 4
            centerYTarget = (cTop[1] + dTop[1] + cBottom[1] + dBottom[1]) / 4
            drawPoint(frame, (int(centerXTarget), int(centerYTarget)), (0,0,0))
            drawPoint(frame, (int(w/2), int(centerYTarget)), (0,0,0))
            leftPixelHeight = distance(cTop, cBottom)
            rightPixelHeight = distance(dTop, dBottom)
            pixelHeight = (leftPixelHeight + rightPixelHeight) / 2

            # 2d points representation of the object on screen in pixels
            # (y,x)
            #imagePoints = np.array([cTop2,dTop2,cBottom2,dBottom2],dtype = "float32")

            # 3d points representation of the object in (y,x,z)
            #objectPoints = np.float32([[3,-6.5,0],[3,6.5,0],[-3,-6.5,0],[-3,6.5,0]])
            #objectPoints = np.float32([[-7.5,3,0],[7.5,3,0],[-7.5,-3,0],[7.5,-3,0]])

            # the hardest math: find the 3d rotation and translation vectors
            # of a known size 3d plane based on screen coordinates
            #ret,rvec,tvec = cv2.solvePnP(objectPoints,imagePoints,cameraMatrix,cameraDistortion,)

            # calculate to rotation matrix
            # don't ask me these weird names, we only want rM
            #rM, jacobian = cv2.Rodrigues(rvec)

            # get the xtheta, ytheta, and ztheta values
            #xTheta,yTheta,zTheta = mat2euler(rM)

            # move to degrees
            #xTheta = math.degrees(xTheta)
            #yTheta = math.degrees(yTheta)
            #zTheta = math.degrees(zTheta)

            # print them out
            #print (xTheta,yTheta,zTheta)

            # return the point as a decimal of the frame's dimensions

            #pixelsOffCenterY = pixelsOffCenter(centerYTarget, h)
            #inchesOffCenterY = pixelsToInches(pixelsOffCenterY, pixelHeight, targetHeight)
            #dist = (deltaAltitude + inchesOffCenterY) / np.tan(cameraAngle)
            #distZ = estimateDistanceFunction1(tvec[2][0])
            #distY = estimateDistanceFunction2(tvec[1][0])
            #distFt = math.sqrt(distZ ** 2 + distY ** 2)
            #print "dist z:", distZ
            #print "dist y:", distY
            #print "dist ft:", distFt

            pixelsOffCenterX = pixelsOffCenter(centerXTarget, w)
            inchesOffCenterX = pixelsToInches(pixelsOffCenterX, pixelHeight, boilerTargetHeight)
            #angle = calculateLateralAngle(inchesOffCenterX, dist)
            #angle = arbitrateValue(estimateAngleFunction1(yTheta),estimateAngleFunction2(yTheta))
            #dist = distFt * 12
            if(pixelHeight != 0):
                dist = (boilerTargetHeight * h) / (2 * pixelHeight * math.tan(0.418224329) * 0.6305) # the angle  is a constant: the tangent of half of the camera's field of view angle.
            angle = math.atan2(inchesOffCenterX , dist)
            angle = math.degrees(angle)

            centerValue = calculateCenter(centerXTarget, cameraWidth)
            centerValue = inchesOffCenterX

    return dist, angle, centerValue

def gearLocate(frame):
    dist = -777
    angle = -777
    centerValue = -777
    ratio = -777
    try:
    # Our operations on the frame come here
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    except cv2.error:
        print "Camera not found. Exiting..."
        exit(1)

    cv2.imwrite('frame.jpg', frame)
    cv2.imwrite('hsv.jpg', hsv)

    # get the calibration values in
    hsvLow = np.array(list(gearCalLowHSV))
    hsvHigh = np.array(list(gearCalHighHSV))
    bgrLow = np.array(list(gearCalLowBGR))
    bgrHigh = np.array(list(gearCalHighBGR))

    # use the calibration values to mask out what we want
    mask = cv2.inRange(hsv, hsvLow, hsvHigh)
    mask2 = cv2.inRange(frame, bgrLow, bgrHigh)

    # combine the masks
    bw = cv2.bitwise_and(mask,mask2)

    # dilate the image to simplify small black bits
    bw = cv2.dilate(bw, None, None, None, 3)

    # contour it
    contours, hierarchy = cv2.findContours(bw,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # find the largest area contour
    largestArea = 0
    secondLargestArea = 0
    largestAreaIndex = -1
    secondLargestAreaIndex = -1
    for i in xrange(len(contours)):
        contours[i] = cv2.convexHull(contours[i])
        area = cv2.contourArea(contours[i])
        if area > largestArea:
            secondLargestArea = largestArea
            secondLargestAreaIndex = largestAreaIndex
            largestArea = area
            largestAreaIndex = i
        elif area > secondLargestArea:
            secondLargestArea = area
            secondLargestAreaIndex = i
    #print "largestAreaIndex:", largestAreaIndex
    #print "secondLargestAreaIndex:", secondLargestAreaIndex

    # if we have a largest area
    if(largestAreaIndex > -1):
        #print "largestArea:", largestArea
        #print "secondLargestArea:", secondLargestArea
        # get the contour
        nearContour = contours[largestAreaIndex]
        farContour = contours[secondLargestAreaIndex]
        #print "Number of corners (Before):", len(nearContour)
        #print "Number of corners (Far Before):", len(farContour)
        cv2.drawContours(frame, [nearContour], -1, (255,0,0),3)
        # simplify it (try to get it to 4 corners)
        nearBox = simplifyContour(nearContour)
        cv2.drawContours(frame, [nearContour], -1, (127,127,0),3)
        farBox = simplifyContour(farContour)

        #print "Near Contour Area:" , cv2.contourArea(nearContour)
        #print "Far Contour Area:" , cv2.contourArea(farContour)
        #print "Near Box Area:" , cv2.contourArea(nearBox)
        #print "Far Box Area:" , cv2.contourArea(farBox)

        if ((cv2.contourArea(farBox) > cv2.contourArea(nearBox) * 0.75) and (secondLargestAreaIndex != -1)):
            # if the two contours are within 90% of each other in area, work with both.
            #print "Using two Contours:"
            #print "Dist:", dist
            # draw it on screen
            cv2.drawContours(frame, [nearBox], -1, (16,16,223), 3)
            cv2.drawContours(frame, [farBox], -1, (0,0,255), 3)

            # if it has 4 corners
            #print "number of corners:", len(nearBox), len(farBox)
            if (len(nearBox) == 4) and (len(farBox) == 4):
                # find the corners
                aNear,bNear,cNear,dNear = findCorners2(nearBox)
                aFar,bFar,cFar,dFar = findCorners2(farBox)

                # draw the corners
                drawPoint(frame, aNear, (0,255,0))
                drawPoint(frame, bNear, (0,255,0))
                drawPoint(frame, cNear, (0,255,255))
                drawPoint(frame, dNear, (0,255,255))
                drawPoint(frame, aFar, (255,0,0))
                drawPoint(frame, bFar, (255,0,0))
                drawPoint(frame, cFar, (255,0,255))
                drawPoint(frame, dFar, (255,0,255))

                centerXNear = (aNear[0] + bNear[0] + cNear[0] + dNear[0]) / 4
                centerXFar = (aFar[0] + bFar[0] + cFar[0] + dFar[0]) / 4
                centerYNear = (aNear[1] + bNear[1] + cNear[1] + dNear[1]) / 4
                centerYFar = (aFar[1] + bFar[1] + cFar[1] + dFar[1]) / 4

                centerXTarget = 0
                centerYTarget = 0
                pixelDist = 0
                pixelHeight = 0
                if (aNear[0] < aFar[0]): # if the near contour is on the left
                    centerXTarget = (aNear[0] + cNear[0] + bFar[0] + dFar[0]) / 4
                    centerYTarget = (aNear[1] + cNear[1] + bFar[1] + dFar[1]) / 4
                    topPixelDist = distance(aNear, bFar)
                    bottomPixelDist = distance(cNear, dFar)
                    pixelDist = (topPixelDist + bottomPixelDist) / 2
                    leftPixelHeight = distance(aNear, cNear)
                    rightPixelHeight = distance(bFar, dFar)
                    pixelHeight = (leftPixelHeight + rightPixelHeight) / 2
                else:
                    centerXTarget = (aFar[0] + cFar[0] + bNear[0] + dNear[0]) / 4
                    centerYTarget = (aFar[1] + cFar[1] + bNear[1] + dNear[1]) / 4
                    topPixelDist = distance(aFar, bNear)
                    bottomPixelDist = distance(cFar, dNear)
                    pixelDist = (topPixelDist + bottomPixelDist) / 2
                    leftPixelHeight = distance(aFar, cFar)
                    rightPixelHeight = distance(bNear, dNear)
                    pixelHeight = (leftPixelHeight + rightPixelHeight) / 2
                #pixelsOffCenterY = pixelsOffCenter(centerYTarget, h)
                #inchesOffCenterY = pixelsToInches(pixelsOffCenterY, pixelDist, targetDist)
                #dist = (deltaAltitude + inchesOffCenterY) / np.tan(cameraAngle)
                #ratio = (leftPixelHeight * 1.0 / rightPixelHeight * 1.0)
                pixelsOffCenterX = pixelsOffCenter(centerXTarget, w)
                inchesOffCenterX = pixelsToInches(pixelsOffCenterX, pixelDist, gearTargetDist)
                inchesOffCenterTargetX = inchesOffCenterTargetX + gearCameraOffset
                centerValue = inchesOffCenterX
                if(pixelHeight != 0):
                    dist = (gearTargetHeight * h) / (2 * pixelHeight * math.tan(0.418224329) * 0.6305) # the angle  is a constant: the tangent of half of the camera's field of view angle.
                angle = math.atan2(inchesOffCenterX , dist)
                angle = math.degrees(angle)
                #print "inchesOffCenterX:" , inchesOffCenterX
                #angle = calculateLateralAngle(inchesOffCenterX, dist)
                #centerValue = calculateCenter(centerXTarget, cameraWidth) + 1
                #print "New Distance:", dist
        else:
            #if one contour is larger, do calculations based on that one.
            #print "Using one Contour:"
            #print "dist:", dist
            # draw it on screen
            cv2.drawContours(frame, [nearBox], -1, (16,16,223), 3)

            # if it has 4 corners
            #print "Number of Corners:", len(nearBox)
            if (len(nearBox) == 4):
                # find the corners
                a,b,c,d = findCorners2(nearBox)

                # draw the corners
                drawPoint(frame, a, (0,255,0))
                drawPoint(frame, b, (255,0,0))
                drawPoint(frame, c, (255,255,0))
                drawPoint(frame, d, (0,255,255))

                centerX = (a[0] + b[0] + c[0] + d[0]) / 4
                centerY = (a[1] + b[1] + c[1] + d[1]) / 4
                #print "center of contour:", centerX, centerY
                e  = ((a[0] + d[0]) / 2, (a[1] + d[1]) / 2)
                f  = ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)
                drawPoint(frame, e, (255,0,255))
                drawPoint(frame, f, (255,0,255))
                thetaY = np.arctan((e[1] - f[1])/(e[0] - f[0]))
                pixelWidth = distance(e, f)
                #print "pixelWidth:", pixelWidth
                inchesOffCenterTargetX = 0
                pixelsOffCenterX = pixelsOffCenter(centerX, w)
                inchesOffCenterX = pixelsToInches(pixelsOffCenterX, pixelWidth, gearTargetWidth)
                inchesOffCenterTargetX = inchesOffCenterX
                if (a[1] < b[1]): # should run if the near contour is on the left (of the peg)
                    inchesOffCenterTargetX = inchesOffCenterX - 4.125
                    #ratio = (distance(a,d) * 1.0 / distance(b,c) * 1.0)
                    #print "Left side of Target"
                else:
                    inchesOffCenterTargetX = inchesOffCenterX + 4.125
                    #ratio = (distance(b,c) * 1.0 / distance(a,d) * 1.0)
                    #print "Right side of Target"
                inchesOffCenterTargetX = inchesOffCenterTargetX + gearCameraOffset
                centerValue = inchesOffCenterTargetX
                #pixelsOffCenterY = pixelsOffCenter(centerY, h)
                #inchesOffCenterY = pixelsToInches(pixelsOffCenterY, pixelWidth, targetWidth)
                #print "pixelsX:", pixelsOffCenterX
                #print "inchesX:", inchesOffCenterX
                #print "inchesTargetX:", inchesOffCenterTargetX
                #dist = (deltaAltitude + inchesOffCenterY) / np.tan(cameraAngle)
                if(pixelWidth != 0):
                    dist = (gearTargetWidth * h) / (2 * pixelWidth * math.tan(0.418224329) * 0.6305) # the angle  is a constant: the tangent of half of the camera's field of view angle.
                angle = math.atan2(inchesOffCenterTargetX , dist)
                angle = math.degrees(angle)
                #centerValue = calculateCenter((cameraWidth / 2) + (inchesOffCenterTargetX / targetWidth), cameraWidth)
    return dist, angle, centerValue

### END FUNCTIONS ###

# instantiate the video capture object
#cap = cv2.VideoCapture(countCameras())

# get the width and height
w = 640
h = 366 # 480 - 114, frame size is altered by frame = frame[114:480, 0:640]

# set the window as a named window so the click function can be bound
# cv2.namedWindow("frame")
# bind the click function
# cv2.setMouseCallback("frame",clickFunc)

#displayThreshold = False

# print instructions
print "Location program started..."
print "Good luck on your Mission!"


t = 0
directory = "calibration-images"
previousCamera = "gear"
gearCap = cv2.VideoCapture(0)
boilerCap = gearCap
# infinite loop until brokwnk
while(True):
    # set exposure
    #cap.set(cv2.cv.CV_CAP_PROP_EXPOSURE,0.1)
    #print "Exposure:", cap.get(15)
    #print "Brightness:", cap.get(10)
    #print "Contrast:", cap.get(11)
    #print "Gain:", cap.get(14)

    # flip the frame (optional)
    #frame = cv2.flip(frame,1)
    #frame = cv2.flip(frame,0)
    t += 1

    if (table.getString("whichCamera") == "gear"):
        if (previousCamera == "boiler"):
            boilerCap.release()
            gearCap = cv2.VideoCapture(0)
            previousCamera = "gear"

        #print "gearFPS:", gearCap.get(cv2.cv.CV_CAP_PROP_FPS)
        gearRet, gearFrame = gearCap.read()
        gearFrame = gearFrame[114:480, 0:640]
        gearDist, gearAngle, gearCV = gearLocate(gearFrame)
        print "gearDist:", gearDist
        print "gearAngle:", gearAngle
        print "gearCV:", gearCV
        table.putNumber('gearDistance', gearDist)
        table.putNumber('gearCV', gearCV)
        table.putNumber('gearAngle', gearAngle)
        cv2.imwrite('gear-frame-out.jpg', gearFrame)
        if (t % 20 == 0):
            filename = "0" * (5 - len(str(t))) + str(t)
            filename = directory+"/"+"gear"+"/"+filename+".jpg"
            logfile.write("Writing %s"%(filename))
            cv2.imwrite(filename, gearFrame)
            logfile.write("Complete")

    if (table.getString("whichCamera") == "boiler"):
        if (previousCamera == "gear"):
            gearCap.release()
            boilerCap = cv2.VideoCapture(1)
            previousCamera = "boiler"
        #print "boilerFPS:", boilerCap.get(5)
        boilerRet, boilerFrame = boilerCap.read()
        boilerFrame = boilerFrame[114:480, 0:640]
        boilerDist, boilerAngle, boilerCV = boilerLocate(boilerFrame)
        table.putNumber('boilerDistance', boilerDist)
        table.putNumber('boilerAngle', boilerAngle)
        table.putNumber('boilerCV', boilerCV)
        print "boilerDist:", boilerDist
        print "boilerAngle:", boilerAngle
        print "boilerCV:", boilerCV
        cv2.imwrite('boiler-frame-out.jpg', boilerFrame)
        if (t % 20 == 0):
            filename = "0" * (5 - len(str(t))) + str(t)
            filename = directory+"/"+"boiler"+"/"+filename+".jpg"
            logfile.write("Writing %s"%(filename))
            cv2.imwrite(filename, boilerFrame)
            logfile.write("Complete")

#release captures after end of everything.
gearCap.release()
boilerCap.release()
