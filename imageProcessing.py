import time
import math
from naoqi import ALProxy, ALBroker, ALModule
import numpy as np
import cv2
import NaoImageProcessing, LinesAndPlanes
import vision_definitions
import ConfigParser, argparse


class ImageProcessingClass ():
    def __init__(self, camera, motionproxy, h, NAOobject, mute, tts, diagnostic, objectColor):
        self.camera = camera
        self.motionproxy = motionproxy
        self.h = h
        self.object = NAOobject
        self.mute = mute
        self.tts = tts
        self.diagnostic = diagnostic
        self.color = objectColor

    def imageProcessingfun(self):
        """
        Used for image segmentation, finding holes on object and calculating grabbing
        point. Grabbing point is identified using function "identifyGrabPoint".
        """
        Gesture_robot = None
        offset_x = 0.0
        offset_z = 0.0
        grab_orientation = 0.0
        maxGrabDiameter = 0.3
        d = 0.0
        d_hor = 0.0
        d_ver = 0.0
        stsel = 0

        self.camera.getImage(11) #change to string if problems occur
        cv2.imwrite('camera.png', self.camera.image)

        # racunanje tocke hvatista, obrada slike te odredivanje tocke hvatista u pikselima, pretvordba u metre preko
        # ravnina i linija
        temp = self.identifyGrabPoint()
        if temp == None:
            manual_break = 1
            return None
        else:
            [grabPointImage, bottomPoint, wLeft, wRight, direction, topPoint] = temp


        lineLeft = LinesAndPlanes.getLineEquation(self.motionproxy,2,wLeft[0],wLeft[1])
        tLeft = LinesAndPlanes.intersectWithPlane(lineLeft, 2, ( 0, 0, 1, -self.h))

        lineRight = LinesAndPlanes.getLineEquation(self.motionproxy,2,wRight[0],wRight[1])
        tRight = LinesAndPlanes.intersectWithPlane(lineRight , 2, ( 0, 0, 1, -self.h))

        lineBottom = LinesAndPlanes.getLineEquation(self.motionproxy, 2, bottomPoint[0], bottomPoint[1])
        tBottom = LinesAndPlanes.intersectWithPlane(lineBottom, 2, (0,0,1,-self.h))

        lineTop = LinesAndPlanes.getLineEquation(self.motionproxy, 2, topPoint[0], topPoint[1])
        tTop = LinesAndPlanes.intersectWithPlane(lineTop, 2,(1,0,0,-tBottom[0]))

        d_hor = abs(tRight[0]-tLeft[0])
        d_ver = abs(tTop[1]-tBottom[1])

        if self.object == 'Cup':
            d=math.sqrt(pow(tRight[0]-tLeft[0],2)+pow(tRight[1]-tLeft[1],2))
        else:
            offset_x = d_hor/2
            offset_z = d_ver/2
            d = d_ver

        print('Showing 3D points')
        print ('Left point : ', tLeft)
        print ('Right point : ', tRight)
        print ('Bottom point : ', tBottom)
        print ('Top point : ', tTop)
        print('Showing object dimensions')
        print ('Object height : ', d_ver)
        print ('Object width : ', d_hor)
        # postavljanje tocke hvatista ovisno o predmetu, potrebni su odredeni pomaci
        grabPoint=None
        if self.object == 'Cup':
            grabPoint=LinesAndPlanes.planePlaneIntersection(self.motionproxy,self.h,d,grabPointImage,bottomPoint)
            print('Grab point: ', grabPoint)
            if (grabPoint[2] < self.h) or (grabPoint[2] > (self.h + 0.07)):
                grabPoint[2] = self.h + 0.06
            print('Grab point corrected: ', grabPoint)
        else:
            grabPoint = [tBottom[0]+offset_x, tBottom[1], tBottom[2]+offset_z]
        saying = ''
        # ako predmet nije prevelik za robota da ga primi, izvrsava se odredivanje ruke s kojom ce robot primiti predmet
        # te poziv funkcije za prihvacanje predmeta
        if d<maxGrabDiameter:
            if direction == -1:
                if grabPoint[1] > 0:
                    direction = 0
                else:
                    direction = 1
            print ('Grab point: ', grabPoint)
            if direction == 0:
                grab_direction = 'L'
                grab_number = 1
                print ('Direction: left')
            else:
                grab_direction = 'R'
                grab_number  = -1
                print('Direction: right')
            if self.diagnostic == 1:
                #print ('ako je dijagnostika 1')
            # dijagnosticna provjera tocke hvatista i ruke s kojom ce robot primiti predmet, potrebna potvrda da
            # su tocke dobre
                stsel=input('Does this look OK to you (1 = yes / 0 = no)? ')
            else:
                stsel = 1
            if stsel == 1:
               return [grabPoint, grab_direction, grab_number, grabPointImage]
            else:
                return None
        else:
            print('The object is too large to grab')
            if not self.mute:
                self.tts.say('The object is too large to grab')
            return None


    def identifyGrabPoint(self):
        """
        Identifies grab point on objects image.
        """
        #binaryImage=NaoImageProcessing.histThresh(image,objectColor, diagnostic) # dobivanje binarne slike
        imageTmp = cv2.medianBlur(self.camera.image, 9)
        imageTmp = cv2.cvtColor(imageTmp, cv2.cv.CV_RGB2HSV)
        satImg = cv2.split(imageTmp)[1]
        hueImg = cv2.split(imageTmp)[0]/180.0
        #retval, binaryImage = cv2.threshold(satImg, 150, 255, cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
        binaryImage = NaoImageProcessing.histThresh(self.camera.image, self.color, self.diagnostic)
        cv2.imwrite('satmask.png', satImg)
        cv2.imwrite('object_segmented.png', binaryImage)
        if cv2.countNonZero(binaryImage) < 1000:
            print('No object segmented')
            return None

        contours, hierarchy = cv2.findContours(binaryImage, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        hierarchy = hierarchy[0]
        print hierarchy
        areas = []
        avgcolors = []
        for i in range(0, len(contours)):
            if (hierarchy[i][3]>=0):
                areas+=[0]
                avgcolors+=[0.5]
            else:
                tempDraw = np.zeros(hueImg.shape)
                cv2.drawContours(tempDraw,contours,i, 255, -1)
                area = len(tempDraw[np.nonzero(tempDraw)])
                avgcolor = sum(hueImg[np.nonzero(tempDraw)])
                avgcolor/=area
                areas+=[area]
                avgcolors+=[min(abs(avgcolor-self.color), abs(1-avgcolor-self.color))]

        bestColor = min(avgcolors)
        if (bestColor > 0.2):
            return None
        maxblobidx = -1
        maxarea = 0
        for i in range(0, len(avgcolors)):
            if abs(bestColor-avgcolors[i])<0.1:
                if areas[i]>maxarea:
                    maxblobidx=i
                    maxarea = areas[i]
        if maxblobidx == -1:
            return None
        objectID=maxblobidx

        print areas
        grabPoint = []
        direction = 0

        objBox = NaoImageProcessing.getBB(contours[objectID])
        objectBB = objBox

        if hierarchy[objectID][2]<0 or object != 'Cup':
            print 'Assuming no hole in object'
            #no holes in object
            objMoments = cv2.moments(contours[objectID])
            grabPoint = [objMoments['m01']/objMoments['m00'], objMoments['m10']/objMoments['m00']]
            if (grabPoint[1]>320):
                direction = 1
            else:
                direction = -1
            #wLeft=[objectBB[2], objectBB[1]]
            #wRight=[objectBB[2], objectBB[3]]
            #objectBottomPoint = [objectBB[2], (objectBB[1]+objectBB[3])/2.0]
            #objectTopPoint = [objectBB[0], (objectBB[1]+objectBB[3])/2.0]

            wLeft = [objectBB[0], objectBB[3]]
            wRight = [objectBB[2], objectBB[3]]

            objectBottomPoint = [(objectBB[0]+objectBB[2])/2.0, objectBB[3]]
            objectTopPoint = [(objectBB[0]+objectBB[2])/2.0, objectBB[1]]

            cv2.rectangle(self.camera.image,(objectBB[0], objectBB[1]), (objectBB[2], objectBB[3]), (255,255,255))
            cv2.imwrite("asdf.png", self.camera.image)

            print('Showing image pixel points: ')
            print ('Left point : ', wLeft)
            print ('Right point : ', wRight)
            print ('Bottom point : ', objectBottomPoint)
            print ('Top point : ', objectTopPoint)

        else:
            print 'Hole detected'
            hole = hierarchy[objectID][2]
            holeList = []
            while (hole>=0):
                holeList += [hole]
                hole = hierarchy[hole][0]

            if self.object == 'Cup':
                bestHole = -1
                bestRatio = 0
                side = 0
                for i in holeList:
                    holeBB = NaoImageProcessing.getBB(contours[i])
                    [upDown, leftRight] = NaoImageProcessing.cmpBB(objBox, holeBB)
                    print upDown, leftRight
                    if (abs(leftRight) > abs(upDown)):
                        if (bestRatio<abs(leftRight)):
                            bestHole = i
                            bestRatio = abs(leftRight)
                            side = leftRight
                if bestHole == -1:
                    print 'Only bad holes detected, assuming no holes'
                    cv2.rectangle(self.camera.image,(objectBB[0], objectBB[1]), (objectBB[2], objectBB[3]), (255,255,255))
                    cv2.imwrite("asdf.png", self.camera.image)
                    objMoments = cv2.moments(contours[objectID])
                    grabPoint = [objMoments['m10']/objMoments['m00'], objMoments['m01']/objMoments['m00']]
                    if (grabPoint[0]>320):
                        direction = 1
                    else:
                        direction = -1
                    #wLeft=[objectBB[2], objectBB[1]]
                    #wRight=[objectBB[2], objectBB[3]]
                    #objectBottomPoint = [objectBB[2], (objectBB[1]+objectBB[3])/2.0]
                    #objectTopPoint = [objectBB[0], (objectBB[1]+objectBB[3])/2.0]


                else:
                    holeBB = NaoImageProcessing.getBB(contours[bestHole])
                    cv2.rectangle(self.camera.image,(holeBB[0], holeBB[1]), (holeBB[2], holeBB[3]), (255,255,255))
                    cv2.rectangle(self.camera.image,(objectBB[0], objectBB[1]), (objectBB[2], objectBB[3]), (255,255,255))
                    cv2.imwrite("asdf.png", self.camera.image)
                    direction = math.copysign(1,side)
                    print direction
                    if (side>0):
                        grabPoint = [(holeBB[3]+objBox[3])/2.0, (holeBB[2]+holeBB[0])/2.0]
                        wLeft=[objectBB[2], holeBB[3]]
                        wRight=[objectBB[2], objectBB[3]]
                        objectBottomPoint = [objectBB[2], (holeBB[3]+objectBB[3])/2.0]
                        objectTopPoint = [objectBB[0], (holeBB[3]+objectBB[3])/2.0]

                    else:
                        grabPoint = [(holeBB[1]+objBox[1])/2.0, (holeBB[0]+holeBB[2])/2.0]
                        wLeft=[objectBB[2], objectBB[1]]
                        wRight=[objectBB[2], holeBB[1]]
                        objectBottomPoint = [objectBB[2], (holeBB[1]+objectBB[1])/2.0]
                        objectTopPoint = [objectBB[0], (holeBB[1]+objectBB[1])/2.0]

        print ('Grab pixel point: ',  grabPoint)

        return [grabPoint, objectBottomPoint, wLeft, wRight, direction, objectTopPoint]