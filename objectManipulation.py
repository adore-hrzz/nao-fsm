import time
import math
from naoqi import ALProxy, ALBroker, ALModule
import numpy as np
import cv2
import NaoImageProcessing, LinesAndPlanes
import vision_definitions
import ConfigParser, argparse
import imageProcessing

class ManipulationClass():
    def __init__(self, motionproxy, Nao_object, grab_number, grabPoint, memory, postureproxy, grab_direction):
        self.motionproxy = motionproxy
        self.grab_number = grab_number
        self.grabPoint = grabPoint
        self.memory = memory
        self.grab_direction = grab_direction
        self.Nao_object = Nao_object
        self.postureproxy = postureproxy

    def objectAction(self, action):
        """
        Called from object_manipulation function, it runs object grabbing with NAO.
        """
        rotControl = True
        if rotControl:
            mask = 15
        else:
            mask = 7

        xOffset_app = 0.0
        xOffset_lift = 0.0
        xOffset_grab = 0.0
        sideOffset_app = 0.0
        sideOffset_grab = 0.0
        sideOffset_lift = 0.0
        heightOffset_app = 0.0
        heightOffset_grab = 0.0
        heightOffset_lift = 0.0
        rotation = 1

        safeUp = [0.1, self.grab_number * 0.20, 0.41, 0, 0, 0]
        beh_pose = [0.05, self.grab_number * 0.05, 0.41, 0, 0, 0]

        hand = str(self.grab_direction) + 'Hand'
        arm = str(self.grab_direction) + 'Arm'

        chainName=arm
        handName=hand
        self.motionproxy.setStiffnesses(arm,1.0)
        #motionProxy.setStiffnesses("RArm",0.0)
        #self.motionproxy.setAngles(hand,1.0,0.4)


        #TODO: check all these offsets, remove hardcoding
        if self.Nao_object == 'Cup':
            sideOffset_app= self.grab_number * 0.04
            rotation= (-1) * self.grab_number * 1.57
            heightOffset_lift = 0.05
            xOffset_lift =0.0
            xOffset_grab = 0.0
            heightOffset_grab = 0.0
            heightOffset_app = 0.0
        else:
            if self.Nao_object == 'Frog':
                heightOffset_app = 0.12
                heightOffset_lift = 0.12
                heightOffset_grab = 0.0
                rotation = 0.0
                xOffset_grab = 0.0
            else:
                if self.Nao_object == 'Plane':
                    heightOffset_app = 0.12
                    heightOffset_lift = 0.12
                    heightOffset_grab = 0.0
                    rotation = 0.0
                    xOffset_grab = 0.02


        approachPoint = [self.grabPoint[0] + xOffset_app, self.grabPoint[1] + sideOffset_app, self.grabPoint[2]+heightOffset_app + 0.02, rotation, 0, 0]
        grabPoint = [self.grabPoint[0] + xOffset_grab, self.grabPoint[1] + sideOffset_grab, self.grabPoint[2] + heightOffset_grab - 0.01, rotation, 0, 0]
        liftPoint = [self.grabPoint[0] + xOffset_lift, self.grabPoint[1] + sideOffset_lift, self.grabPoint[2] + heightOffset_lift, rotation, 0, 0]

        if action == "Grab":
            self.motionproxy.setAngles(hand,1.0,0.4)
            listOfPointsBeforeGrasp = [safeUp, approachPoint, grabPoint]
            listOfTimesBeforeGrasp = [2, 4, 5]

            test = self.memory.getData('ObjectGrabber')
            if test:
                return None

            self.motionproxy.wbEnableEffectorControl(chainName, True)
            self.motionproxy.positionInterpolations([chainName], 2, listOfPointsBeforeGrasp,mask,listOfTimesBeforeGrasp,True)
            # TODO: remove this print
            print(grabPoint)
            print(self.motionproxy.getPosition(chainName, 2, True))
            self.motionproxy.setAngles(handName, 0.0, 0.3)
            time.sleep(1.0)
            test = self.memory.getData('ObjectGrabber')
            if test:
                return None
            self.motionproxy.positionInterpolations([chainName], 2, liftPoint, mask, 1, True)
            time.sleep(0.5)

            #self.motionproxy.positionInterpolation(chainName, 2, beh_pose, mask, 2, True)
            self.motionproxy.positionInterpolations(["Torso"], 2, beh_pose, mask, 1, True)
            return 1

        elif action == "putBack":
            grabPoint2 = [self.grabPoint[0] + xOffset_grab + 0.1, self.grabPoint[1] + sideOffset_grab, self.grabPoint[2] + heightOffset_grab, rotation, 0, 0]
            grabPoint3 = [self.grabPoint[0] + xOffset_grab + 0.05, self.grabPoint[1] + sideOffset_grab + self.grab_number * 0.15, self.grabPoint[2] + 0.10, rotation, 0, 0]
            self.motionproxy.setAngles(handName, 0.0, 0.3)
            #self.motionproxy.positionInterpolation(chainName, 2, grabPoint, mask, 2, True)
            self.motionproxy.positionInterpolation(chainName, 2, grabPoint2, mask, 2, True)
            time.sleep(0.5)
            self.motionproxy.setAngles(handName, 1.0, 0.5)
            test = self.memory.getData('ObjectGrabber')
            if test:
                return None
            time.sleep(1)
            self.motionproxy.positionInterpolation(chainName, 2, grabPoint3, mask, 2, True)
            #self.motionproxy.positionInterpolation(chainName, 2, approachPoint, mask, 2, True)
            #self.motionproxy.positionInterpolation(chainName, 2, safeUp, mask, 1, True)
            time.sleep(1)
            test = self.memory.getData('ObjectGrabber')
            if test:
                return None
            safeUp2=[0.05, self.grab_number * 0.05, 0.35, 0, 0, 0]
            #self.motionproxy.positionInterpolation(chainName, 2, safeUp2, mask, 2, True)
            self.motionproxy.positionInterpolations([chainName, "Torso"],2,[[safeUp2],[[0.05,0,0.32,0,0,0]]],[7, 7],[[2],[2]],True)
            #self.motionproxy.positionInterpolations([chainName, "Torso"], 2, [[safeUp], [[0.15, 0, 0.34, 0, 0, 0]]],
            #                                        [15, 15], [[2], [2]], True)


            #self.motionproxy.positionInterpolation("Torso", 2, beh_pose, mask, 1, True)

            #self.postureproxy.goToPosture("StandInit", 0.8)
            self.motionproxy.wbEnableEffectorControl(chainName, False)
            return 1
        else:
            return None