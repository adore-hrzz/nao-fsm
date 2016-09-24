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

        safeUp = [0.1, self.grab_number * 0.15, 0.41, 0, 0, 0]
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
            sideOffset_app = self.grab_number * 0.04
            xOffset_app = 0.03
            sideOffset_grab = self.grab_number * 0.01
            rotation = (-1) * self.grab_number * 1.57
            heightOffset_lift = 0.05
            xOffset_lift =0.0
            xOffset_grab = 0.00
            heightOffset_grab = 0.0
            heightOffset_app = 0.0
        else:
            if self.Nao_object == 'Frog':
                # TODO: frog should be approached from the top, offsets need to be configured
                heightOffset_app = 0.12
                heightOffset_lift = 0.12
                heightOffset_grab = 0.0
                rotation = 0.0
                xOffset_grab = 0.0

            elif self.Nao_object == 'Cylinder':
                rotation = (-1) * self.grab_number * 1.57
                xOffset_app = 0.02
                xOffset_grab = 0.01
                sideOffset_app = self.grab_number * 0.04
                sideOffset_grab = self.grab_number * 0.01
                heightOffset_app = 0.01
                heightOffset_grab = 0.01
            else:
                if self.Nao_object == 'Plane':
                    # TODO: plane needs to be defined.
                    heightOffset_app = 0.12
                    heightOffset_lift = 0.12
                    heightOffset_grab = 0.0
                    rotation = 0.0
                    xOffset_grab = 0.02


        approachPoint = [self.grabPoint[0] + xOffset_app, self.grabPoint[1] + sideOffset_app, self.grabPoint[2]+heightOffset_app + 0.02, rotation, 0, 0]
        grabPoint = [self.grabPoint[0] + xOffset_grab, self.grabPoint[1] + sideOffset_grab, self.grabPoint[2] + heightOffset_grab - 0.01, rotation, 0, 0]
        liftPoint = [self.grabPoint[0] + xOffset_lift, self.grabPoint[1] + sideOffset_lift, self.grabPoint[2] + heightOffset_lift, rotation, 0, 0]
        # TODO: remove these prints
        #print("Approach point %s" % approachPoint)
        #print("Grab point %s" % grabPoint)
        #print("Lift point %s" % liftPoint)
        if action == "Grab":
            self.motionproxy.setAngles(hand,1.0,0.4)
            listOfPointsBeforeGrasp = [safeUp, approachPoint, grabPoint]
            listOfTimesBeforeGrasp = [2, 4, 5]

            test = self.memory.getData('ObjectGrabber')
            if test:
                return None

            self.motionproxy.wbEnableEffectorControl(chainName, True)
            self.motionproxy.positionInterpolations([chainName], 2, listOfPointsBeforeGrasp,mask,listOfTimesBeforeGrasp,True)

            # robot fails to reach the goal point
            # motion command is repeated until the difference between goal point and reach point is small enough
            goal_point = np.asarray(grabPoint[0:3])
            reached_point = np.asarray(self.motionproxy.getPosition(chainName, 2, True)[0:3])
            diff = np.linalg.norm(reached_point-goal_point)
            while diff > 0.015:
                interval = diff * 10
                self.motionproxy.positionInterpolations([chainName], 2, grabPoint, mask, [interval], True)
                reached_point = np.asarray(self.motionproxy.getPosition(chainName, 2, True)[0:3])
                diff = np.linalg.norm(reached_point-goal_point)
                print('Goal point %s' % goal_point)
                print('Reached point %s' % reached_point)
                print('Diff %s' % diff)

            # close hand (grab the object)
            self.motionproxy.setAngles(handName, 0.0, 0.3)

            # TODO: check what this does
            test = self.memory.getData('ObjectGrabber')
            if test:
                return None


            self.motionproxy.positionInterpolations([chainName], 2, liftPoint, mask, 1, True)
            #time.sleep(0.5)

            #self.motionproxy.positionInterpolation(chainName, 2, beh_pose, mask, 2, True)
            self.motionproxy.positionInterpolations(["Torso"], 2, beh_pose, mask, 1, True)
            return 1
        # TODO: check all of this code, putting back the object needs some polishing too.
        elif action == "putBack":
            grabPoint2 = [self.grabPoint[0] + xOffset_grab + 0.1, self.grabPoint[1] + sideOffset_grab, self.grabPoint[2] + heightOffset_grab, rotation, 0, 0]
            if self.grab_direction == 'L':
                grabPoint3 = [self.grabPoint[0] + xOffset_grab + 0.05, self.grabPoint[1] + sideOffset_grab + self.grab_number * 0.15, self.grabPoint[2] + 0.10, rotation, 0, 0]
            else:
                grabPoint3 = [self.grabPoint[0] + xOffset_grab + 0.05, self.grabPoint[1] - sideOffset_grab + self.grab_number * 0.15, self.grabPoint[2] + 0.10, rotation, 0, 0]

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