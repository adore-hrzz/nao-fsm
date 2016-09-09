# main file fot protocol, this is where state machine is running and all other
# needed programs are called from

import time
import math
from naoqi import ALProxy, ALBroker, ALModule
import numpy as np
import cv2
import NaoImageProcessing, LinesAndPlanes
import vision_definitions
import ConfigParser, argparse
import objectManipulation
import random
import cProfile

from ghmm import *

from vision_definitions import kVGA, kBGRColorSpace

import os

from transitions import Machine

ObjectTracker = None # ObjectTracker has to be global because otherwise communication with the module running on
                     # NAO robot does not work

class ObjectTrackerModule(ALModule):
    """
    Class for object detection and tracking. With it's inheritance ALModule it can use object tracking module
    that is running on NAO itself, Object Tracker.
    All it's methods are using Object Tracker features.

    Init function defines proxies and gestures that can be recognized.

    :param myBroker: broker for communication between NAO and computer
    :param name: name of ALModule
    """
    def __init__(self, name, myBroker):
        ALModule.__init__(self, name)
        self.behaviors = []
        self.exists = []
        self.kindNames = []
        self.waiting = []
        self.tts = ALProxy("ALTextToSpeech")
        self.gestureProxy = ALProxy("NAOObjectGesture", myBroker)
        self.motionProxy = ALProxy("ALMotion", myBroker)
        self.memProxy = ALProxy("ALMemory", myBroker)
        self.motionProxy.setStiffnesses("Head", 1.0)
        self.gestureProxy.startTracker(15, 1)
        self.gestureProxy.addGesture("drink", [2, 6])
        self.gestureProxy.addGesture("frogL", [1, 0, 7])
        self.gestureProxy.addGesture("frogR", [3, 4, 5])
        self.gestureProxy.addGesture("duckR", [4, 0])
        self.gestureProxy.addGesture("duckL", [0, 4])

    def startTracker(self, camId):
        """
        Starts object tracking with defined camera.

        :param camId: Id of NAOs camera.
        """
        self.gestureProxy.startTracker(15, camId)
        self.gestureProxy.focusObject(-1)
        #    self.memProxy.subscribeToMicroEvent(name, "ObjectTracker", name, "onObjGet")
        #    self.memProxy.subscribeToMicroEvent(name, "ObjectTracker", name, "storeData")

    def stopTracker(self):
        """
        Stops object tracking.
        """
        self.gestureProxy.stopTracker()
        self.gestureProxy.stopFocus()
        #    self.memProxy.unsubscribeToMicroEvent(name, "ObjectTracker")
        #    self.memProxy.unsubscribeToMicroEvent(name, "ObjectTracker")


    def load(self, path, name):
        """
        Loads image sets of objects from NAO to tracker and defines object name.

        :param path: Path to image sets in NAOs memory.
        :param name: Object name.
        """
        global ObjectTracker
        self.gestureProxy.loadDataset(path)
        self.kindNames.append(name)
        self.exists.append(False)
        self.behaviors.append([])
        self.waiting.append(None)
        self.gestureProxy.trackObject(name, -len(self.kindNames))
        self.memProxy.subscribeToMicroEvent(name, "ObjectTracker", name, "onObjGet")
        #    self.memProxy.subscribeToMicroEvent(name, "ObjectTracker", name, "storeData")

    def getIdx(self, name):
        if (name in self.kindNames):
            return self.kindNames.index(name)
        else:
            return None

    def getBehaviors(self, name):
        idx = self.getIdx(name)
        if idx!=None:
            return self.behaviors[idx]
        else:
            return None

    def getExist(self, name):
        idx = self.getIdx(name)
        if idx!=None:
            return self.exists[idx]
        else:
            return None

    def getWaiting(self, name):
        idx = self.getIdx(name)
        if idx!=None:
            return self.waiting[idx]
        else:
            return None

    def waitForBehavior(self, name, behavior):
        idx = self.getIdx(name)
        self.gestureProxy.clearEventTraj(name)
        print('Waiting for behavior: ' + str(behavior))
        if idx!=None:
            if behavior == "frog":
                self.waiting[idx] = ["frogL", "frogR"]
            else:
                if behavior == "duck":
                    self.waiting[idx] = ["duckL", "duckR"]
                else:
                    self.waiting[idx] = [behavior]
        else:
            return None

    def onObjGet(self, key, value, message):
        id = -1
        if (key in self.kindNames):
            id = self.kindNames.index(key)
        else:
            return
        if (value != None):
            if (value[0] != 0):
                self.exists[id]=True
                if (value[5]!=None):
                    #print (value[5])
                    self.behaviors[id] = value[5]
                    if (self.waiting[id]!= None):
                        for tmp in self.waiting[id]:
                            if tmp in value[5]:
                                self.waiting[id] = None
                                break
            else:
                self.exists[id]=False
                if (value[1]!=None):
                    #print (value[1])
                    self.behaviors[id] = value[1]
                    if (self.waiting[id]!= None):
                        for tmp in self.waiting[id]:
                            if tmp in value[1]:
                                self.waiting[id] = None
                                break

#    def storeData(self, key, value, message):
#        if value:
#            if value[0]:
#                print("I see the cup")
#                #self.log.write(str(value[3][0])+", "+str(value[3][1])+"\n") ########################################
#                self.data = value[3]
#            else:
#                self.data = 0
#                print("I don't see the cup")

    def unload(self):
        """
        Removes all images, and gestures from memory, stops Object Tracker.
        """
        self.gestureProxy.stopTracker()
        #    self.memProxy.unsubscribeToMicroEvent(name, "ObjectTracker")
        #    self.memProxy.unsubscribeToMicroEvent(name, "ObjectTracker")
        for i in range(0, len(self.exists)):
            self.gestureProxy.removeObjectKind(0)
            self.gestureProxy.removeEvent(self.kindNames[i])
        self.gestureProxy.removeGesture("drink")
        self.gestureProxy.removeGesture("frogL")
        self.gestureProxy.removeGesture("frogR")
        self.gestureProxy.removeGesture("duckL")
        self.gestureProxy.removeGesture("duckR")


class Fsm():
    """
    Fsm = Finite State Machine
    State machine is defined with this class, all its states, transitions and functions.

    :param states: List of all FSM states.

    :param transitions: List of all transitions between states. Transition is defined with several arguments:

                        1) trigger: command that triggers transition. After transition it runs certain function.

                        2) source: source state.

                        3) dest: destination state.

                        4) conditions: transition conditions if required

                        5) after: argument that can be named before, it contains a name of
                           function that will be called after transition.

    Init function defines all proxies for this class and it reads data from configuration file.
    """
    states = ['Start', 'Initial', 'Search', 'Image_processing',
              'Object_manipulation', 'Object_tracking']

    transitions = [
        {'trigger': 'start', 'source': 'Start', 'dest': 'Initial', 'after': 'initial_state'},
        {'trigger': 'search', 'source': 'Initial', 'dest': 'Search', 'after': 'object_detection'},
        {'trigger': 'process', 'source': ['Initial', 'Search'], 'dest': 'Image_processing', 'after': 'image_process'},
        {'trigger': 'grab', 'source': 'Image_processing', 'dest': 'Object_manipulation', 'after': 'object_manipulation'},
        {'trigger': 'track', 'source': ['Initial', 'Object_manipulation'], 'dest': 'Object_tracking', 'after': 'object_tracking'},
        {'trigger': 'initial', 'source': ['Search', 'Image_processing', 'Object_action', 'Object_tracking'], 'dest': 'Initial', 'after': 'initial_state'}
        ]

    #ObjectTracker = None

    def __init__(self):

        print ("Initializing program ...")
        self.robRotation = 0

        # arguments handling
        parser = argparse.ArgumentParser()
        parser.add_argument("config")
        parser.add_argument("state")
        args = parser.parse_args()
        cfile = args.config

        self.start_state = args.state

        config = ConfigParser.ConfigParser()
        config.read(cfile)

        # reading from protocol text file
        self.state_file = "fsm_state.ini"
        self.state_config = ConfigParser.ConfigParser()
        self.state_config.read(self.state_file)

        self.IP = config.get('Settings', 'IP')
        self.PORT = config.get('Settings', 'PORT')
        self.PORT = int(self.PORT)

        self.objectColor = float(config.get('Settings', 'hue')) # potreban fix !!! ######################################################################
        # initialization of all proxies for working with NAO
        self.myBroker = ALBroker("myBroker", "0.0.0.0", 0, self.IP, self.PORT)
        self.camera=NaoImageProcessing.NaoImgGetter(self.IP, self.PORT, 1)
        self.motionproxy=ALProxy('ALMotion', self.myBroker)
        self.motionproxy.killAll()
        self.tts=ALProxy('ALTextToSpeech', self.myBroker)
        self.behaveproxy=ALProxy('ALBehaviorManager', self.myBroker)
        self.postureproxy=ALProxy('ALRobotPosture', self.myBroker)
        self.navigationProxy = ALProxy('ALNavigation', self.myBroker)
        self.sound = ALProxy('ALAudioDevice', self.myBroker)
        self.memory = ALProxy('ALMemory', self.myBroker)
        self.memory.insertData('ObjectGrabber', int(0))
        # gesture proxy should not be used from this class, ObjectTrackerModule should be used
        # self.gestureProxy = ALProxy("NAOObjectGesture", self.myBroker)
        self.alvideoproxy = ALProxy("ALVideoDevice", self.IP, self.PORT)
        self.video = self.alvideoproxy.subscribe("video", kVGA, kBGRColorSpace, 30)

        self.camProxy = ALProxy("ALVideoDevice", self.IP, self.PORT)

        # getting data from configuration file
        self.object_is = config.get ('Settings', 'Object')

        self.volume = config.get('Settings', 'Volume')
        self.volume = int(float(self.volume))
        self.sound.setOutputVolume(self.volume)

        self.mute = config.get ('Settings', 'Mute')
        self.mute = int(self.mute)

        self.diagnostic = config.get ('Settings', 'Diagnostics')
        self.diagnostic = int(float(self.diagnostic))

        self.h = config.get ('Settings', 'Height')
        self.h = float(self.h)

        self.program_count = config.get('Settings', 'Counter_program')
        self.program_count = int(float(self.program_count))

        self.behaviour_count = config.get('Settings', 'Counter')
        self.behaviour_count = int(float(self.behaviour_count))

        # setting up state machine
        self.machine = Machine(model=self, states=self.states, transitions=self.transitions, initial='Start')
        self.head_pitch = [0, 0]
        self.head_yaw = [0, 0]
        self.found = False
        self.back_to_initial = False

        # initialization of object trackier module by object tracker class
        global ObjectTracker
        ObjectTracker = ObjectTrackerModule("ObjectTracker", self.myBroker)


        # grabbing parameters for later use
        self.grabPoint = 0
        self.grab_direction = 0
        self.grab_number = 0
        self.Nao_object = ''
        self.grab_pix = 0



        print ("Initialization complete ...")

    def initial_state(self):
        """
        Puts NAO in its initial state, standing position.
        From here, Fsm goes to its next state.
        """

        #postavljanje vrijednosti u text file koji sluzi za komunikaciju sa zasebnim programom za gesture recognition
        self.state_config.set("State info", "state", "Initial")
        self.state_config.set("State info", "grab_point_x", "0")
        self.state_config.set("State info", "grab_point_y", "0")
        self.state_config.set("State info", "start_tracking", "0")
        self.state_config.set("State info", "stop_tracking", "1")
        self.state_config.set("State info", "end", "0")
        self.state_config.set("State info", "pix_x", "0")
        self.state_config.set("State info", "pix_y", "0")

        with open(self.state_file, 'wb') as configfile:
            self.state_config.write(configfile)


        # setting robot in initial standing position
        self.postureproxy.goToPosture("StandInit", 0.8)
        self.motionproxy.setAngles('HeadPitch', 0, 0.5)
        time.sleep(0.5)
        self.motionproxy.setAngles('HeadYaw', 0, 0.5)
        time.sleep(0.5)
        self.motionproxy.setStiffnesses("Head", 1.0)
        print('NAO is in initial position')

        # criteria for running program and repeating it
        if not self.back_to_initial:
            #if not self.mute:
            #    self.tts.say('Are you ready for some fun')

            if self.start_state == 'grabbing':
                self.Nao_object = 'Frog'
                self.process()
            elif self.start_state == 'tracking':
                self.track()
            else:
                self.search()
        else:

            repeat = raw_input("Repeat? : ")
            if repeat == "all":
                self.search()
            elif repeat == "grabbing":
                self.Nao_object = 'Frog'
                self.process()
            elif repeat == "tracking":
                self.track()
            elif repeat == "search":
                self.search()
            else:
                print("You said shutdown but no no no")
                self.shutdown()

    def object_detection(self):
        """
        Runs object searching using Object Tracking Module. First, NAO is searching for object near him, then, if
        object is not found, he moves his head up and search for object in distance in front of him. If the object is still
        not found, NAO moves his head right, and then left searching for object.
        If object is found, FSM goes to next state, if not, it goes back to initial state.
        """
        global ObjectTracker
        self.state_config.set("State info", "state", "Searching")
        with open(self.state_file, 'wb') as configfile:
            self.state_config.write(configfile)
        time.sleep(1)
        #if self.object_is == 'Cup':

        # loading images to object tracker and starting it
        ObjectTracker.load("/home/nao/ImageSets/frog", 'Frog')
        ObjectTracker.startTracker(1)

        # list of head positions for object searching
        head_pitch_list = [[0, 0.5], [-0.2, 0.5], [-0.2, 0.5], [-0.2, 0.5]]
        head_yaw_list = [[0, 0.5], [0, 0.5], [1, 0.5], [-1, 0.5]]
        look_dir = ["close", "far", "right", "left"]

        # searching for object with one second sleep because of head movement
        for i in range(0, 4):
            ObjectTracker.gestureProxy.stopFocus()
            self.head_pitch = head_pitch_list[i]
            self.head_yaw = head_yaw_list[i]
            print("Searching for object: " + look_dir[i])
            self.move_head()
            self.searching_for_object()
            if self.found:
                self.robRotation = self.head_yaw[0]
                time.sleep(1)
                ObjectTracker.stopTracker()
                ObjectTracker.unload()
                self.behaveproxy.stopAllBehaviors()
                time.sleep(1.0)
                self.postureproxy.goToPosture("StandInit", 0.5)
                self.motionproxy.killAll()
                #self.memory.unsubscribeToMicroEvent('Cup', "ObjectTracker")
                #self.gestureProxy.stopFocus()
                if i == 0:
                    self.process()
                else:
                    self.move_to_object()
            #time.sleep(1)
        if not self.found:
            #time.sleep(1)
            ObjectTracker.stopTracker()
            ObjectTracker.unload()
            self.behaveproxy.stopAllBehaviors()
            #time.sleep(1.0)
            self.postureproxy.goToPosture("StandInit", 0.5)
            self.motionproxy.killAll()
            self.back_to_initial = True
            self.initial()

    def searching_for_object(self):
        """
        Object_detection function is calling this function. It starts object tracking with Object Tracker module, and
        stops tracking if object is found.
        """

        #detekcija predmeta
        global ObjectTracker
        t = 1
        timeObject = 4
        while t < timeObject:
            test_1 = ObjectTracker.getExist('Frog')
            if test_1 is not None and test_1:
                if not self.mute:
                    self.tts.say('I found a frog')
                print 'Frog found'
                self.Nao_object = 'Frog'
                self.found = True
                return 1

            #test_2 = ObjectTracker.getExist('Duck')
            #if test_2 is not None and test_2:
            #    print 'Duck exists'
            #    self.Nao_object = 'Duck'
            #    self.found = True
            #    return 1

            time.sleep(1)
            print t
            t += 1
        t = 1
        #time.sleep(2)
        self.found = False
        return 0

    def move_head(self):
        """
        Moves NAO's head depending on where NAO should search for object.
        """
        self.motionproxy.setAngles('HeadPitch', self.head_pitch[0], self.head_pitch[1])
        self.motionproxy.setAngles('HeadYaw', self.head_yaw[0], self.head_yaw[1])
        return 1

    def move_to_object(self):
        """
        This function is executed if object is placed far from NAO and he has to walk towards it.
        NAO is walking until he hits an obstacle with its foot number.
        """
        self.motionproxy.setAngles('HeadYaw', 0, 0.5)
        self.motionproxy.setAngles('HeadPitch', 0, 0.5)
        self.navigationProxy.moveTo(0.0, 0.0, self.robRotation)
        time.sleep(0.5)
        self.navigationProxy.moveTo(1.0, 0.0, 0.0)
        self.navigationProxy.setSecurityDistance(0.3)
        time.sleep(0.5)
        self.navigationProxy.moveTo(-0.07, 0.0, 0.0)
        time.sleep(0.5)
        self.postureproxy.goToPosture("StandInit", 0.8)
        time.sleep(1)
        self.process()

    def image_process(self):
        """
        Used for image segmentation, finding holes on object and calculating grabbing
        point. Grabbing point is identified using function "identifyGrabPoint".
        """
        self.state_config.set("State info", "state", "Image processing")
        with open(self.state_file, 'wb') as configfile:
            self.state_config.write(configfile)

        Gesture_robot = None
        offset_x = 0.0
        offset_z = 0.0
        grab_orientation = 0.0
        maxGrabDiameter = 0.3
        d = 0.0
        d_hor = 0.0
        d_ver = 0.0
        stsel = 0

        self.camera.getImage(kBGRColorSpace) #change to string if problems occur
        cv2.imwrite('camera.png',self.camera.image)
        #cv2.imshow("Camera", self.camera.image)
        #cv2.waitKey(0)
        # calculating grabbing point with image processing
        # NOTE - this part was developed before which is why there is no detailed description of image processing
        # algorithm and it's scripts
        temp = self.identifyGrabPoint()
        print("Temp %s" % temp)
        if temp == None:
            manual_break = 1
            return None
        else:
            [grabPointImage,bottomPoint,wLeft,wRight,direction,topPoint]=temp

            self.state_config.set("State info", "pix_x", str(grabPointImage[0]))
            self.state_config.set("State info", "pix_y", str(grabPointImage[1]))
            with open(self.state_file, 'wb') as configfile:
                self.state_config.write(configfile)
            #time.sleep(5)
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

        if self.Nao_object == 'Cup':
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

        grabPoint=None
        if self.Nao_object == 'Cup':
            self.grabPoint=LinesAndPlanes.planePlaneIntersection(self.motionproxy,self.h,d,grabPointImage,bottomPoint)
            print('Grab point: ', grabPoint)
            if (self.grabPoint[2] < self.h) or (self.grabPoint[2] > (self.h + 0.07)):
                self.grabPoint[2] = self.h + 0.06
            print('Grab point corrected: ', self.grabPoint)
        else:
            self.grabPoint = [tBottom[0]+offset_x, tBottom[1], tBottom[2]+offset_z]
        saying = ''
        # if the object is not too large to grab, hand to grab object with is determined and grabbing starts
        if d<maxGrabDiameter:
            if direction == -1:
                if self.grabPoint[1] > 0:
                    direction = 0
                else:
                    direction = 1
            print ('Grab point: ', self.grabPoint)
            if direction == 0:
                self.grab_direction = 'L'
                self.grab_number= 1
                print ('Direction: left')
            else:
                self.grab_direction = 'R'
                self.grab_number  = -1
                print('Direction: right')
            if self.diagnostic == 1:
            # diagnostic enables user to see image processing data and decide is program should continue or not
                stsel=input('Does this look OK to you (1 = yes / 0 = no)? ')
            else:
                stsel = 1
            if stsel == 1:
                self.tts.say('I see how to grab the object')
                #time.sleep(1)
                self.grab()
            else:
                self.back_to_initial = True
                self.initial()
                return None
        else:
            print('The object is too large to grab')
            if not self.mute:
                self.tts.say('The object is too large to grab')
            self.back_to_initial = True
            self.initial()

    def object_manipulation(self):
        """
        Responsible for all of object manipulation NAO has to do.
        This function uses data from previous state and makes NAO grab object (function Grab), do the gesture, and
        put object back to its place (function putBack).
        """
        self.state_config.set("State info", "state", "Object manipulation")
        with open(self.state_file, 'wb') as configfile:
            self.state_config.write(configfile)

        self.grab_direction = 'L'
        self.grabPoint = [0.23, 0.07, 0.30]
        self.grab_number = 1

        manipulation = objectManipulation.ManipulationClass(self.motionproxy, self.Nao_object, self.grab_number,
                                                            self.grabPoint, self.memory, self.postureproxy,
                                                            self.grab_direction)
        if not self.mute:
            if self.grab_direction == 'R':
                hand = 'right'
            elif self.grab_direction == 'L':
                hand = 'left'
            saying = 'Grabbing a ' + self.Nao_object + ' with' + hand + ' hand'
            self.tts.say(saying)
    # calling function for grabbing object that is defined in objectManipulation script
        work = manipulation.objectAction("Grab")
        if not work:
            return None
    # depending on type of object a corresponding gesture is executed
        if self.Nao_object == 'Frog':
            self.behaviour = 'frog' + str(self.grab_direction)
        else:
            self.behaviour = self.Nao_object.lower() + str(self.grab_direction)
        print(self.behaviour)

        #self.behaveproxy.runBehavior(self.behaviour)
        print("PERFORMING GESTURES")
        time.sleep(1.0)
        print("PERFORMING GESTURES")
        if self.behaviour == 'frog' + str(self.grab_direction):
            self.behaviour = 'frog'
        self.Gesture_robot = self.Nao_object

        test = self.memory.getData('ObjectGrabber')
        if test:
            manual_break = 1
            return None
        # after the gesture, object is put back in place
        #if not self.mute:
        #    self.tts.say('I have to put the object back')
        work = manipulation.objectAction("putBack")
        if not work:
            return None
        else:
            self.track()

    def object_tracking(self):
        """
        Stars object tracking that is used to detect objects trajectory and evaluate if it's trajectory
        is similar to some of defined gestures.
        """
        #time.sleep(1)
        self.postureproxy.goToPosture("StandInit", 0.5)
        self.motionproxy.killAll()
        self.state_config.read(self.state_file)
        self.state_config.set("State info", "state", "Object tracking")
        self.state_config.set("State info", "start_tracking", "1")
        self.state_config.set("State info", "stop_tracking", "0")
        self.state_config.set("State info", "end", "0")
        with open(self.state_file, 'wb') as configfile:
            self.state_config.write(configfile)

        print ("starting gesture recognition")

        #time.sleep(2)

        os.system('python recognition.py Config.ini')

        time.sleep(2)
        self.myBroker = ALBroker("myBroker", "0.0.0.0", 0, self.IP, self.PORT)
        self.back_to_initial = True
        self.initial()

    def identifyGrabPoint(self):
        """
        Identifies grab point on objects image.
        """
        #binaryImage=NaoImageProcessing.histThresh(image,objectColor, diagnostic) # dobivanje binarne slike
        imageTmp = cv2.medianBlur(self.camera.image, 9)
        imageTmp=cv2.cvtColor(imageTmp,cv2.cv.CV_RGB2HSV)
        satImg = cv2.split(imageTmp)[1]
        hueImg = cv2.split(imageTmp)[0]/180.0
        #cv2.imshow("SatImg", satImg)
        #.imshow("HueImg", hueImg)
        #cv2.waitKey(0)
        #cv2.destroyWindow("SatImg")
        #cv2.destroyWindow("HueImg")
        #retval, binaryImage = cv2.threshold(satImg, 150, 255, cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
        binaryImage = NaoImageProcessing.histThresh(self.camera.image, self.objectColor, self.diagnostic)
        #cv2.imwrite('satmask.png',satImg)
        cv2.imwrite('object_segmented.png',binaryImage)

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
                avgcolors+=[min(abs(avgcolor-self.objectColor), abs(1-avgcolor-self.objectColor))]

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

        if hierarchy[objectID][2]<0 or self.Nao_object != 'Cup':
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

            if self.Nao_object == 'Cup':
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

    def shutdown(self):
        cv2.destroyAllWindows()
        ObjectTracker.unload()
        self.behaveproxy.stopAllBehaviors()
        time.sleep(1.0)
        self.postureproxy.goToPosture("StandInit", 0.5)
        self.motionproxy.killAll()
        self.myBroker.shutdown()
        #self.alvideoproxy.unsubscribeAllInstances(self.video)
        #exit()



if __name__ == '__main__':
    # main is used only to initialize state machine and to start it
    pr = cProfile.Profile()
    pr.enable()
    nao = Fsm()
    try:
        nao.start()
    finally:
        pr.disable()
        #pr.print_stats(sort='time')
        nao.shutdown()
