import time
import math
from naoqi import ALProxy, ALBroker, ALModule
import numpy as np
import cv2
import NaoImageProcessing, LinesAndPlanes
import vision_definitions
import ConfigParser, argparse

from transitions import Machine

global ObjectTracker


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
        self.gestureProxy.addGesture("drink", [2,6])
        self.gestureProxy.addGesture("frogL", [1,0,7])
        self.gestureProxy.addGesture("frogR", [3,4,5])
        self.gestureProxy.addGesture("planeR", [4,0])
        self.gestureProxy.addGesture("planeL", [0,4])

    def startTracker(self, camId):
        """
        Starts object tracking with defined camera.

        :param camId: Id of NAOs camera.
        """
        self.gestureProxy.startTracker(15, camId)
        self.gestureProxy.focusObject(-1)

    def stopTracker(self):
        """
        Stops object tracking.
        """
        self.gestureProxy.stopTracker()
        self.gestureProxy.stopFocus()

    def load(self, path, name):
        """
        Loads image sets of objects from NAO to tracker and defines object name.

        :param path: Path to image sets in NAOs memory.
        :param name: Object name.
        """
        self.gestureProxy.loadDataset(path)
        self.kindNames.append(name)
        self.exists.append(False)
        self.behaviors.append([])
        self.waiting.append(None)
        self.gestureProxy.trackObject(name, -len(self.kindNames))
        self.memProxy.subscribeToMicroEvent(name, "ObjectTracker", name, "onObjGet")

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
                if behavior == "plane":
                    self.waiting[idx] = ["planeL", "planeR"]
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
                    print (value[5])
                    self.behaviors[id] = value[5]
                    if (self.waiting[id]!= None):
                        for tmp in self.waiting[id]:
                            if tmp in value[5]:
                                self.waiting[id] = None
                                break
            else:
                self.exists[id]=False
                if (value[1]!=None):
                    print (value[1])
                    self.behaviors[id] = value[1]
                    if (self.waiting[id]!= None):
                        for tmp in self.waiting[id]:
                            if tmp in value[1]:
                                self.waiting[id] = None
                                break

    def unload(self):
        """
        Removes all images, and gestures from memory, stops Object Tracker.
        """
        self.gestureProxy.stopTracker()
        for i in range(0, len(self.exists)):
            self.gestureProxy.removeObjectKind(0)
            self.gestureProxy.removeEvent(self.kindNames[i])
        self.gestureProxy.removeGesture("drink")
        self.gestureProxy.removeGesture("frogL")
        self.gestureProxy.removeGesture("frogR")
        self.gestureProxy.removeGesture("planeL")
        self.gestureProxy.removeGesture("planeR")


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
        {'trigger': 'process', 'source': 'Search', 'dest': 'Image_processing', 'after': 'image_process'},
        {'trigger': 'grab', 'source':'Image_processing', 'dest': 'Object_manipulation', 'after': 'object_manipulation'},
        {'trigger': 'track', 'source': 'Object_action', 'dest': 'Object_tracking', 'after': 'object_tracking'},
        {'trigger': 'initial', 'source': ['Search', 'Image_processing', 'Object_action', 'Object_tracking'], 'dest': 'Initial', 'after': 'initial_state'}
        ]

    ObjectTracker = None

    def __init__(self):
        global ObjectTracker
        self.robRotation = 0

        parser = argparse.ArgumentParser()
        parser.add_argument("config")
        args = parser.parse_args()
        cfile = args.config

        config = ConfigParser.ConfigParser()
        config.read(cfile)

        self.IP = config.get('Grab settings', 'IP')
        self.PORT = config.get('Grab settings', 'PORT')
        self.PORT = int(self.PORT)

        self.objectColor = 0.0
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

        self.volume = config.get('Grab settings', 'Volume')
        self.volume = int(float(self.volume))
        self.sound.setOutputVolume(self.volume)

        self.mute = config.get ('Grab settings', 'Mute')
        self.mute = int(self.mute)

        self.diagnostic = config.get ('Grab settings', 'Diagnostics')
        self.diagnostic = int(float(self.diagnostic))

        self.h = config.get ('Grab settings', 'Height')
        self.h = float(self.h)

        self.program_count = config.get('Grab settings', 'Counter_program')
        self.program_count = int(float(self.program_count))

        self.behaviour_count = config.get('Grab settings', 'Counter')
        self.behaviour_count = int(float(self.behaviour_count))

        self.machine = Machine(model=self, states=self.states, transitions=self.transitions, initial='Start')
        self.head_pitch = [0, 0]
        self.head_yaw = [0, 0]
        self.found = False
        self.back_to_initial = False
        ObjectTracker = ObjectTrackerModule("ObjectTracker", self.myBroker)

    def initial_state(self):
        """
        Puts NAO in its initial state, standing position.
        From here, Fsm goes to its next state.
        """
        print (self.machine.state)
        self.postureproxy.goToPosture("StandInit", 0.8)
        self.motionproxy.setAngles('HeadPitch', 0, 0.5)
        time.sleep(0.5)
        self.motionproxy.setAngles('HeadYaw', 0, 0.5)
        time.sleep(0.5)
        self.motionproxy.setStiffnesses("Head", 1.0)
        print('NAO is in initial state')

        if self.back_to_initial == False:
            self.search()
        else:
            return 1

    def object_detection(self):
        """
        Runs object searching using Object Tracking Module. First, NAO is searching for object near him, then, if
        object is not found, he moves his head up and search for object in distance in front of him. If the object is still
        not found, NAO moves his head right, and then left searching for object.
        If object is found, FSM goes to next state, if not, it goes back to initial state.
        """
        global ObjectTracker
        print (self.machine.state)
        ObjectTracker.load("/home/nao/ImageSets/cup", 'Cup')
        ObjectTracker.load("/home/nao/ImageSets/zvaljak", 'Plane')
        self.head_pitch = [0, 0.5]
        self.head_yaw = [0, 0.5]
        self.move_head()
        ObjectTracker.startTracker(1)
        self.searching_for_object()
        print('Searching for object : close')
        if self.found == True:
            ObjectTracker.stopTracker()
            self.process()
        else:
            self.head_yaw = [0, 0.5]
            self.head_pitch = [-0.2, 0.5]
            print('Searching for object : far')
            self.move_head()
            self.searching_for_object()
            if self.found == True:
                ObjectTracker.stopTracker()
                self.move_to_object()
            else:
                self.head_yaw = [1, 0.5]
                self.head_pitch = [-0.2, 0.5]
                print('Searching for object : right')
                self.move_head()
                self.searching_for_object()
                if self.found == True:
                    ObjectTracker.stopTracker()
                    self.robRotation = 1
                    self.move_to_object()
                else:
                    self.head_yaw = [-1, 0.5]
                    self.head_pitch = [-0.2, 0.5]
                    print('Searching for object : left')
                    self.move_head()
                    self.searching_for_object()
                    if self.found == True:
                        ObjectTracker.stopTracker()
                        self.robRotation = -1
                        self.move_to_object()
                    else:
                        ObjectTracker.stopTracker()
                        self.back_to_initial = True
                        self.initial()

    def searching_for_object(self):
        """
        Object_detection function is calling this function. It starts object tracking with Object Tracker module, and
        stops tracking if object is found.
        """
        global ObjectTracker
        t = 1
        timeObject = 3
        while t < timeObject:
            test_1 = ObjectTracker.getExist('Cup')
            if test_1 is not None and test_1:
                print 'Cup exists'
                self.Nao_object = 'Cup'
                self.found = True
                return 1
            test_2 = ObjectTracker.getExist('Plane')
            if test_2 is not None and test_2:
                print 'Plane exists'
                self.Nao_object = 'Plane'
                self.found = True
                return 1
            time.sleep(1)
            print t
            t += 1
        t = 1
        time.sleep(2)
        self.found = False
        return 0

    def move_head(self):
        """
        Moves NAO's head depending on where NAO should search for object.
        """
        self.motionproxy.setAngles('HeadPitch',self.head_pitch[0], self.head_pitch[1])
        self.motionproxy.setAngles('HeadYaw',self.head_yaw[0], self.head_yaw[1])
        return 1

    def move_to_object(self):
        """
        This function is executed if object is placed far from NAO and he has to walk towards it.
        NAO is walking until he hits an obstacle with its foot number.
        """
        self.motionproxy.setAngles('HeadYaw',0,0.5)
        self.motionproxy.setAngles('HeadPitch',0,0.5)
        self.navigationProxy.moveTo(0.0, 0.0, self.robRotation)
        time.sleep(0.5)
        self.navigationProxy.moveTo(1.0, 0.0, 0.0)
        self.navigationProxy.setSecurityDistance(0.3)
        time.sleep(0.5)
        self.navigationProxy.moveTo(-0.07, 0.0, 0.0)
        time.sleep(0.5)
        self.postureproxy.goToPosture("StandInit",0.8)
        time.sleep(1)
        self.process()

    def image_process(self):
        """
        Used for image segmentation, finding holes on object and calculating grabbing
        point. Grabbing point is identified using function "identifyGrabPoint".
        """
        print (self.machine.state)
        Gesture_robot = None
        offset_x = 0.0
        offset_z = 0.0
        grab_orientation = 0.0
        maxGrabDiameter = 0.3
        d = 0.0
        d_hor = 0.0
        d_ver = 0.0

        self.camera.getImage(11) #change to string if problems occur
        cv2.imwrite('camera.png',self.camera.image)

        # racunanje tocke hvatista, obrada slike te odredivanje tocke hvatista u pikselima, pretvordba u metre preko
        # ravnina i linija
        temp = self.identifyGrabPoint()
        if temp == None:
            manual_break = 1
            return None
        else:
            [grabPointImage,bottomPoint,wLeft,wRight,direction,topPoint]=temp


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
        # postavljanje tocke hvatista ovisno o predmetu, potrebni su odredeni pomaci
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
        # ako predmet nije prevelik za robota da ga primi, izvrsava se odredivanje ruke s kojom ce robot primiti predmet
        # te poziv funkcije za prihvacanje predmeta
        if d<maxGrabDiameter:
            if direction==-1:
                if self.grabPoint[1]>0:
                    direction=0
                else:
                    direction=1
            print ('Grab point: ', self.grabPoint)
            if direction == 0:
                self.grab_direction = 'L'
                self.grab_number= 1
                print ('Direction: left')
            else:
                self.grab_direction = 'R'
                self.grab_number  = -1
                print('Direction: right')
            if self.diagnostic:
            # dijagnosticna provjera tocke hvatista i ruke s kojom ce robot primiti predmet, potrebna potvrda da
            # su tocke dobre
                stsel=input('Does this look OK to you (1 = yes / 0 = no)? ')
            else:
                stsel=1
            if stsel==1:

               self.grab()import time
import math
from naoqi import ALProxy, ALBroker, ALModule
import numpy as np
import cv2
import NaoImageProcessing, LinesAndPlanes
import vision_definitions
import ConfigParser, argparse

from transitions import Machine

global ObjectTracker


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
        self.gestureProxy.addGesture("drink", [2,6])
        self.gestureProxy.addGesture("frogL", [1,0,7])
        self.gestureProxy.addGesture("frogR", [3,4,5])
        self.gestureProxy.addGesture("planeR", [4,0])
        self.gestureProxy.addGesture("planeL", [0,4])

    def startTracker(self, camId):
        """
        Starts object tracking with defined camera.

        :param camId: Id of NAOs camera.
        """
        self.gestureProxy.startTracker(15, camId)
        self.gestureProxy.focusObject(-1)

    def stopTracker(self):
        """
        Stops object tracking.
        """
        self.gestureProxy.stopTracker()
        self.gestureProxy.stopFocus()

    def load(self, path, name):
        """
        Loads image sets of objects from NAO to tracker and defines object name.

        :param path: Path to image sets in NAOs memory.
        :param name: Object name.
        """
        self.gestureProxy.loadDataset(path)
        self.kindNames.append(name)
        self.exists.append(False)
        self.behaviors.append([])
        self.waiting.append(None)
        self.gestureProxy.trackObject(name, -len(self.kindNames))
        self.memProxy.subscribeToMicroEvent(name, "ObjectTracker", name, "onObjGet")

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
                if behavior == "plane":
                    self.waiting[idx] = ["planeL", "planeR"]
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
                    print (value[5])
                    self.behaviors[id] = value[5]
                    if (self.waiting[id]!= None):
                        for tmp in self.waiting[id]:
                            if tmp in value[5]:
                                self.waiting[id] = None
                                break
            else:
                self.exists[id]=False
                if (value[1]!=None):
                    print (value[1])
                    self.behaviors[id] = value[1]
                    if (self.waiting[id]!= None):
                        for tmp in self.waiting[id]:
                            if tmp in value[1]:
                                self.waiting[id] = None
                                break

    def unload(self):
        """
        Removes all images, and gestures from memory, stops Object Tracker.
        """
        self.gestureProxy.stopTracker()
        for i in range(0, len(self.exists)):
            self.gestureProxy.removeObjectKind(0)
            self.gestureProxy.removeEvent(self.kindNames[i])
        self.gestureProxy.removeGesture("drink")
        self.gestureProxy.removeGesture("frogL")
        self.gestureProxy.removeGesture("frogR")
        self.gestureProxy.removeGesture("planeL")
        self.gestureProxy.removeGesture("planeR")


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
        {'trigger': 'process', 'source': 'Search', 'dest': 'Image_processing', 'after': 'image_process'},
        {'trigger': 'grab', 'source':'Image_processing', 'dest': 'Object_manipulation', 'after': 'object_manipulation'},
        {'trigger': 'track', 'source': 'Object_manipulation', 'dest': 'Object_tracking', 'after': 'object_tracking'},
        {'trigger': 'initial', 'source': ['Search', 'Image_processing', 'Object_action', 'Object_tracking'], 'dest': 'Initial', 'after': 'initial_state'}
        ]

    ObjectTracker = None

    def __init__(self):
        global ObjectTracker
        self.robRotation = 0

        parser = argparse.ArgumentParser()
        parser.add_argument("config")
        args = parser.parse_args()
        cfile = args.config

        config = ConfigParser.ConfigParser()
        config.read(cfile)

        self.IP = config.get('Grab settings', 'IP')
        self.PORT = config.get('Grab settings', 'PORT')
        self.PORT = int(self.PORT)

        self.objectColor = 0.0
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

        self.volume = config.get('Grab settings', 'Volume')
        self.volume = int(float(self.volume))
        self.sound.setOutputVolume(self.volume)

        self.mute = config.get ('Grab settings', 'Mute')
        self.mute = int(self.mute)

        self.diagnostic = config.get ('Grab settings', 'Diagnostics')
        self.diagnostic = int(float(self.diagnostic))

        self.h = config.get ('Grab settings', 'Height')
        self.h = float(self.h)

        self.program_count = config.get('Grab settings', 'Counter_program')
        self.program_count = int(float(self.program_count))

        self.behaviour_count = config.get('Grab settings', 'Counter')
        self.behaviour_count = int(float(self.behaviour_count))

        self.machine = Machine(model=self, states=self.states, transitions=self.transitions, initial='Start')
        self.head_pitch = [0, 0]
        self.head_yaw = [0, 0]
        self.found = False
        self.back_to_initial = False
        ObjectTracker = ObjectTrackerModule("ObjectTracker", self.myBroker)

    def initial_state(self):
        """
        Puts NAO in its initial state, standing position.
        From here, Fsm goes to its next state.
        """
        print (self.machine.state)
        self.postureproxy.goToPosture("StandInit", 0.8)
        self.motionproxy.setAngles('HeadPitch', 0, 0.5)
        time.sleep(0.5)
        self.motionproxy.setAngles('HeadYaw', 0, 0.5)
        time.sleep(0.5)
        self.motionproxy.setStiffnesses("Head", 1.0)
        print('NAO is in initial state')

        if self.back_to_initial == False:
            self.search()
        else:
            return 1

    def object_detection(self):
        """
        Runs object searching using Object Tracking Module. First, NAO is searching for object near him, then, if
        object is not found, he moves his head up and search for object in distance in front of him. If the object is still
        not found, NAO moves his head right, and then left searching for object.
        If object is found, FSM goes to next state, if not, it goes back to initial state.
        """
        global ObjectTracker
        print (self.machine.state)
        ObjectTracker.load("/home/nao/ImageSets/cup", 'Cup')
        ObjectTracker.load("/home/nao/ImageSets/zvaljak", 'Plane')
        self.head_pitch = [0, 0.5]
        self.head_yaw = [0, 0.5]
        self.move_head()
        ObjectTracker.startTracker(1)
        print('Searching for object : close')
        self.searching_for_object()
        if self.found == True:
            ObjectTracker.stopTracker()
            self.process()
        else:
            self.head_yaw = [0, 0.5]
            self.head_pitch = [-0.2, 0.5]
            print('Searching for object : far')
            self.move_head()
            self.searching_for_object()
            if self.found == True:
                ObjectTracker.stopTracker()
                self.move_to_object()
            else:
                self.head_yaw = [1, 0.5]
                self.head_pitch = [-0.2, 0.5]
                print('Searching for object : right')
                self.move_head()
                self.searching_for_object()
                if self.found == True:
                    ObjectTracker.stopTracker()
                    self.robRotation = 1
                    self.move_to_object()
                else:
                    self.head_yaw = [-1, 0.5]
                    self.head_pitch = [-0.2, 0.5]
                    print('Searching for object : left')
                    self.move_head()
                    self.searching_for_object()
                    if self.found == True:
                        ObjectTracker.stopTracker()
                        self.robRotation = -1
                        self.move_to_object()
                    else:
                        ObjectTracker.stopTracker()
                        self.back_to_initial = True
                        self.initial()

    def searching_for_object(self):
        """
        Object_detection function is calling this function. It starts object tracking with Object Tracker module, and
        stops tracking if object is found.
        """
        global ObjectTracker
        t = 1
        timeObject = 3
        while t < timeObject:
            test_1 = ObjectTracker.getExist('Cup')
            if test_1 is not None and test_1:
                print 'Cup exists'
                self.Nao_object = 'Cup'
                self.found = True
                return 1
            test_2 = ObjectTracker.getExist('Plane')
            if test_2 is not None and test_2:
                print 'Plane exists'
                self.Nao_object = 'Plane'
                self.found = True
                return 1
            time.sleep(1)
            print t
            t += 1
        t = 1
        time.sleep(2)
        self.found = False
        return 0

    def move_head(self):
        """
        Moves NAO's head depending on where NAO should search for object.
        """
        self.motionproxy.setAngles('HeadPitch',self.head_pitch[0], self.head_pitch[1])
        self.motionproxy.setAngles('HeadYaw',self.head_yaw[0], self.head_yaw[1])
        return 1

    def move_to_object(self):
        """
        This function is executed if object is placed far from NAO and he has to walk towards it.
        NAO is walking until he hits an obstacle with its foot number.
        """
        self.motionproxy.setAngles('HeadYaw',0,0.5)
        self.motionproxy.setAngles('HeadPitch',0,0.5)
        self.navigationProxy.moveTo(0.0, 0.0, self.robRotation)
        time.sleep(0.5)
        self.navigationProxy.moveTo(1.0, 0.0, 0.0)
        self.navigationProxy.setSecurityDistance(0.3)
        time.sleep(0.5)
        self.navigationProxy.moveTo(-0.07, 0.0, 0.0)
        time.sleep(0.5)
        self.postureproxy.goToPosture("StandInit",0.8)
        time.sleep(1)
        self.process()

    def image_process(self):
        """
        Used for image segmentation, finding holes on object and calculating grabbing
        point. Grabbing point is identified using function "identifyGrabPoint".
        """
        print (self.machine.state)
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
        cv2.imwrite('camera.png',self.camera.image)

        # racunanje tocke hvatista, obrada slike te odredivanje tocke hvatista u pikselima, pretvordba u metre preko
        # ravnina i linija
        temp = self.identifyGrabPoint()
        if temp == None:
            manual_break = 1
            return None
        else:
            [grabPointImage,bottomPoint,wLeft,wRight,direction,topPoint]=temp


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
        # postavljanje tocke hvatista ovisno o predmetu, potrebni su odredeni pomaci
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
        # ako predmet nije prevelik za robota da ga primi, izvrsava se odredivanje ruke s kojom ce robot primiti predmet
        # te poziv funkcije za prihvacanje predmeta
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
                print ('ako je dijagnostika 1')
            # dijagnosticna provjera tocke hvatista i ruke s kojom ce robot primiti predmet, potrebna potvrda da
            # su tocke dobre
                stsel=input('Does this look OK to you (1 = yes / 0 = no)? ')
            else:
                stsel = 1
            if stsel == 1:
               self.grab()
            else:
                self.back_to_initial = True
                self.initial()
                return None
        else:
            print('The object is too large to grab')
            if not self.mute:
                self.tts.say('The object is too large to grab')
            self.initial()


    def object_manipulation(self):
        """
        Responsible for all of object manipulation NAO has to do.
        This function uses data from previous state and makes NAO grab object (function Grab), do the gesture, and
        put object back to its place (function putBack).
        """
        print (self.machine.state)
        if not self.mute:
            saying = 'Grabbing a ' + self.Nao_object + ' with left hand'
            self.tts.say(saying)
    # poziv funkcije Grab kojom se vrsi kretanje do tocke hvatista te samo hvatanje
        work = self.Grab()
        if work == None:
            return None
    # ovisno o tome koji predmet je robot primio zvrsava se odgovarajuca gesta
        if self.Nao_object == 'Cup':
            self.behaviour = 'drink' + self.grab_direction
        else:
            self.behaviour = self.Nao_object.lower() + self.grab_direction
        print(self.behaviour)

        self.behaveproxy.runBehavior(self.behaviour)
        if self.behaviour == 'drink' + self.grab_direction:
            self.behaviour = 'drink'
        self.Gesture_robot = self.Nao_object

        test = self.memory.getData('ObjectGrabber')
        if test:
            manual_break = 1
            return None
        # nakon izvodenja geste robot predmet vraca natrag na mjesto
        work = self.putBack()
        if work == None:
            return None
        self.track()

    def object_tracking(self):
        """
        Stars object tracking that is used to detect objects trajectory and evaluate if it's trajectory
        is similar to some of defined gestures.
        """
        global ObjectTracker
        self.behaveproxy.runBehavior('asadati')
        print (self.machine.state)
        print('Starting behavior tracking')
        time.sleep(1.0)
        behfound = False
        maxtime = 20
        t = 0
        ObjectTracker.startTracker(0)
        ObjectTracker.waitForBehavior(self.Nao_object, self.behaviour)
    # pracenje predmeta se izvodi odredeni vremenski period maxtime
        while (not behfound) and (t < maxtime):
            print('Waiting for behavior')
            test = self.memory.getData('ObjectGrabber')
            if test:
                break
            time.sleep(1)
            t += 1
            beh = ObjectTracker.getWaiting(self.Nao_object)
            if beh[0] == self.behaviour:
                behfound = True
            if behfound:
                if not self.mute:
                    self.tts.say('That was fun.')
            self.back_to_initial = True
            self.initial()


    def Grab(self):
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

        safeUp=[0.1, self.grab_number * 0.20, 0.41, 0, 0, 0]
        beh_pose = [0.15, self.grab_number * 0.15, 0.41, 0, 0, 0]

        hand = self.grab_direction + 'Hand'
        arm = self.grab_direction + 'Arm'

        chainName=arm
        handName=hand
        self.motionproxy.setStiffnesses(arm,1.0)
        #motionProxy.setStiffnesses("RArm",0.0)
        self.motionproxy.setAngles(hand,1.0,0.4)

        if self.Nao_object == 'Cup':
            sideOffset_app= self.grab_number * 0.04
            rotation= (-1) * self.grab_number * 1.57
            heightOffset_lift = 0.05
            xOffset_lift =0.02
            xOffset_grab = 0.02
            heightOffset_grab = 0.0
            heightOffset_app = 0.0
        else:
            if self.Nao_object == 'Frog':
                heightOffset_app = 0.12
                heightOffset_lift = 0.12
                heightOffset_grab = 0.01
                rotation = 0.0
                xOffset_grab = 0.02
            else:
                if self.Nao_object == 'Plane':
                    heightOffset_app = 0.12
                    heightOffset_lift = 0.12
                    heightOffset_grab = 0.01
                    rotation = 0.0
                    xOffset_grab = 0.02


        approachPoint = [self.grabPoint[0] + xOffset_app, self.grabPoint[1] + sideOffset_app, self.grabPoint[2]+heightOffset_app, rotation, 0, 0]
        grabPoint = [self.grabPoint[0] + xOffset_grab, self.grabPoint[1] + sideOffset_grab, self.grabPoint[2] + heightOffset_grab, rotation, 0, 0]
        liftPoint = [self.grabPoint[0] + xOffset_lift, self.grabPoint[1] + sideOffset_lift, self.grabPoint[2] + heightOffset_lift, rotation, 0, 0]

        listOfPointsBeforeGrasp=[safeUp,approachPoint, grabPoint]
        listOfTimesBeforeGrasp=[2,3,4]

        test = self.memory.getData('ObjectGrabber')
        if test:
            return None

        self.motionproxy.wbEnableEffectorControl(chainName,True)
        self.motionproxy.positionInterpolation(chainName,2,listOfPointsBeforeGrasp,mask,listOfTimesBeforeGrasp,True)
        self.motionproxy.setAngles(handName,0.0,0.3)
        time.sleep(1.0)
        test = self.memory.getData('ObjectGrabber')
        if test:
            return None
        self.motionproxy.positionInterpolation(chainName,2,liftPoint,mask,1,True)
        time.sleep(0.5)

        if object != 0:
            self.motionproxy.positionInterpolation(chainName,2,beh_pose,mask,2,True)

        return 1

    def putBack(self):
        """
        Called from object_manipulation function it runs the process of putting object back to its place with NAO.
        """
        rotControl = True
        if rotControl:
            mask=15
        else:
            mask=7

        xOffset_app = 0.0
        xOffset_lift = 0.0
        xOffset_grab = 0.0
        sideOffset_app = 0.0
        sideOffset_grab = 0.0
        sideOffset_lift = 0.0
        heightOffset_app = 0.0
        heightOffset_grab = 0.0
        heightOffset_lift = 0.0

        safeUp=[0.05, self.grab_number * 0.1, 0.33, 0, 0, 0]
        beh_pose = [0.15, self.grab_number * 0.15, 0.41, 0, 0, 0]

        hand = self.grab_direction + 'Hand'
        arm = self.grab_direction + 'Arm'

        chainName=arm
        handName=hand
        self.motionproxy.setStiffnesses(arm,1.0)
        #motionProxy.setStiffnesses("RArm",0.0)

        if self.Nao_object == 'Cup':
            rotation=(-1) * self.grab_number * 1.57
            xOffset_app = 0.13
            xOffset_grab = 0.12
            sideOffset_app = self.grab_number * 0.08
            sideOffset_grab = self.grab_number * 0.04
            heightOffset_app = 0.0
            heightOffset_grab = 0.0

        else:
            rotation = 0.0
            heightOffset_app = 0.1
            heightOffset_grab = 0.02
            xOffset_grab = 0.1
            xOffset_app = 0.05

        approachPoint = [self.grabPoint[0] + xOffset_app, self.grabPoint[1] + sideOffset_app, self.grabPoint[2] + heightOffset_app, rotation, 0, 0]
        liftPoint = [self.grabPoint[0] + xOffset_lift, self.grabPoint[1] + sideOffset_lift, self.grabPoint[2] + heightOffset_lift, rotation, 0, 0]
        grabPoint = [self.grabPoint[0] + xOffset_grab, self.grabPoint[1] + sideOffset_grab, self.grabPoint[2] + heightOffset_grab, rotation, 0, 0]

        if self.Nao_object == 'Cup':
            self.motionproxy.positionInterpolation(chainName,2,grabPoint,mask,2,True)
            time.sleep(0.5)
            self.motionproxy.setAngles(handName,1.0,0.5)
            test = self.memory.getData('ObjectGrabber')
            if test:
                return None
            time.sleep(0.5)
            self.motionproxy.positionInterpolation(chainName,2,approachPoint,mask,2,True)
            time.sleep(0.5)
            test = self.memory.getData('ObjectGrabber')
            if test:
                return None

            self.motionproxy.positionInterpolations([chainName, "Torso"],2,[[safeUp],[[0.07,0,0.32,0,0,0]]],[7, 7],[[2],[2]],True)

        else:
            self.motionproxy.wbEnableEffectorControl(chainName,True)
            listOfPointsBefore=[approachPoint, grabPoint]
            listOfTimesBefore=[2,3]

            self.motionproxy.positionInterpolation(chainName,2,listOfPointsBefore,mask,listOfTimesBefore,True)

            time.sleep(0.5)
            self.motionproxy.setAngles(handName,1.0,0.5)
            time.sleep(0.5)
            test = self.memory.getData('ObjectGrabber')
            if test:
                return None

            self.motionproxy.positionInterpolation(chainName,2,approachPoint,mask,1,True)
            test = self.memory.getData('ObjectGrabber')
            if test:
                return None
            time.sleep(0.5)
            self.motionproxy.positionInterpolations([chainName, "Torso"],2,[[safeUp],[[0.07,0,0.32,0,0,0]]],[7, 7],[[2],[2]],True)

        self.postureproxy.goToPosture("StandInit",0.8)
        self.motionproxy.wbEnableEffectorControl(chainName,False)
        return 1

    def identifyGrabPoint(self):
        """
        Identifies grab point on objects image. 
        """
        #binaryImage=NaoImageProcessing.histThresh(image,objectColor, diagnostic) # dobivanje binarne slike
        imageTmp = cv2.medianBlur(self.camera.image, 9)
        imageTmp=cv2.cvtColor(imageTmp,cv2.cv.CV_RGB2HSV)
        satImg = cv2.split(imageTmp)[1]
        hueImg = cv2.split(imageTmp)[0]/180.0
        #retval, binaryImage = cv2.threshold(satImg, 150, 255, cv2.THRESH_BINARY)#+cv2.THRESH_OTSU)
        binaryImage = NaoImageProcessing.histThresh(self.camera.image, self.objectColor, self.diagnostic)
        cv2.imwrite('satmask.png',satImg)
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


if __name__ == '__main__':
    nao = Fsm()
    nao.start()
