# object tracking algorithm with trajectory points plotting on
# video stream window and gesture recognition

from naoqi import ALProxy, ALBroker, ALModule
import time
from vision_definitions import kVGA, kBGRColorSpace
import cv2 as opencv
import numpy as np
import random
from ghmm import *
import ConfigParser, argparse
import training

global ObjectTracker

# object tracking module
class ObjectTrackerModule(ALModule):
    def __init__(self, name, myBroker):
        ALModule.__init__(self, name)
        self.data = 0
        self.behaviors = []
        self.exists = []
        self.kindNames = []
        self.waiting = []
        self.tts = ALProxy("ALTextToSpeech")
        self.gestureProxy = ALProxy("NAOObjectGesture", myBroker)
        self.motionProxy = ALProxy("ALMotion", myBroker)
        self.memProxy = ALProxy("ALMemory", myBroker)

        self.motionProxy.setStiffnesses("Head", 1.0)
        self.gestureProxy.startTracker(15, 0)

        #self.log = open("temp.txt", "w") ############################################################

    def startTracker(self, camId):
        self.gestureProxy.startTracker(15, camId)
        #self.gestureProxy.focusObject(-1)

    def stopTracker(self):
        self.gestureProxy.stopTracker()
        self.gestureProxy.stopFocus()

    def load(self, path, name):
        self.gestureProxy.loadDataset(path)
        self.kindNames.append(name)
        self.exists.append(False)
        self.behaviors.append([])
        self.waiting.append(None)
        self.gestureProxy.trackObject(name, -len(self.kindNames))
        self.memProxy.subscribeToMicroEvent(name, "ObjectTracker", name, "storeData")

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

    def clearWaiting(self):
        for i in range(len(self.waiting)):
            self.waiting[i] = None

    def waitForBehavior(self, name, behavior):
        idx = self.getIdx(name)
        self.gestureProxy.clearEventTraj(name)
        print('Waiting for behavior: ' + str(behavior))
        if idx!=None:
            if behavior == "Frog":
                self.waiting[idx] = ["FrogL", "FrogR"]
            else:
                if behavior == "Plane":
                    self.waiting[idx] = ["PlaneL", "PlaneR"]
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

    def storeData(self, key, value, message):
        if value:
            if value[0]:
                print("I see the cup")
                #self.log.write(str(value[3][0])+", "+str(value[3][1])+"\n") ########################################
                self.data = value[3]
            else:
                self.data = 0
                print("I don't see the cup")

    def unload(self):
        self.gestureProxy.stopTracker()
        #self.log.close()
        for i in range(0, len(self.exists)):
            self.gestureProxy.removeObjectKind(0)
            self.gestureProxy.removeEvent(self.kindNames[i])

# class with functions for Kalman filter
class KalmanFilter(object):

    def __init__(self, process_variance, estimated_measurement_variance):
        self.process_variance = process_variance
        self.estimated_measurement_variance = estimated_measurement_variance
        self.posteri_estimate = 0.0
        self.posteri_error_estimate = 1.0

    def input_latest_noisy_measurement(self, measurement):
        priori_estimate = self.posteri_estimate
        priori_error_estimate = self.posteri_error_estimate + self.process_variance

        blending_factor = priori_error_estimate / (priori_error_estimate + self.estimated_measurement_variance)
        self.posteri_estimate = priori_estimate + blending_factor * (measurement - priori_estimate)
        self.posteri_error_estimate = (1 - blending_factor) * priori_error_estimate

    def get_latest_estimated_measurement(self):
        return self.posteri_estimate

# function for getting video stream from nao camera
def nao_image_getter(alvideoproxy, video):
    alimg = alvideoproxy.getImageRemote(video)
    imgheader = opencv.cv.CreateImageHeader((alimg[0], alimg[1]), opencv.cv.IPL_DEPTH_8U, 3)
    opencv.cv.SetData(imgheader, alimg[6])
    img = np.asarray(imgheader[:, :])
    return img

class GestureRecognitionClass():
    def __init__(self):
        self.IP = "192.168.1.105" # set NAO IP adress
        self.PORT = 9559
        self.myBroker =  ALBroker("myBroker", "0.0.0.0", 0, self.IP, self.PORT)

        self.alvideoproxy = ALProxy("ALVideoDevice", self.IP, self.PORT)
        self.video = self.alvideoproxy.subscribeCamera("video", 0, kVGA, kBGRColorSpace, 30)
        self.motionproxy=ALProxy('ALMotion', self.myBroker)

        self.tts = ALProxy('ALTextToSpeech', self.myBroker)
        self.behaveproxy = ALProxy('ALBehaviorManager', self.myBroker)
        self.postureproxy = ALProxy('ALRobotPosture', self.myBroker)
        self.navigationProxy = ALProxy('ALNavigation', self.myBroker)
        self.sound = ALProxy('ALAudioDevice', self.myBroker)
        self.memory = ALProxy('ALMemory', self.myBroker)
        self.memory.insertData('ObjectGrabber', int(0))
        self.camProxy = ALProxy("ALVideoDevice", self.IP, self.PORT)
        self.postureproxy.goToPosture("StandInit", 0.8)
        # Reading Config File
        cfile = "Config.ini"
        config = ConfigParser.ConfigParser()
        config.read(cfile)
        set_num = config.get("Settings", "Dataset")
        new_set = int(set_num) + 1

        self.path = config.get("Settings", "path")
        self.filename = 'gest' + str(new_set) # self.filename of a file where trajectory whill be saved

        config.set("Settings", "Dataset", str(new_set))
        with open(cfile, 'wb') as configfile:
            config.write(configfile)

        self.motionproxy.killAll()
        self.postureproxy.goToPosture("StandInit", 0.8)
        self.motionproxy.setAngles('HeadPitch', 0, 0.5)
        time.sleep(0.5)
        self.motionproxy.setAngles('HeadYaw', 0, 0.5)
        time.sleep(0.5)
        self.cluster_data = []
        self.vars = []

    def trackingInit(self):
        self.iteration_count = 500
        self.measurement_standard_deviation = np.std([random.random() * 2.0 - 1.0 for j in xrange(self.iteration_count)])

        self.process_variance = 1e-1  # greater = faster, worse estimation, lower = slower, better estimation

        self.estimated_measurement_variance = self.measurement_standard_deviation ** 2  # 0.05 ** 2
        self.kalman_filter = KalmanFilter(self.process_variance, self.estimated_measurement_variance)
        self.posteri_estimate_graph = []

        self.image_position = np.zeros(shape=2)
        self.pos_vec = np.zeros(shape=2)
        self.i = 0
        self.log = open(self.filename + ".txt", "w") ####################################################################################
        self.estimation = np.zeros(shape=(1, 2))
        self.points_x = []
        self.points_y = []
        self.gesture_started = False
        self.gesture_complete = False

        global ObjectTracker
        ObjectTracker = ObjectTrackerModule("ObjectTracker", self.myBroker)
        #ObjectTracker.load("/home/nao/ImageSets/cup", 'Cup')
        ObjectTracker.load("/home/nao/ImageSets/frog", 'Frog')
        ObjectTracker.gestureProxy.stopTracker()
        print ('Starting tracker...')
        ObjectTracker.startTracker(0)

        self.tts.say("Now you repeat the gesture")

        return 1

    def trackingLoop(self):
        gesture_started = False
        while True:
            # if object is detected do data analysis
            image = nao_image_getter(self.alvideoproxy, self.video)
            if ObjectTracker.data:
                # angular position data from micro event
                pos_data = np.asarray(ObjectTracker.data)
                #print "data: "
                #print ObjectTracker.data
                # calculating image position based on angular position of object
                image_position = self.camProxy.getImagePositionFromAngularPosition(0, [pos_data[0], pos_data[1]])
                image_position = np.asarray(image_position)

                # applying kalman filter on image position data
                self.kalman_filter.input_latest_noisy_measurement(image_position)
                self.posteri_estimate_graph.append(self.kalman_filter.get_latest_estimated_measurement())
                # separating estimated values for easier plotting
                estimation = np.zeros(shape=(len(self.posteri_estimate_graph), 2))
                index = np.zeros(len(self.posteri_estimate_graph))
                for i in range(0, len(self.posteri_estimate_graph)):
                    temp2 = self.posteri_estimate_graph[i]
                    estimation[i, 0] = temp2[0]
                    estimation[i, 1] = temp2[1]

                height, width = image.shape[:2]
                if len(estimation) > 3:
                    start_pos_y = estimation[2,1] * height + 15

                    opencv.ellipse(image, (int(estimation[-1, 0] * width), int(estimation[-1, 1] * height + 15)),
                                       (70, 90), -180, 0, 360, (255, 0, 0), 2)

                    if (start_pos_y - (estimation[-1,1] * height + 15)) > 10:
                        gesture_started = True
                        self.points_x.append(int(estimation[-1,0]* width))
                        self.points_y.append(int(estimation[-1,1]* height + 15))
                        for j in range(0, len(self.points_x)):
                            opencv.circle(image, (int(self.points_x[j]), int(self.points_y[j])), 5, (0, 0, 255), -1)

                    elif (start_pos_y - (estimation[-1,1] * height + 15)) <= 10 and gesture_started == True:
                        break

            opencv.putText(image, "Object", (10, 70), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            opencv.putText(image, "tracking", (10, 140), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            opencv.imshow("Robot camera feed", image)

            if opencv.waitKey(10) == 27:
                break
        return 1

    def trackingFinal(self):
        n = len(self.points_x) - 1
        for i in range(0, n):
            self.log.write(str(self.points_x[i]) + ", " + str(self.points_y[i])+"\n")
        self.log.close()

        #self.shutDown()
        return 1

    def shutDown(self):
        ObjectTracker.gestureProxy.stopTracker()
        print('Ending tracking...')
        time.sleep(1)
        self.alvideoproxy.unsubscribe(self.video)
        opencv.destroyAllWindows()
        ObjectTracker.unload()
        self.behaveproxy.stopAllBehaviors()
        time.sleep(1.0)
        self.motionproxy.killAll()
        self.myBroker.shutdown()

    def readData(self):
        self.state_num = int(self.f.readline())
        self.output_num = int(self.f.readline())
        self.gesture_treshold = self.f.readline()
        self.f.close()
        return 1

    def costFunction(self):
        n = len(self.points_x)
        cost_cup = 0
        cost_frog = 0
        cost_plane = 0
        cost_neg1 = 0
        cost_neg2 = 0
        max_x = 0
        min_x = 600
        idx_max = 0
        idx_min = 0

        # finding start x position and final x position, if they are close, then it is not plane or frog gesture
        x_diff = abs(self.points_x[0] - self.points_x[n-1])
        if x_diff < 35:
            cost_frog = 10
            cost_plane = 10
        elif x_diff > 40:
            cost_cup = 10
            cost_neg1 = 10
            cost_neg2 = 10
        # finding max x position
        for i in range(0, n):
            x = self.points_x[i]
            if  (x > max_x):
                max_x = self.points_x[i]
                idx_max = i
            elif (x < min_x):
                min_x = self.points_x[i]
                idx_min = i
        # if max x position is not far from start x position, then it is drinking gesture
        # else it is spilling
        if abs(abs(self.points_x[idx_max]) - abs(self.points_x[idx_min])) < 50:
            cost_neg1 = 10
            cost_neg2 = 10
        else:
            cost_cup = 10

        self.cost_list = [cost_cup, cost_frog, cost_neg1, cost_neg2, cost_plane]
        print self.cost_list

    def recognitionFun(self):
        # beginning gesture recognition
        self.costFunction()
        diff = [0, 0, 0, 0, 0]
        for i in range(0,5):
            if i == 0:
                self.f = open(self.path + '/trained/drink/gesture_file.txt', 'r')
                self.m = HMMOpen(self.path + "/trained/drink/m_file.xml")
                self.readData()
            elif i == 1:
                self.f = open(self.path + '/trained/frog/gesture_file.txt', 'r')
                self.m = HMMOpen(self.path + "/trained/frog/m_file.xml")
                self.readData()
            elif i == 2:
                self.f = open(self.path + '/trained/neg1/gesture_file.txt', 'r')
                self.m = HMMOpen(self.path + "/trained/neg1/m_file.xml")
                self.readData()
            elif i == 3:
                self.f = open(self.path + '/trained/neg2/gesture_file.txt', 'r')
                self.m = HMMOpen(self.path + '/trained/neg2/m_file.xml')
                self.readData()
            elif i == 4:
                self.f = open(self.path + '/trained/plane1/gesture_file.txt', 'r')
                self.m = HMMOpen(self.path + "/trained/plane1/m_file.xml")
                self.readData()

            cluster_data = training.cluster(self.filename + ".txt", self.output_num)
            sigma = IntegerRange(0, self.output_num)
            test_seq = EmissionSequence(sigma, cluster_data.tolist())
            diff[i] =  abs(abs(float(self.m.viterbi(test_seq)[1])) - abs(float(self.gesture_treshold)))

        for i in range(0, len(diff)):
            diff[i] += self.cost_list[i]

        print diff
        treshold = float(7)
        min_diff = min(diff)
        if (diff[0] > treshold) and (diff[1] > treshold) and (diff[2] > treshold) and (diff[3] > treshold) and (diff[4] > treshold):
            print ("NEMA GESTE")
        elif diff[0] == min_diff:
            print ('Pijenje iz case !!!')
        elif diff[1] == min_diff:
            print ('Zaba !!!')
        elif diff[2] == min_diff or diff[3] == min_diff:
            print ('Proljevanje !!!')
        elif diff[4] == min_diff:
            print ("Avion!")
        return 1

if __name__ == '__main__':
    # object tracking part
    try:
        track = GestureRecognitionClass()
        track.trackingInit()
        track.trackingLoop()
        track.trackingFinal()
        # gesture recognition part
        track.recognitionFun()
    finally:
        track.shutDown()
