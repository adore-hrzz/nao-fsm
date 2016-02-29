# object tracking algorithm with trajectory points plotting on
# video stream window

from naoqi import ALProxy, ALBroker, ALModule
import time
from vision_definitions import kVGA, kBGRColorSpace
import cv2 as opencv
import numpy as np
import random
import gesture_recognition
from ghmm import *
import ConfigParser, argparse
import training

global ObjectTracker

# object tracking module
class ObjectTrackerModule(ALModule):
    def __init__(self, name):
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


if __name__ == '__main__':
    # initializing proxies and other required parameters
    IP = "192.168.1.105"
    PORT = 9559
    myBroker = ALBroker("myBroker", "0.0.0.0", 0, IP, PORT)

    #opencv.namedWindow("Robot camera feed")
    # get sample image to detect size
    alvideoproxy = ALProxy("ALVideoDevice", IP, PORT)

    video = alvideoproxy.subscribeCamera("video", 0, kVGA, kBGRColorSpace, 30)

    motionproxy=ALProxy('ALMotion', myBroker)
    motionproxy.killAll()
    tts = ALProxy('ALTextToSpeech', myBroker)
    behaveproxy = ALProxy('ALBehaviorManager', myBroker)
    postureproxy = ALProxy('ALRobotPosture', myBroker)
    navigationProxy = ALProxy('ALNavigation', myBroker)
    sound = ALProxy('ALAudioDevice', myBroker)
    memory = ALProxy('ALMemory', myBroker)
    memory.insertData('ObjectGrabber', int(0))
    camProxy = ALProxy("ALVideoDevice", IP, PORT)

    postureproxy.goToPosture("StandInit", 0.8)
    motionproxy.setAngles('HeadPitch', 0, 0.5)
    time.sleep(0.5)
    motionproxy.setAngles('HeadYaw', 0, 0.5)
    time.sleep(0.5)
    motionproxy.setStiffnesses("Head", 1.0)

    cfile = "Config.ini"
    config = ConfigParser.ConfigParser()
    config.read(cfile)
    set_num = config.get("Grab settings", "Dataset")

    new_set = int(set_num) + 1

    filename = 'gest' + str(new_set)

    config.set("Grab settings", "Dataset", str(new_set))

    with open(cfile, 'wb') as configfile:
        config.write(configfile)
    # try object tracking
    try:
        # kalman filter preparations

        iteration_count = 500
        measurement_standard_deviation = np.std([random.random() * 2.0 - 1.0 for j in xrange(iteration_count)])

        process_variance = 1e-1  # greater = faster, worse estimation, lower = slower, better estimation

        estimated_measurement_variance = measurement_standard_deviation ** 2  # 0.05 ** 2
        kalman_filter = KalmanFilter(process_variance, estimated_measurement_variance)
        posteri_estimate_graph = []
        # initilazing tracking
        ObjectTracker = ObjectTrackerModule("ObjectTracker")
        ObjectTracker.load("/home/nao/ImageSets/cup", 'Cup')
        ObjectTracker.gestureProxy.stopTracker()
        time.sleep(2)
        tts.say("Now you repeat the gesture")
        time.sleep(2)
        print ('Starting tracker...')
        ObjectTracker.startTracker(0)
        image_position = np.zeros(shape=2)
        pos_vec = np.zeros(shape=2)
        i = 0
        log = open(filename + ".txt", "w") ####################################################################################
        estimation = np.zeros(shape=(1, 2))
        # while loop where tracking is executed
        while len(estimation) < 20:
            # if object is detected do data analysis
            image = nao_image_getter(alvideoproxy, video)
            if ObjectTracker.data:
                # angular position data from micro event
                pos_data = np.asarray(ObjectTracker.data)
                print "data: "
                print ObjectTracker.data
                # calculating image position based on angular position of object
                image_position = camProxy.getImagePositionFromAngularPosition(0, [pos_data[0], pos_data[1]])
                image_position = np.asarray(image_position)
                print image_position
                # applying kalman filter on image position data
                kalman_filter.input_latest_noisy_measurement(image_position)
                posteri_estimate_graph.append(kalman_filter.get_latest_estimated_measurement())
                # separating estimated values for easier plotting
                estimation = np.zeros(shape=(len(posteri_estimate_graph), 2))
                for i in range(0, len(posteri_estimate_graph)):
                    temp2 = posteri_estimate_graph[i]
                    estimation[i, 0] = temp2[0]
                    estimation[i, 1] = temp2[1]
                # video frame size

                height, width = image.shape[:2]

                opencv.ellipse(image, (int(estimation[-1, 0] * width), int(estimation[-1, 1] * height + 15)),
                               (70, 90), -180, 0, 360, (255, 0, 0), 2)
                # plotting trajectory points
                for j in range(2, len(estimation)):
                    opencv.circle(image, (int(estimation[j, 0] * width), int(estimation[j, 1] * height + 15)), 5, (0, 0, 255), -1)

            opencv.putText(image, "Object", (10, 70), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
            opencv.putText(image, "tracking", (10, 140), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)

            #opencv.putText(image, "Object tracking", (100, 100), opencv.FONT_HERSHEY_DUPLEX, 2.0, (0, 0, 255))

            opencv.imshow("Robot camera feed", image)
            #opencv.imwrite("Slike/Tracking/image" + str(len(estimation)) + ".png", image)

            if opencv.waitKey(10) == 27:
                break


    # if try doesn't work for any reason program breaks and stops after
    # stopping video subscribe and other things
    finally:
        n = len(estimation)
        for i in range(0, n):
            log.write(str(estimation[i, 0])+", "+str(estimation[i, 1])+"\n")
        log.close()
        ObjectTracker.gestureProxy.stopTracker()
        print('Ending tracking...')
        time.sleep(1)
        alvideoproxy.unsubscribe(video)
        opencv.destroyAllWindows()
        ObjectTracker.unload()
        behaveproxy.stopAllBehaviors()
        time.sleep(1.0)
        motionproxy.killAll()
        myBroker.shutdown()
            # definiranje varijabli ##########################################
        cluster_data = []
        vars = []
        ##################################################################
        diff1 = 0
        diff2 = 0
        diff3 = 0
        diff4 = 0
        for i in range(1, 5):
            if i == 1:
                f = open('cup_file.txt', 'r')
                state_num = int(f.readline())
                output_num = int(f.readline())
                gesture_treshold = f.readline()
                f.close()

                #[sigma, A, B, pi] = trening.matrices(state_num, output_num)
                #m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)

                cluster_data = training.cluster(filename + ".txt", output_num)
                #HMMfactory = HMMOpenFactory(GHMM_FILETYPE_XML)
                m = HMMOpen("cup_m_file.xml")
                print m
                sigma = IntegerRange(0, output_num)

                test_seq = EmissionSequence(sigma, cluster_data.tolist())
                print test_seq
                print m.viterbi(test_seq)
                print gesture_treshold
                diff1 = abs(abs(float(m.viterbi(test_seq)[1])) - abs(float(gesture_treshold)))
                print "Diff 1: "
                print diff1
            elif i == 2:
                f = open('roll_file.txt', 'r')
                state_num = int(f.readline())
                output_num = int(f.readline())
                gesture_treshold = f.readline()
                f.close()

                #[sigma, A, B, pi] = trening.matrices(state_num, output_num)
                #m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)

                cluster_data = training.cluster(filename + ".txt", output_num)
                #HMMfactory = HMMOpenFactory(GHMM_FILETYPE_XML)
                m = HMMOpen("roll_m_file.xml")
                print m
                sigma = IntegerRange(0, output_num)

                test_seq = EmissionSequence(sigma, cluster_data.tolist())
                print test_seq
                print m.viterbi(test_seq)
                print gesture_treshold
                diff2 = abs(abs(float(m.viterbi(test_seq)[1])) - abs(float(gesture_treshold)))
                print "Diff 2: "
                print diff2
            elif i == 3:
                f = open('sl_file.txt', 'r')
                state_num = int(f.readline())
                output_num = int(f.readline())
                gesture_treshold = f.readline()
                f.close()

                #[sigma, A, B, pi] = trening.matrices(state_num, output_num)
                #m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)

                cluster_data = training.cluster(filename + ".txt", output_num)
                #HMMfactory = HMMOpenFactory(GHMM_FILETYPE_XML)
                m = HMMOpen("sl_m_file.xml")
                print m
                sigma = IntegerRange(0, output_num)

                test_seq = EmissionSequence(sigma, cluster_data.tolist())
                print test_seq
                print m.viterbi(test_seq)
                print gesture_treshold
                diff3 = abs(abs(float(m.viterbi(test_seq)[1])) - abs(float(gesture_treshold)))
                print "Diff 3: "
                print diff3
            elif i == 4:
                f = open('sr_file.txt', 'r')
                state_num = int(f.readline())
                output_num = int(f.readline())
                gesture_treshold = f.readline()
                f.close()

                #[sigma, A, B, pi] = trening.matrices(state_num, output_num)
                #m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)

                cluster_data = training.cluster(filename + ".txt", output_num)
                #HMMfactory = HMMOpenFactory(GHMM_FILETYPE_XML)
                m = HMMOpen("sr_m_file.xml")
                print m
                sigma = IntegerRange(0, output_num)

                test_seq = EmissionSequence(sigma, cluster_data.tolist())
                print test_seq
                print m.viterbi(test_seq)
                print gesture_treshold
                diff4 = abs(abs(float(m.viterbi(test_seq)[1])) - abs(float(gesture_treshold)))
                print "Diff 4: "
                print diff4

        diff_all = [diff1, diff2, diff3, diff4]
        print diff_all
        min_diff = min(diff_all)
        print min_diff
        if diff1 == min_diff:
            tts.say("I recognize the gesture, you repeated drinking from a cup, great job !")
            print ('Pijenje iz case !!!')
        elif diff3 == min_diff or diff4 == min_diff:
            tts.say("Why did you spill my drink?")
            print ('Nema geste, proljevanje !!!')
        else:
            tts.say("I did not recognize the gesture, please try again")
            print ('Nema geste, kotrljanje case !!!')


 #       if float(m.viterbi(test_seq)[1]) > float(gesture_treshold):
 #           tts.say("I recognize the gesture, great job")
 #           print ('Gesta prepoznata !!!')
 #       else:
 #           tts.say("I did not recognize the gesture, please try again")
 #           print('Nema geste !!!')


