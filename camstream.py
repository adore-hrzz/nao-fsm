# camera streaming algorithm showing state of FSM

import time

import cv2 as opencv
import numpy as np
from naoqi import ALProxy, ALBroker, ALModule
from vision_definitions import kVGA, kBGRColorSpace
import ConfigParser, argparse

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

    ######################################################################
    #opencv.namedWindow("Robot camera feed")
    # get sample image to detect size
    alvideoproxy = ALProxy("ALVideoDevice", IP, PORT)
    video = alvideoproxy.subscribeCamera("video", 1, kVGA, kBGRColorSpace, 30)

    cfile = "fsm_state.ini"
    config = ConfigParser.ConfigParser()

    tts = ALProxy('ALTextToSpeech', myBroker)
    #######################
    motionproxy=ALProxy('ALMotion', myBroker)
    motionproxy.killAll()
    behaveproxy = ALProxy('ALBehaviorManager', myBroker)
    postureproxy = ALProxy('ALRobotPosture', myBroker)
    navigationProxy = ALProxy('ALNavigation', myBroker)
    sound = ALProxy('ALAudioDevice', myBroker)
    ##########################
    memory = ALProxy('ALMemory', myBroker)
    camProxy = ALProxy("ALVideoDevice", IP, PORT)
    tts = ALProxy('ALTextToSpeech', myBroker)
    ####################3
    motionproxy.setAngles('HeadPitch', 0, 0.4)
    time.sleep(0.5)
    motionproxy.setAngles('HeadYaw', 0, 0.2)
    time.sleep(0.5)
    #############################
    try:
        image_position = np.zeros(shape=2)
        pos_vec = np.zeros(shape=2)
        i = 0
        #log = open(filename + ".txt", "w") ####################################################################################
        estimation = np.zeros(shape=(1, 2))
        # while loop where tracking is executed
        i = 0

        #global ObjectTracker
        #memory.insertData('ObjectGrabber', int(0))

        #ObjectTracker.gestureProxy.stopTracker()
        #time.sleep(2)
        reset = False
        while True:
            config.read(cfile)
            image = nao_image_getter(alvideoproxy, video)
            if config.has_section("State info"):
                state = config.get("State info", "state")
                end = config.get("State info", "end")
                start = config.get("State info", "start_tracking")
                pointx = config.get("State info", "pix_x")
                pointy = config.get("State info", "pix_y")
                plotx = int(round(float(pointx)))
                ploty = int(round(float(pointy)))

                if state == "Initial":
                    opencv.putText(image, "Initial", (10, 70), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                    opencv.imwrite("Slike/init.png", image)
                elif state == "Searching":
                    opencv.putText(image, "Searching", (10, 70), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                    opencv.imwrite("Slike/search.png", image)
                elif state == "Image processing":
                    opencv.putText(image, "Image", (10, 70), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                    opencv.putText(image, "processing", (10, 140), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                    opencv.imwrite("Slike/process.png", image)
                    if plotx != 0 or ploty != 0:
                        #opencv.circle(image, (ploty + 70, plotx), 7, (0, 255, 0), -1)
                        #time.sleep(0.5)
                        #opencv.putText(image, "Grab point", (ploty - 70, plotx + 60), opencv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        #opencv.circle(image, (ploty + 70, plotx), 63, (0, 0, 255), -1)
                        #opencv.ellipse(image, (ploty + 70, plotx), (30, 30), -180, 0, 360, (0, 0, 255), 2)
                        #time.sleep(2)

                        opencv.circle(image, (ploty, plotx + 50), 7, (0, 255, 0), -1)
                        time.sleep(0.5)
                        opencv.putText(image, "Grab point", (ploty - 70, plotx + 110), opencv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        #opencv.circle(image, (ploty + 70, plotx), 63, (0, 0, 255), -1)
                        opencv.ellipse(image, (ploty, plotx + 50), (30, 30), -180, 0, 360, (0, 0, 255), 2)
                        time.sleep(2)
                        opencv.imwrite("Slike/processres.png", image)

                elif state == "Object manipulation":
                    opencv.putText(image, "Object", (10, 70), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                    opencv.putText(image, "manipulation", (10, 140), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                    opencv.imwrite("Slike/manip.png", image)
                elif state == "Object tracking":
                    opencv.putText(image, "Object", (10, 70), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                    opencv.putText(image, "tracking", (10, 140), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                    opencv.imwrite("Slike/track.png", image)
                    myBroker.shutdown()
                    time.sleep(20)
                    myBroker = ALBroker("myBroker", "0.0.0.0", 0, IP, PORT)

                #opencv.putText(image, state, (10, 70), opencv.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5)
                #opencv.putText(image, state, org, fontFace, fontScale, color[, thickness[, lineType[, bottomLeftOrigin]]])
                #opencv.putText(image, state, (70, 70), opencv.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 255))
            opencv.imshow("Robot camera feed", image)
            if opencv.waitKey(10) == 27:
                break

    finally:
        time.sleep(1)
        alvideoproxy.unsubscribe(video)
        opencv.destroyAllWindows()
        time.sleep(1.0)
        myBroker.shutdown()





