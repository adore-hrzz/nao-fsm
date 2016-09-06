import time
from naoqi import ALProxy, ALBroker
import cv2
import NaoImageProcessing
import ConfigParser, argparse
import paramiko, os

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    cfile = args.config

    config = ConfigParser.ConfigParser()
    config.read(cfile)

    IP = config.get('Grab settings', 'IP')
    PORT = config.get('Grab settings', 'PORT')
    PORT = int(float(PORT))

    myBroker = ALBroker("myBroker", "0.0.0.0", 0, IP, PORT)

    camera=NaoImageProcessing.NaoImgGetter(IP, PORT, 1)
    motionproxy=ALProxy('ALMotion', myBroker)
    motionproxy.killAll()
    tts=ALProxy('ALTextToSpeech', myBroker)
    behaveproxy=ALProxy('ALBehaviorManager', myBroker)
    postureproxy=ALProxy('ALRobotPosture', myBroker)
    postureproxy.goToPosture("StandInit",0.7)
    motionproxy.setStiffnesses("Head",1.0)
    motionproxy.setAngles('HeadPitch',0.0,0.5)
    time.sleep(0.5)
    motionproxy.setAngles('HeadYaw',0,0.5)
    time.sleep(0.5)

    objectColor = config.get('Grab settings', 'Hue')
    objectColor = float(objectColor)

    camera.getImage(11)
    objectName = config.get('Grab settings', 'Object_name')
    binaryImage=NaoImageProcessing.histThresh(camera.image, objectColor, True)

    if not os.path.exists(objectName):
        os.makedirs(objectName)
        os.makedirs(objectName+'/Dataset')
        os.makedirs(objectName+'/GroundTruth')
    cv2.imwrite(objectName +'/Dataset/object.jpg', cv2.cvtColor(camera.image, cv2.cv.CV_BGR2RGB))
    cv2.imwrite(objectName +'/GroundTruth/object.png', binaryImage*255.0)

    camera.switchCamera(0)
    motionproxy.setAngles('HeadYaw',-1,0.5)
    time.sleep(0.5)
    camera.getImage(11)
    cv2.imwrite(objectName +'/Dataset/background0.jpg',cv2.cvtColor(camera.image, cv2.cv.CV_BGR2RGB))
    cv2.imwrite(objectName +'/GroundTruth/background0.png', binaryImage*0.0)

    motionproxy.setAngles('HeadYaw',0,0.5)
    time.sleep(0.5)
    camera.getImage(11)
    cv2.imwrite(objectName +'/Dataset/background1.jpg',cv2.cvtColor(camera.image, cv2.cv.CV_BGR2RGB))
    cv2.imwrite(objectName +'/GroundTruth/background1.png', binaryImage*0.0)

    motionproxy.setAngles('HeadYaw',1,0.5)
    time.sleep(0.5)
    camera.getImage(11)
    cv2.imwrite(objectName +'/Dataset/background2.jpg',cv2.cvtColor(camera.image, cv2.cv.CV_BGR2RGB))
    cv2.imwrite(objectName +'/GroundTruth/background2.png', binaryImage*0.0)

    motionproxy.setAngles('HeadYaw',0,0.5)
    time.sleep(0.5)

    #os.system('sshpass -p "nao" scp -r /home/frano/devel/code/git/nao-imitation-fsm/Image\ sets/' + objectName + '/ nao@' + IP + ':/home/nao/ImageSets/')
    print('done')

    myBroker.shutdown()
