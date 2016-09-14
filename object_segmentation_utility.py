from naoqi import ALProxy
import ConfigParser, argparse
import cv2 as opencv
from vision_definitions import kBGRColorSpace, kVGA
import numpy as np
from NaoImageProcessing import histThresh


def nothing(dummy):
    pass


def nao_image_getter(alvideoproxy, video):
    alimg = alvideoproxy.getImageRemote(video)
    imgheader = opencv.cv.CreateImageHeader((alimg[0], alimg[1]), opencv.cv.IPL_DEPTH_8U, 3)
    opencv.cv.SetData(imgheader, alimg[6])
    img = np.asarray(imgheader[:, :])
    return img


def Trackbars():
    opencv.namedWindow('Trackbars', opencv.WINDOW_AUTOSIZE)
    opencv.createTrackbar('H_Min', 'Trackbars', 0, 256, nothing)
    opencv.createTrackbar('H_Max', 'Trackbars', 256, 256, nothing)
    opencv.createTrackbar('S_Min', 'Trackbars', 0, 256, nothing)
    opencv.createTrackbar('S_Max', 'Trackbars', 256, 256, nothing)
    opencv.createTrackbar('V_Min', 'Trackbars', 0, 256, nothing)
    opencv.createTrackbar('V_Max', 'Trackbars', 256, 256, nothing)


def Trackbars2():
    opencv.namedWindow('Trackbars', opencv.WINDOW_AUTOSIZE)
    opencv.createTrackbar('ObjColor', 'Trackbars', 0, 256, nothing)


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config")
    args = arg_parser.parse_args()
    conf_parser = ConfigParser.ConfigParser()
    conf_parser.read(args.config)

    ip = conf_parser.get('Settings', 'IP')
    port = conf_parser.getint('Settings', 'PORT')
    camera = conf_parser.getint('Settings', 'camera')
    object_name = conf_parser.get('Settings', 'object')

    opencv.namedWindow("Segmented")
    alvideoproxy = ALProxy("ALVideoDevice", ip, port)
    alvideoproxy.setParam(18, camera)
    try:
        video = alvideoproxy.subscribe("video", kVGA, kBGRColorSpace, 30)
    except RuntimeError as e:
        if e.args[0].split()[0] == 'ALVideoDevice::Subscribe':
            alvideoproxy.unsubscribeAllInstances("video")
            video = alvideoproxy.subscribe("video", kVGA, kBGRColorSpace, 30)
    Trackbars()
    hue_min = 0
    hue_max = 255
    sat_min = 0
    sat_max = 255
    val_min = 0
    val_max = 255
    while True:
        hue_min = opencv.getTrackbarPos('H_Min', 'Trackbars')
        hue_max = opencv.getTrackbarPos('H_Max', 'Trackbars')
        sat_min = opencv.getTrackbarPos('S_Min', 'Trackbars')
        sat_max = opencv.getTrackbarPos('S_Max', 'Trackbars')
        val_min = opencv.getTrackbarPos('V_Min', 'Trackbars')
        val_max = opencv.getTrackbarPos('V_Max', 'Trackbars')
        image = nao_image_getter(alvideoproxy, video)
        img_hsv = opencv.cvtColor(image, opencv.COLOR_BGR2HSV)
        segmented = opencv.inRange(img_hsv, (hue_min, sat_min, val_min), (hue_max, sat_max, val_max))
        opencv.imshow("Segmented", segmented)
        opencv.imshow("Original", image)
        if opencv.waitKey(10) == 27:
            break
    alvideoproxy.unsubscribe(video)
    opencv.destroyAllWindows()

    if object_name not in conf_parser.sections():
        conf_parser.add_section(object_name)

    conf_parser.set(object_name, 'hmin', hue_min)
    conf_parser.set(object_name, 'hmax', hue_max)
    conf_parser.set(object_name, 'smin', sat_min)
    conf_parser.set(object_name, 'smax', sat_max)
    conf_parser.set(object_name, 'vmin', val_min)
    conf_parser.set(object_name, 'vmax', val_max)
    with open(args.config, 'wb') as configfile:
        conf_parser.write(configfile)

def main2():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config")
    args = arg_parser.parse_args()
    conf_parser = ConfigParser.ConfigParser()
    conf_parser.read(args.config)

    ip = conf_parser.get('Settings', 'IP')
    port = conf_parser.getint('Settings', 'PORT')
    camera = conf_parser.getint('Settings', 'camera')
    object_name = conf_parser.get('Settings', 'object')

    opencv.namedWindow("Segmented")
    alvideoproxy = ALProxy("ALVideoDevice", ip, port)
    alvideoproxy.setParam(18, camera)
    try:
        video = alvideoproxy.subscribe("video", kVGA, kBGRColorSpace, 30)
    except RuntimeError as e:
        if e.args[0].split()[0] == 'ALVideoDevice::subscribe':
            alvideoproxy.unsubscribeAllInstances("video")
            video = alvideoproxy.subscribe("video", kVGA, kBGRColorSpace, 30)
        else:
            raise e
    Trackbars2()
    while True:
        obj_color = opencv.getTrackbarPos('ObjColor', 'Trackbars')
        image = nao_image_getter(alvideoproxy, video)
        segmented = histThresh(image, obj_color/256.0, 0)
        opencv.imshow("Segmented", segmented)
        opencv.imshow("Original", image)
        if opencv.waitKey(10) == 27:
            break
    alvideoproxy.unsubscribe(video)
    opencv.destroyAllWindows()
    if object_name not in conf_parser.sections():
        conf_parser.add_section(object_name)
    conf_parser.set(object_name, 'hue', obj_color/256.0)
    with open(args.config, 'wb') as configfile:
        conf_parser.write(configfile)


if __name__ == '__main__':
    main2()

