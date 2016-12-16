import time
from naoqi import ALProxy
import ConfigParser, argparse
import cv2 as opencv
from vision_definitions import kBGRColorSpace, kVGA
import numpy as np
from grabnao import NAOImageGetter
from NaoImageProcessing import histThresh, hist_thresh_new
import os


def nothing(dummy):
    pass


def nao_image_getter(alvideoproxy, video):
    alimg = alvideoproxy.getImageRemote(video)
    imgheader = opencv.cv.CreateImageHeader((alimg[0], alimg[1]), opencv.cv.IPL_DEPTH_8U, 3)
    opencv.cv.SetData(imgheader, alimg[6])
    img = np.asarray(imgheader[:, :])
    return img


def trackbars():
    opencv.namedWindow('Trackbars', opencv.WINDOW_AUTOSIZE)
    opencv.createTrackbar('H_Min', 'Trackbars', 0, 256, nothing)
    opencv.createTrackbar('H_Max', 'Trackbars', 256, 256, nothing)
    opencv.createTrackbar('S_Min', 'Trackbars', 0, 256, nothing)
    opencv.createTrackbar('S_Max', 'Trackbars', 256, 256, nothing)
    opencv.createTrackbar('V_Min', 'Trackbars', 0, 256, nothing)
    opencv.createTrackbar('V_Max', 'Trackbars', 256, 256, nothing)


def trackbars2():
    opencv.namedWindow('Trackbars', opencv.WINDOW_AUTOSIZE)
    opencv.createTrackbar('ObjColor', 'Trackbars', 0, 256, nothing)


def trackbars3():
    opencv.namedWindow('Trackbars', opencv.WINDOW_AUTOSIZE)
    opencv.createTrackbar('ObjColor', 'Trackbars', 0, 256, nothing)
    opencv.createTrackbar('SatCutoff', 'Trackbars', 0, 256, nothing)
    opencv.createTrackbar('ValCutoff', 'Trackbars', 0, 256, nothing)


def segmentation_hsv(img_getter, conf_parser, object_name, args):
    trackbars()
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
        image = img_getter.get_image()
        img_hsv = opencv.cvtColor(image, opencv.COLOR_BGR2HSV)
        segmented = opencv.inRange(img_hsv, (hue_min, sat_min, val_min), (hue_max, sat_max, val_max))
        segmented = opencv.dilate(segmented*1.0, np.ones((10, 10)))
        segmented = opencv.erode(segmented*1.0, np.ones((10, 10)))
        opencv.imshow("Segmented", segmented)
        opencv.imshow("Original", image)
        if opencv.waitKey(10) == 27:
            break
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

    return image, segmented


def segmentation_hue(img_getter, conf_parser, object_name, args):
    trackbars2()
    while True:
        obj_color = opencv.getTrackbarPos('ObjColor', 'Trackbars')
        image = img_getter.get_image()
        segmented = histThresh(image, obj_color/256.0, 0)
        opencv.imshow("Segmented", segmented)
        opencv.imshow("Original", image)
        if opencv.waitKey(10) == 27:
            break
    opencv.destroyAllWindows()
    if object_name not in conf_parser.sections():
        conf_parser.add_section(object_name)
    conf_parser.set(object_name, 'hue', obj_color/256.0)
    with open(args.config, 'wb') as configfile:
        conf_parser.write(configfile)
    return image, segmented


def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config")
    args = arg_parser.parse_args()
    conf_parser = ConfigParser.ConfigParser()
    conf_parser.read(args.config)

    ip = conf_parser.get('Settings', 'IP')
    port = conf_parser.getint('Settings', 'PORT')
    object_name = conf_parser.get('Settings', 'object')

    opencv.namedWindow("Segmented")
    img_getter = NAOImageGetter(ip, port, camera=1)
    motion_proxy = ALProxy('ALMotion', ip, port)
    posture_proxy = ALProxy('ALRobotPosture', ip, port)
    posture_proxy.goToPosture("StandInit", 0.5)

    segmentation_type = conf_parser.getint('Settings', 'segmentation_type')

    if segmentation_type == 0:
        image, binary_image = segmentation_hue(img_getter, conf_parser, object_name, args)
    else:
        image, binary_image = segmentation_hsv(img_getter, conf_parser, object_name, args)

    if not os.path.exists(object_name):
        os.makedirs(object_name)
        os.makedirs(object_name+'/Dataset')
        os.makedirs(object_name+'/GroundTruth')
    opencv.imwrite(object_name + '/Dataset/object.jpg', image)
    opencv.imwrite(object_name + '/GroundTruth/object.png', binary_image)

    img_getter = NAOImageGetter(ip, port, camera=0)

    motion_proxy.setAngles('HeadYaw', -1, 0.5)
    time.sleep(0.5)
    image = img_getter.get_image()
    opencv.imwrite(object_name + '/Dataset/background0.jpg', image)
    opencv.imwrite(object_name + '/GroundTruth/background0.png', binary_image*0.0)

    motion_proxy.setAngles('HeadYaw',0,0.5)
    time.sleep(0.5)
    image = img_getter.get_image()
    opencv.imwrite(object_name + '/Dataset/background1.jpg', image)
    opencv.imwrite(object_name + '/GroundTruth/background1.png', binary_image*0.0)

    motion_proxy.setAngles('HeadYaw',1,0.5)
    time.sleep(0.5)
    image = img_getter.get_image()
    opencv.imwrite(object_name + '/Dataset/background2.jpg', image)
    opencv.imwrite(object_name + '/GroundTruth/background2.png', binary_image*0.0)

    motion_proxy.setAngles('HeadYaw', 0, 0.5)
    time.sleep(0.5)


def main2():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("config")
    args = arg_parser.parse_args()
    conf_parser = ConfigParser.ConfigParser()
    conf_parser.read(args.config)

    ip = conf_parser.get('Settings', 'IP')
    port = conf_parser.getint('Settings', 'PORT')
    object_name = conf_parser.get('Settings', 'object')

    opencv.namedWindow("Segmented")
    img_getter = NAOImageGetter(ip, port, camera=1)
    motion_proxy = ALProxy('ALMotion', ip, port)
    posture_proxy = ALProxy('ALRobotPosture', ip, port)
    posture_proxy.goToPosture("StandInit", 0.5)
    trackbars3()
    obj_color = 0.0
    sat_cutoff = 0.0
    val_cutoff = 0.0
    while True:
        obj_color = opencv.getTrackbarPos('ObjColor', 'Trackbars')
        sat_cutoff = opencv.getTrackbarPos('SatCutoff', 'Trackbars')
        val_cutoff = opencv.getTrackbarPos('ValCutoff', 'Trackbars')
        print(val_cutoff)
        image = img_getter.get_image()
        segmented = hist_thresh_new(image, obj_color/255.0, sat_cutoff, val_cutoff, 10, 128)
        opencv.imshow("Segmented", segmented)
        opencv.imshow("Original", image)
        if opencv.waitKey(10) == 27:
            break
    opencv.destroyAllWindows()
    if object_name not in conf_parser.sections():
        conf_parser.add_section(object_name)
    conf_parser.set(object_name, 'hue', obj_color)
    conf_parser.set(object_name, 'sat_cutoff', sat_cutoff)
    conf_parser.set(object_name, 'val_cutoff', val_cutoff)
    with open(args.config, 'wb') as configfile:
        conf_parser.write(configfile)


if __name__ == '__main__':
    main2()

