from vision_definitions import *  # kVGA, kBGRColorSpace, kCameraAutoWhiteBalanceID
from naoqi import ALProxy, ALModule, ALBroker
import motion
import NaoImageProcessing
import ConfigParser
import numpy as np
import argparse
import pickle
import time
import math
import cv2


class ImageProcessing:
    def __init__(self, config_parser):
        self.parser = config_parser
        self.processing_settings = dict(self.parser.items('Image processing'))

    def calculate_grab_point(self, image, object_name):
        object_settings = dict(self.parser.items(object_name))
        img = cv2.medianBlur(image, 9)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_img = cv2.split(img)[0]/180.0

        if self.processing_settings['segmentation_type'] == '0':
            hue = float(object_settings['hue'])
            binary_image = NaoImageProcessing.histThresh(image, hue, 0)
        elif self.processing_settings['segmentation_type'] == '1':
            h_min = int(object_settings['hmin'])
            h_max = int(object_settings['hmax'])
            s_min = int(object_settings['smin'])
            s_max = int(object_settings['smax'])
            v_min = int(object_settings['vmin'])
            v_max = int(object_settings['vmax'])
            binary_image = cv2.inRange(img, (h_min, s_min, v_min), (h_max, s_max, v_max))
            binary_image = cv2.dilate(binary_image*1.0, np.ones((10, 10)))
            binary_image = cv2.erode(binary_image*1.0, np.ones((10, 10)))
            binary_image = cv2.convertScaleAbs(binary_image*255)
        else:
            hue = int(object_settings['hue'])
            saturation = int(object_settings['sat_cutoff'])
            value = int(object_settings['val_cutoff'])
            binary_image = NaoImageProcessing.hist_thresh_new(img, hue, saturation, value, 15, 128)

        # TODO: remove debug output
        cv2.imwrite('object_segmented.png', binary_image)
        if cv2.countNonZero(binary_image) < int(self.processing_settings['min_size']):
            print('Object too small')
            return -1, None

        contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        hierarchy = hierarchy[0]
        areas = []
        avg_colors = []
        object_color = float(object_settings['hue'])/180.0
        for i in range(0, len(contours)):
            if hierarchy[i][3] >= 0:
                areas += [0]
                avg_colors += [0.5]
            else:
                temp_draw = np.zeros(hue_img.shape)
                cv2.drawContours(temp_draw, contours, i, 255, -1)
                area = len(temp_draw[np.nonzero(temp_draw)])
                avg_color = sum(hue_img[np.nonzero(temp_draw)])
                avg_color /= area
                areas += [area]
                avg_colors += [min(abs(avg_color-object_color), abs(1-avg_color-object_color))]
        best_color = min(avg_colors)
        # print('best color %s' % best_color)
        color_threshold = float(self.processing_settings['color_threshold'])
        if best_color > color_threshold:
            print('Best color over threshold %s' % best_color)
            return -1, None

        max_blob_idx = -1
        max_area = 0
        for i in range(0, len(avg_colors)):
            if abs(best_color-avg_colors[i]) < 0.1:
                if areas[i] > max_area:
                    max_blob_idx = i
                    max_area = areas[i]
        if max_blob_idx == -1:
            print('Max blob id negative')
            return -1, None

        object_id = max_blob_idx

        object_bb = NaoImageProcessing.getBB(contours[object_id])

        if hierarchy[object_id][2] < 0 or object_name != 'Cup':
            if object_name == 'Frog':
                obj_moments = cv2.moments(contours[object_id])
                grab_point_image = [obj_moments['m10']/obj_moments['m00'], obj_moments['m01']/obj_moments['m00']]
            elif object_name == 'Cylinder':
                grab_point_image = [(object_bb[0]+object_bb[2])/2.0, object_bb[3]]

            else:
                print('Failed to detect hole in the cup')
                return -1, None

            image_midpoint = float(self.processing_settings['cx'])

            if grab_point_image[0] > image_midpoint:
                direction = -1
            else:
                direction = 1
            cv2.circle(image, (int(grab_point_image[0]), int(grab_point_image[1])), 5, (0, 0, 255), -1)
            cv2.imwrite('GrabPoint.png', image)

        elif object_name == 'Cup':
            hole = hierarchy[object_id][2]
            hole_list = []
            while hole >= 0:
                hole_list += [hole]
                hole = hierarchy[hole][0]

            best_hole = -1
            best_ratio = 0
            side = 0

            for i in hole_list:
                hole_bb = NaoImageProcessing.getBB(contours[i])
                [up_down, left_right] = NaoImageProcessing.cmpBB_new(object_bb, hole_bb)

                if abs(left_right) > abs(up_down):
                    print('abs(left_right) > abs(up_down)')
                    if best_ratio < abs(left_right):
                        print('best_ration < abs(left_right)')
                        best_hole = i
                        best_ratio = abs(left_right)
                        side = left_right
                        print('side %s' % side)

            if best_hole == -1:
                print('Determination of best hole failed')
                return -1, None

            else:
                hole_bb = NaoImageProcessing.getBB(contours[best_hole])
                direction = math.copysign(1, side)

                grab_point_image = [(hole_bb[0]+hole_bb[2])/2, (hole_bb[1]+hole_bb[3])/2]
                cv2.circle(image, (int(grab_point_image[0]), int(grab_point_image[1])), 5, (0, 0, 255), -1)
                cv2.imwrite('GrabPoint.png', image)

        else:
            print('Unexpected failure in image processing')
            return -1, None
        cv2.destroyAllWindows()
        return 1, [grab_point_image, direction]


class NAOImageGetter:
    def __init__(self, ip, port, camera=1, resolution=kVGA, color_space=kBGRColorSpace):
        self.video_proxy = ALProxy('ALVideoDevice', ip, port)
        print(self.video_proxy.getSubscribers())
        #for subscriber in self.video_proxy.getSubscribers():
        #    self.video_proxy.unsubscribe(subscriber)
        # print(self.video_proxy.getSubscribers())
        self.video = self.video_proxy.subscribeCamera('NAOImgGet', camera, resolution, color_space, 30)
        # self.video_proxy.setCameraParameter(self.video, kCameraAutoWhiteBalanceID, 0)
        # TODO: check what this does to frog detection
        self.video_proxy.setCameraParameter(self.video, kCameraAutoExpositionID, 0)
        self.video_proxy.setCameraParameter(self.video, kCameraExposureID, 100)
        self.video_proxy.setCameraParameter(self.video, kCameraGainID, 63)


    def get_image(self):
        al_image = self.video_proxy.getImageRemote(self.video)
        image_width = al_image[0]
        image_height = al_image[1]
        channels = al_image[2]
        img_data = al_image[6]
        image = np.array(np.frombuffer(img_data, dtype=np.uint8))
        image = image.reshape((image_height, image_width, channels))
        self.video_proxy.releaseImage(self.video)
        return image

    def set_camera(self, index):
        self.video_proxy.setActiveCamera(index)

    def cleanup(self):
        print('Trying to unsubscribe %s'%self.video)
        self.video_proxy.unsubscribe(self.video)

    def __del__(self):
        self.video_proxy.unsubscribe(self.video)


class ObjectGestureModule(ALModule):
    def __init__(self, name, broker):
        ALModule.__init__(self, name)

        self.object_gesture = ALProxy('NAOObjectGesture', broker)
        self.memory = ALProxy('ALMemory', broker)
        self.data = []
        self.time_start = time.time()

    def load(self, path, object_name, module_name):
        self.object_gesture.loadDataset(path)
        self.object_gesture.trackObject(object_name, -1)
        self.memory.subscribeToMicroEvent(object_name, module_name, object_name, "on_object_detected")

    def start_tracker(self, camera_id, focus=True):
        self.object_gesture.startTracker(15, camera_id)
        if focus:
            self.object_gesture.focusObject(-1)
        self.time_start = time.time()

    def stop_tracker(self):
        self.object_gesture.stopFocus()
        self.object_gesture.stopTracker()

    def on_object_detected(self, key, value, message):
        if value:
            if value[0]:
                time_passed = time.time() - self.time_start
                data = [time_passed, value[3]]
                #print(data)
                self.data.append(data)

    def write_data(self, filename):
        with open(filename, 'w') as file_to_write:
            pickle.dump(self.data, file_to_write)

    def unload(self, object_name):
        self.object_gesture.stopTracker()
        self.object_gesture.removeObjectKind(0)
        self.object_gesture.removeEvent(object_name)


class NAO:

    def __init__(self, host, port=9559):

        print('Connecting to {0} on port {1}'.format(host, port))

        self.broker = ALBroker("nao_broker", '0.0.0.0', 0, host, port)
        self.motion = ALProxy('ALMotion')
        self.posture = ALProxy('ALRobotPosture')
        self.behavior = ALProxy('ALBehaviorManager')
        self.memory = ALProxy('ALMemory')
        self.tts = ALProxy('ALTextToSpeech')
        self.camera = NAOImageGetter(host, port)
        self.video_recorder = ALProxy('ALVideoRecorder')
        self.video_recorder.setResolution(1)
        self.video_recorder.setFrameRate(30)
        self.video_recorder.setVideoFormat("MJPG")
        self.motion.setSmartStiffnessEnabled(False)

    def cleanup(self):
        self.camera.cleanup()


class GrabNAO:
    def __init__(self, config_file_general, host, port=9559, robot=None):
        self.parser = ConfigParser.ConfigParser()
        self.parser.read(config_file_general)
        if robot:
            self.robot = robot
        else:
            self.robot = NAO(host, port)
        self.image_processor = ImageProcessing(self.parser)
        self.return_point = None

    def init_pose(self):
        self.robot.posture.goToPosture("StandInit", 0.5)
        self.robot.motion.setAngles('HeadPitch', 0, 0.5)
        self.robot.motion.setAngles('HeadYaw', 0, 0.5)

    def calculate_3d_grab_point(self, object_name):
        image = self.robot.camera.get_image()
        ret_val, data = self.image_processor.calculate_grab_point(image, object_name)

        if ret_val == -1:
            return -1, None

        camera_transform = self.robot.motion.getTransform("CameraBottom", 2, True)
        t_cam_al = np.asarray(camera_transform)
        t_cam_al = np.reshape(t_cam_al, (4, 4))

        t_al_cv = np.zeros((4, 4), dtype=np.float64)
        t_al_cv[0, 2] = 1    # this means Z'= X
        t_al_cv[1, 0] = 1    # this means X'= Y
        t_al_cv[2, 1] = 1    # this means Y'= Z
        t_al_cv[3, 3] = 1    # homogeneous coordinates!

        t_cam_cv = np.dot(t_cam_al, t_al_cv)

        cam_point = np.asarray([t_cam_al[0, 3], t_cam_al[1, 3], t_cam_al[2, 3]])

        c_x = float(self.image_processor.processing_settings['cx'])
        c_y = float(self.image_processor.processing_settings['cy'])
        f = float(self.image_processor.processing_settings['focus'])

        grab_point_image = data[0]
        direction = data[1]

        pix_point = np.asarray([c_x-grab_point_image[0], c_y-grab_point_image[1], f, 1])
        pix_point_transformed = np.dot(t_cam_cv, pix_point)
        pix_point_transformed = pix_point_transformed[0:3]

        pix_vec = pix_point_transformed - cam_point
        pix_vec /= np.linalg.norm(pix_vec)

        A = 0.0
        B = 0.0
        C = 1.0
        D = -float(self.image_processor.processing_settings['height'])
        D -= self.image_processor.parser.getfloat(object_name, 'height_offset')

        t = (-1)*(A*t_cam_cv[0, 3] + B*t_cam_cv[1, 3] + C*t_cam_cv[2, 3] + D)/(A*pix_vec[0]+B*pix_vec[1]+C*pix_vec[2])
        point = cam_point + t*pix_vec

        self.return_point = point

        return 1, [point, direction]

    def grab_object(self, object_name, point, direction, orientation_control=True):
        # TODO: make this a parameter
        # if object is more than 30cm from the robot, abort
        if point[0] > 0.32:
            print(point)
            return -1, None
        if orientation_control:
            motion_mask = motion.AXIS_MASK_ALL
        else:
            motion_mask = motion.AXIS_MASK_VEL

        safe_up = [0.15, direction * 0.13, 0.39, 0, 0, 0]
        behavior_pose = [0.05, direction * 0.05, 0.38, 0, 0, 0]

        # grabbing parameters
        grab_settings = dict(self.image_processor.parser.items(object_name))

        x_offset_approach = float(grab_settings['x_offset_approach'])
        y_offset_approach = direction*float(grab_settings['y_offset_approach'])
        z_offset_approach = float(grab_settings['z_offset_approach'])

        x_offset_grab = float(grab_settings['x_offset_grab'])
        y_offset_grab = direction*float(grab_settings['y_offset_grab'])
        z_offset_grab = float(grab_settings['z_offset_grab'])

        x_offset_lift = float(grab_settings['x_offset_lift'])
        y_offset_lift = direction*float(grab_settings['y_offset_lift'])
        z_offset_lift = float(grab_settings['z_offset_lift'])

        rotation = float(grab_settings['rotation'])

        distance_tolerance_approach = float(grab_settings['tolerance_approach'])
        distance_tolerance_grab = float(grab_settings['tolerance_grab'])

        if direction == -1:
            hand_name = 'RHand'
            chain_name = 'RArm'
        else:
            hand_name = 'LHand'
            chain_name = 'LArm'

        if object_name == 'Cup':
            print('Grabbing cup')

        elif object_name == 'Frog':
            print('Grabbing frog')

        elif object_name == 'Cylinder':
            print('Grabbing cylinder')

        else:
            print('Unknown object')
            return -1, None

        approach_point_x = point[0] + x_offset_approach
        approach_point_y = point[1] + y_offset_approach
        approach_point_z = point[2] + z_offset_approach
        approach_rotation = [-direction*rotation, 0, 0]

        grab_point_x = point[0] + x_offset_grab
        grab_point_y = point[1] + y_offset_grab
        grab_point_z = point[2] + z_offset_grab
        grab_rotation = [-direction*rotation, 0, 0]

        lift_point_x = point[0] + x_offset_lift
        lift_point_y = point[1] + y_offset_lift
        lift_point_z = point[2] + z_offset_lift
        lift_rotation = [-direction*rotation, 0, 0]

        approach_point = [approach_point_x, approach_point_y, approach_point_z,
                          approach_rotation[0], approach_rotation[1], approach_rotation[2]]
        grab_point = [grab_point_x, grab_point_y, grab_point_z, grab_rotation[0], grab_rotation[1], grab_rotation[2]]
        lift_point = [lift_point_x, lift_point_y, lift_point_z, lift_rotation[0], lift_rotation[1], lift_rotation[2]]

        self.robot.motion.setAngles(hand_name, 1.0, 0.3)

        points_before_grasp = [safe_up, approach_point]
        times_before_grasp = [1.5, 2.5]

        print('Safe up %s' % safe_up)
        print('Approach point %s' % approach_point)

        self.robot.motion.wbEnableEffectorControl(chain_name, True)
        self.robot.motion.positionInterpolations([chain_name], 2, points_before_grasp, motion_mask, times_before_grasp, True)

        goal_point = np.asarray(approach_point[0:3])

        reached_point = np.asarray(self.robot.motion.getPosition(chain_name, 2, True)[0:3])
        diff = np.linalg.norm(reached_point-goal_point)
        count = 0

        while diff > distance_tolerance_approach:
            time_interval = diff * 10
            self.robot.motion.positionInterpolations([chain_name], 2, approach_point, motion_mask, [time_interval], True)
            reached_point = np.asarray(self.robot.motion.getPosition(chain_name, 2, True)[0:3])
            diff = np.linalg.norm(reached_point-goal_point)
            count += 1
            print(count)
            if count > 10:
                break
        print('Distance to approach point %s' % diff)
        goal_point = np.asarray(grab_point[0:3])
        print('Grab point %s' % goal_point)
        reached_point = np.asarray(self.robot.motion.getPosition(chain_name, 2, True)[0:3])
        diff = np.linalg.norm(reached_point-goal_point)
        count = 0

        while diff > distance_tolerance_grab:
            interval = diff * 15
            self.robot.motion.positionInterpolations([chain_name], 2, grab_point, motion_mask, [interval])
            reached_point = np.asarray(self.robot.motion.getPosition(chain_name, 2, True)[0:3])
            diff = np.linalg.norm(reached_point-goal_point)
            count += 1
            if count > 10:
                print('Failed to reach the object')
                print('Distance to grab point %s' % diff)
                return -1, None
        print('Distance to grab point %s' % diff)
        self.robot.motion.setAngles(hand_name, 0.0, 0.7)
        time.sleep(1.0)

        self.robot.motion.positionInterpolations([chain_name], 2, lift_point, motion_mask, 0.5)
        self.robot.motion.positionInterpolations(["Torso"], 2, behavior_pose, motion_mask, 1)
        self.robot.motion.wbEnableEffectorControl(chain_name, False)
        return 1, grab_point

    def close_hand(self, direction):
        if direction == -1:
            hand_name = 'RHand'
            # chain_name = 'RArm'
        else:
            hand_name = 'LHand'
            # chain_name = 'LArm'
        self.robot.motion.setAngles(hand_name, 0.0, 0.3)
        # lift hand
        # current_point = np.asarray(self.robot.motion.getPosition(chain_name, 2, True)[0:3])
        # current_point[2] += 0.05
        # self.robot.motion.wbEnableEffectorControl(chain_name, True)
        # self.robot.motion.positionInterpolations([chain_name], 2, current_point, 7, 1)
        # self.robot.motion.wbEnableEffectorControl(chain_name, False)

    def grab_assisted(self, direction):
        # TODO: change to use the hand based on direction
        hand_name = ''
        count = 0
        # TODO: debug tactile sensor 
        while True:
            count += 1
            val1 = self.robot.memory.getData("FrontTactilTouched")
            val2 = self.robot.memory.getData("MiddleTactilTouched")
            val3 = self.robot.memory.getData("RearTactilTouched")
            if val1 or val2 or val3:
                if count > 3:
                    break
            if direction == -1:
                hand_name = 'RHand'
            #     val1 = self.robot.memory.getData("HandRightLeftTouched")
            #     val2 = self.robot.memory.getData("HandRightBackTouched")
            #     val3 = self.robot.memory.getData("HandRightRightTouched")
            else:
                hand_name = 'LHand'
            #     val1 = self.robot.memory.getData("HandLeftLeftTouched")
            #     val2 = self.robot.memory.getData("HandLeftBackTouched")
            #     val3 = self.robot.memory.getData("HandLeftRightTouched")
        if hand_name:
            self.robot.motion.setAngles(hand_name, 0.0, 0.3)
        # TODO: return flag and return point

    def put_object_back(self, return_point, object_name, direction):
        if object_name == 'Frog':
            if direction == -1:
                bhv = 'adoregestures-359a39/Return (Frog, right)'
            else:
                bhv = 'adoregestures-359a39/Return (Frog, left)'
            self.robot.behavior.runBehavior(bhv)

            return

        if not return_point:
            print('There is no return point')
            if direction == -1:
                if object_name == 'Cup':
                    return_point = [0.11727181077003479, 0.12741899490356445, 0.2591964900493622, -1.2810202836990356, 0.5643281936645508, -0.009264674037694931]
                else:
                    return_point = [0.21650418639183044, -0.06975226104259491, 0.3368774652481079, 0.051720183342695236, 0.33109599351882935, 0.26362404227256775]
            else:
                direction = 1
                if object_name == 'Cup':
                    return_point = [0.18634453415870667, 0.024101756513118744, 0.33464834094047546, -1.7019013166427612, 0.10675026476383209, -0.437185138463974]
                else:
                    return_point = [0.20216235518455505, 0.04532930999994278, 0.33338943123817444, 0.025482231751084328, 0.20920389890670776, -0.5539942383766174]

        print('Return point %s' % return_point)
        print('Direction %s' % direction)
        if direction == -1:
            hand_name = 'RHand'
            chain_name = 'RArm'
        else:
            hand_name = 'LHand'
            chain_name = 'LArm'
        return_point[0] += 0.06
        return_point[2] += 0.03

        return_point_1 = return_point[:]
        return_point_1[0] += 0.0
        return_point_1[2] += 0.02
        return_point_2 = return_point[:]
        return_point_2[0] -= 0.02
        return_point_2[1] += direction*0.02
        return_point_2[2] += 0.05
        self.robot.motion.wbEnableEffectorControl(chain_name, True)

        points_list = [return_point_2, return_point_2, return_point_1, return_point_1, return_point, return_point]
        times_list = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

        self.robot.motion.wbEnableEffectorControl(chain_name, True)
        self.robot.motion.positionInterpolations([chain_name], 2, points_list, 15, times_list)
        self.robot.motion.wbEnableEffectorControl(chain_name, False)
        self.robot.motion.setAngles(hand_name, 1.0, 0.7)
        if direction == -1:
            self.robot.motion.setAngles('RShoulderRoll', -30*np.pi/180, 0.1)
        else:
            self.robot.motion.setAngles('LShoulderRoll', 30*np.pi/180, 0.1)
        time.sleep(2)
        return

        # return_point_1[2] += 0.02
        # return_point_1[1] += direction*0.05
        # return_point_2[2] += 0.02
        # return_point_2[1] += direction*0.05
        # points_list_2 = [return_point_1, return_point_1, return_point_2, return_point_2]
        # times_list_2 = [1.0, 1.5, 2.5, 3.0]
        # self.robot.motion.wbEnableEffectorControl(chain_name, True)
        # self.robot.motion.positionInterpolations([chain_name], 2, points_list_2, 15, times_list_2)
        # self.robot.motion.wbEnableEffectorControl(chain_name, False)

    def cleanup(self):
        print('Cleanup grabnao')
        self.robot.cleanup()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config_file = args.config
    # object_name = 'Cylinder'
    object_name = 'Frog'
    # object_name = 'Cup'
    grabber = GrabNAO(config_file)
    grabber.init_pose()
    ret_val, [grab_point_3d, grab_direction] = grabber.calculate_3d_grab_point(object_name)

    if ret_val == 1:
        ret_val_grabbing, grab_point = grabber.grab_object(object_name, grab_point_3d, grab_direction)

        if ret_val_grabbing == 1:
            if grab_direction == -1:
                hand = 'right'
            else:
                hand = 'left'

            if object_name == 'Cylinder' or object_name == 'Frog':
                behavior = 'Frog'
            else:
                behavior = 'Drinking'
            behavior_to_run = behavior + ' (%s)' % hand
            print(behavior_to_run)
            grabber.robot.behavior.runBehavior(behavior_to_run)
            print('Grab point %s' % grab_point)
            grabber.put_object_back(grab_point, grab_direction)

            grabber.robot.behavior.runBehavior('Sada ti (%s)' % hand)

    grabber.init_pose()



