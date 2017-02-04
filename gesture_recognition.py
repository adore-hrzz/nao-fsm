import numpy as np
from hmmlearn import hmm
from naoqi import ALProxy, ALModule
import vision_definitions


class ObjectTrackerModule(ALModule):
    def __init__(self, name, robot):
        ALModule.__init__(self, name)
        self.gestureProxy = ALProxy("NAOObjectGesture")
        self.robot = robot
        self.kindNames = []
        self.count = 0
        self.data = []

    def start(self, camId, focus=False):
        self.gestureProxy.startTracker(15, camId)
        if focus:
            self.motionProxy.setStiffnesses("Head", 1.0)
            self.gestureProxy.focusObject(-1)

    def stop(self):
        self.gestureProxy.stopTracker()
        self.gestureProxy.stopFocus()

    def load(self, path, name):
        self.gestureProxy.loadDataset(path)
        self.kindNames.append(name)
        self.gestureProxy.trackObject(name, -1)
        self.robot.memory.subscribeToEvent(name, "object_tracker", name, "on_obj_get")

    def on_obj_get(self, key, value, message):
        if value[1]:
            self.data.append(value[3])
        self.count += 1

    def unload(self):
        self.gestureProxy.stopTracker()
        for name in self.kindNames:
            self.gestureProxy.removeObjectKind(0)
            self.gestureProxy.removeEvent(name)


class PicoSubscriberModule(ALModule):
    def __init__(self, name, camera=vision_definitions.kTopCamera, resolution=vision_definitions.kVGA, fps=5):
        ALModule.__init__(self, name)

        self.name = name
        self.memory = ALProxy('ALMemory')
        self.pico_proxy = ALProxy('PicoModule')
        self.pico_proxy.setActiveCamera(camera)
        self.pico_proxy.setResolution(resolution)
        self.pico_proxy.setFrameRate(fps)
        self.count = 0
        self.data = []

    def add_object(self, object_type, path_to_classifier):
        self.pico_proxy.addClassifier(object_type, path_to_classifier, -1, -1, -1, -1, -1)

    def start(self):
        # self.pico_proxy.subscribe(self.name)
        self.memory.subscribeToEvent("picoDetections", self.name, "on_event_raised")

    def stop(self):
        self.memory.unsubscribeToEvent("picoDetections", self.name)
        # self.pico_proxy.unsubscribe(self.name)

    def on_event_raised(self, key, value, message):
        self.memory.unsubscribeToEvent("picoDetections", self.name)
        self.data = value
        self.count += 1
        # print('Event no %s detected' % self.count)
        self.memory.subscribeToEvent("picoDetections", self.name, "on_event_raised")


def load_model(path, name):
    trans_mat = np.load(path+'/%s_' % name+'hmm_transition_matrix.npy')
    emission_mat = np.load(path+'/%s_' % name+'hmm_emission_matrix.npy')
    start_prob = np.load(path+'/%s_' % name+'hmm_start_probability.npy')

    num_states = trans_mat.shape[0]

    model = hmm.MultinomialHMM(n_components=num_states, init_params='')
    model.transmat_ = trans_mat
    model.emissionprob_ = emission_mat
    model.startprob_ = start_prob

    return model


def calculate_features(observations, num_bins):
    angle_delta = 2*np.pi/num_bins
    c_x = 0
    c_y = 0
    for point in observations:
        c_x += point[0]
        c_y += point[1]

    c_x /= len(observations)
    c_y /= len(observations)

    theta = [0]*(len(observations)-1)

    for i in range(0, len(observations)-1):
        x_0 = observations[i][0]-c_x
        y_0 = observations[i][1]-c_y
        x_1 = observations[i+1][0]-c_x
        y_1 = observations[i+1][1]-c_y

        theta[i] = np.arctan2(y_1-y_0, x_1-x_0) % (2*np.pi)

    features = [int(np.floor(x/angle_delta)) for x in theta]

    return features


def find_gesture_segments(non_gesture_model, gesture_model, features, win_size):
    start_index = 0
    end_index = win_size
    segments = []
    if gesture_model.score(np.atleast_2d(features)) > non_gesture_model.score(np.atleast_2d(features)):
        return [[0, len(features)]]
    last_was = False
    while start_index < len(features) and end_index < len(features)+win_size:
        sequence_to_eval = np.atleast_2d(features[start_index:min(end_index, len(features))])
        score_ngm = non_gesture_model.score(sequence_to_eval)
        score_gesture = gesture_model.score(sequence_to_eval)
        # print(i, start_index[i], end_index[i], score_ngm, score_gesture)
        if score_gesture > score_ngm:
            last_was = True
            end_index += win_size
        else:
            if last_was and not start_index == end_index-win_size:
                # print('Segments before %s' % segments)
                # print('Appending %s to %s' % ([start_index[i], end_index[i]-win_size], i))
                segments.append([start_index, end_index-win_size])
                # print('Segments after %s' % segments)
                start_index = end_index - win_size
            else:
                start_index += max(win_size/2, 1)
                end_index += max(win_size/2, 1)
            last_was = False
    if last_was:
        segments.append([start_index, min(end_index-win_size, len(features))])
    # raw_input('test')
    return segments


if __name__ == '__main__':
    print('This does nothing by itself.')
