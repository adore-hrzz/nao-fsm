#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transitions import Machine
from grabnao import GrabNAO
from gesture_recognition import PicoSubscriberModule
import time
from naoqi import ALBroker
# from transitions.extensions import GraphMachine as Machine

from naoqi import ALProxy

import argparse
from ConfigParser import ConfigParser

states = ['init','invite','grab','assist','introduce','demo',
          'release','encourage','recognize','recourage', 
          'bravo','end']


class Imitation(Machine):

    def __init__(self, gesture, host, objects, initial='init', hand='left', interactive=True):

        # Matching transitions are searched for sequentially
        # Therefore, the wildcard transition '*' has to be defined last
        transitions = [ {'trigger': 'start', 'source': 'init', 'dest': 'invite', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'invite', 'dest': 'grab'}, #'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'grab', 'dest': 'introduce'}, #'unless': 'user_quit'},
                        {'trigger': 'fail', 'source': 'grab', 'dest': 'assist'},
                        {'trigger': 'success', 'source': 'assist', 'dest': 'introduce'}, # 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'introduce', 'dest': 'demo'},
                        {'trigger': 'success', 'source': 'demo', 'dest': 'release'},
                        {'trigger': 'success', 'source': 'release', 'dest': 'encourage'},
                        {'trigger': 'success', 'source': 'encourage', 'dest': 'recognize'},
                        {'trigger': 'success', 'source': 'recognize', 'dest': 'bravo'},
                        {'trigger': 'fail', 'source': 'recognize', 'dest': 'recourage'},
                        {'trigger': 'success', 'source': 'recourage', 'dest': 'grab'},# 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'bravo', 'dest': 'init',  'unless': 'user_quit'},
                        {'trigger': 'success', 'source': '*', 'dest': 'end'},
                        {'trigger': 'fail', 'source': '*', 'dest': 'end'}
                      ]

        Machine.__init__(self,states=states,transitions=transitions,initial=initial)

        self.interactive = interactive
        
        parser = ConfigParser()
        parser.readfp(open(gesture))

        self.grab_point = None
        self.direction = None
        # used to exit after three fails
        self.demonstration_count = 0
        self.hand = hand
        self.object_name = parser.get('Gesture','object')
        self.gesture = parser.get('Gesture','name')
        
        # TODO: yaml config might be more convenient
        # Behavior names have to match state names!
        self.behaviors = {'left': {'invite' : parser.get('Lefthanded','invite'),
                                   'introduce' : parser.get('Lefthanded','introduce'),
                                   'demo' : parser.get('Lefthanded','demo'),
                                   'encourage' : parser.get('Lefthanded','encourage'),
                                   'recourage' : parser.get('Lefthanded', 'recourage'),
                                   'bravo' : parser.get('Lefthanded','bravo'),
                                   'assist' : parser.get('Lefthanded', 'assist')
                               },
                          'right': {'invite' : parser.get('Righthanded','invite'),
                                    'introduce' : parser.get('Righthanded','introduce'),
                                    'demo' : parser.get('Righthanded','demo'),
                                    'encourage' : parser.get('Righthanded','encourage'),
                                    'recourage' : parser.get('Righthanded','recourage'),
                                    'bravo' : parser.get('Righthanded','bravo'),
                                    'assist' : parser.get('Righthanded', 'assist')
                               }
                      }
        self.grabber = GrabNAO(objects,host)

    def on_enter_invite(self):
        """
        Invite person to approach the robot.
        """
        print("Inviting...")
        self.grabber.init_pose()
        # bhv = self.behaviors[self.hand][self.state]
        # if bhv:
        #     self.grabber.robot.behavior.runBehavior(bhv)
        self.success()

    def on_enter_grab(self):
        """
        Grab the object.
        """
        print('Grabbing...')
        self.grabber.init_pose()
        ret_val_calc = self.grabber.calculate_3d_grab_point(self.object_name)
        if ret_val_calc[0] == -1:
            print('Grab point calculation failed')
            self.fail()
        else:
            grab_point, self.direction = ret_val_calc[1]
            ret_val_grab = self.grabber.grab_object(self.object_name, grab_point, self.direction)

            if ret_val_grab == -1:
                print('Grabbing failed')
                self.fail()
            else:
                self.grab_point = ret_val_grab[1]
                user_input = raw_input("Robot is grabbing the object. Hit <Enter> to confirm successful grab.")
                if user_input == '':
                    # Empty input (only <Enter> is interpretd as success)
                    # TODO: add close hand function
                    self.grabber.close_hand(self.direction)
                    self.success()
                else:
                    # Any other input is interpreted as failure
                    self.fail()

    def on_enter_assist(self):
        """
        Assist robot with grabbing
        """
        print ('Test, robot is being assisted, press <Enter> to continue')
        if self.direction == -1:
            self.hand = 'right'
        else:
            self.hand = 'left'
        bhv = self.behaviors[self.hand][self.state]
        if bhv:
            self.grabber.robot.behavior.runBehavior(bhv)
        self.grabber.grab_assisted(self.direction)
        self.success()

    def on_enter_introduce(self):
        """
        Introduce the task.
        """
        print('Introducing...')
        bhv = self.behaviors[self.hand][self.state]
        if bhv:
            self.grabber.robot.behavior.runBehavior(bhv)
        self.success()

    def on_enter_demo(self):
        """
        Demonstrate the gesture.
        """
        if self.direction == -1:
            self.hand = 'right'
        else:
            self.hand = 'left'

        bhv = self.behaviors[self.hand][self.state]
        if bhv:
            self.grabber.robot.behavior.runBehavior(bhv)
        self.demonstration_count += 1
        self.success()

    def on_enter_release(self):
        """
        Release the object.
        """
        print('Releasing...')
        self.grabber.put_object_back(self.grab_point, self.object_name, self.direction)
        #self.grabber.init_pose()
        self.success()

    def on_enter_encourage(self):
        """
        Encourage the person to repeat the gesture.
        """
        print('Encouraging...')
        bhv = self.behaviors[self.hand][self.state]
        if bhv:
            self.grabber.robot.behavior.post.runBehavior(bhv)
        self.success()

    def on_enter_recognize(self):
        """
        Run gesture recognition.
        """
        print('Recognizing...')

        time_str = time.strftime("%Y%m%d-%H%M%S")+'-%s-%s.avi' % (self.object_name, self.gesture)
        self.grabber.robot.motion.setAngles('HeadPitch', -0.1, 0.3)

        self.grabber.robot.video_recorder.setCameraID(0)
        self.grabber.robot.video_recorder.startRecording('/home/nao/recordings/', time_str)

        user_input = raw_input("Tracking the child's gesture. Hit <Enter> to stop.")
        _, path = self.grabber.robot.video_recorder.stopRecording()
        print('Video saved to: %s' % path)
        if user_input == '':
            # Empty input (only <Enter> is interpretd as success)
            self.success()
        else:
            # Any other input is interpreted as failure
            self.fail()

    def on_enter_recourage(self):
        """
        Re-encourage the person if gesture was not recognized.
        """
        print('Recouraging...')
        # TODO: make this a parameter
        if self.demonstration_count >= 3:
            self.fail()
        else:
            bhv = self.behaviors[self.hand][self.state]
            if bhv:
                self.grabber.robot.behavior.post.runBehavior(bhv)
        self.success()

    def on_enter_bravo(self):
        """
        Compliment the person if gesture was recognized.
        """
        print('Bravo!')
        bhv = self.behaviors[self.hand][self.state]
        if bhv:
            self.grabber.robot.behavior.runBehavior(bhv)
        self.success()

    def on_enter_end(self):
        """
        We're done.
        """
        print('Bye-bye!')
        self.grabber.cleanup()

    def user_quit(self):
        """
        Checks user input. Returns False if the user only pressed <Enter>.
        Returns True otherwise.
        """
        user_input = ''
        message = 'Completed state {0}.'.format(self.state)

        if self.interactive:
            user_input = raw_input(message + ' Hit <Enter> to continue or type q<Enter> to quit.')
        else:
            print(message)

        if user_input:
            return True
        else:
            return False

    def cleanup(self):
        """
        cleaning up
        """
        self.grabber.cleanup()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the immitation protocol.')
    parser.add_argument('--gesture', required=True, help='File containing the gesture descriptor.')
    parser.add_argument('hostname', help='The hostname or ip address of the robot we are working with.')
    parser.add_argument('--initial-state', help='Start from this state ({0}).'.format(states), default='init')
    parser.add_argument('--hand', help='The default hand. This is important if we start from a state other than init.', default='left')
    parser.add_argument('--objects', default='objects.cfg', help='File containing object descriptors.')
    args = parser.parse_args()

    im = Imitation(args.gesture, args.hostname, args.objects, args.initial_state, args.hand)
    #im.graph.draw('state_diagram.png',prog='dot')
    try:
        if args.initial_state == 'init':
            im.start()
        else:
            im.success()

        im.cleanup()

    except KeyboardInterrupt:
        im.cleanup()
        raise
