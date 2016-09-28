#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transitions import Machine
from grabnao import GrabNAO, ObjectGestureModule
import time
#from transitions.extensions import GraphMachine as Machine

from naoqi import ALProxy

import argparse
from ConfigParser import ConfigParser

states = ['init','invite','grab','introduce','demo',
          'release','encourage','recognize','recourage', 
          'bravo','end']

ObjectGesture = []


class Imitation(Machine):

    def __init__(self, gesture, host, objects, initial='init', hand='left', interactive=True):

        # Matching transitions are searched for sequentially
        # Therefore, the wildcard transition '*' has to be defined last
        transitions = [ {'trigger': 'start', 'source': 'init', 'dest': 'invite', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'invite', 'dest': 'grab', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'grab', 'dest': 'introduce', 'unless': 'user_quit'},
                        {'trigger': 'fail', 'source': 'grab', 'dest': 'grab', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'introduce', 'dest': 'demo', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'demo', 'dest': 'release', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'release', 'dest': 'encourage', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'encourage', 'dest': 'recognize', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'recognize', 'dest': 'bravo', 'unless': 'user_quit'},
                        {'trigger': 'fail', 'source': 'recognize', 'dest': 'recourage', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'recourage', 'dest': 'grab', 'unless': 'user_quit'},
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
        self.hand = 'left'
        self.object_name = parser.get('Gesture','object')
        self.gesture = parser.get('Gesture','name')
        
        # TODO: yaml config might be more convenient
        # Behavior names have to match state names!
        self.behaviors = {'left': {'invite' : parser.get('Lefthanded','invite'),
                                   'introduce' : parser.get('Lefthanded','introduce'),
                                   'demo' : parser.get('Lefthanded','demo'),
                                   'encourage' : parser.get('Lefthanded','encourage'),
                                   'bravo' : parser.get('Lefthanded','bravo')
                               },
                          'right': {'invite' : parser.get('Righthanded','invite'),
                                   'introduce' : parser.get('Righthanded','introduce'),
                                   'demo' : parser.get('Righthanded','demo'),
                                   'encourage' : parser.get('Righthanded','encourage'),
                                   'bravo' : parser.get('Righthanded','bravo')
                               }
                      }

        self.grabber = GrabNAO(objects,host)
        global ObjectGesture
        ObjectGesture = ObjectGestureModule('ObjectGesture', self.grabber.robot.broker)

    def on_enter_invite(self):
        """
        Invite person to approach the robot.
        """
        print("Inviting...")
        self.grabber.init_pose()
        bhv = self.behaviors[self.hand][self.state]
        if bhv:
            self.grabber.robot.behavior.runBehavior(bhv)        
        self.success()

    def on_enter_grab(self):
        """
        Grab the object.
        """
        print('Grabbing...')
        ret_val_calc = self.grabber.calculate_3d_grab_point(self.object_name)
        if ret_val_calc[0] == -1:
            print('Grab point calculation failed')
            self.fail()
        else:
            grab_point, direction = ret_val_calc[1]
            ret_val_grab = self.grabber.grab_object(self.object_name, grab_point, direction)

            if ret_val_grab == -1:
                print('Grabbing failed')
                self.fail()
            else:
                self.grab_point = ret_val_grab[1]
                self.direction = direction

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
        self.success()

    def on_enter_release(self):
        """
        Release the object.
        """
        print('Releasing...')
        self.grabber.put_object_back(self.grab_point, self.direction)
        #self.grabber.init_pose()
        self.success()

    def on_enter_encourage(self):
        """
        Encourage the person to repeat the gesture.
        """
        print('Encouraging...')
        bhv = self.behaviors[self.hand][self.state]
        if bhv:
            self.grabber.robot.behavior.runBehavior(bhv)
        self.success()

    def on_enter_recognize(self):
        """
        Run gesture recognition.
        """
        global ObjectGesture
        ObjectGesture.load('/home/nao/ImageSets/%s/' % self.object_name, self.object_name, 'ObjectGesture')
        ObjectGesture.start_tracker(0, True)
        print('Recognizing...')
        user_input = raw_input("Tracking the child's gesture. Hit <Enter> to stop.")
        time_str = './logs/'+time.strftime("%Y%m%d-%H%M%S")+'.txt'
        ObjectGesture.write_data(time_str)
        ObjectGesture.stop_tracker()
        ObjectGesture.unload()
        self.success()

    def on_enter_recourage(self):
        """
        Re-encourage the person if gesture was not recognized.
        """
        print('Recouraging...')
        bhv = self.behaviors[self.hand][self.state]
        if bhv:
            self.grabber.robot.behavior.runBehavior(bhv)
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run the immitation protocol.')
    parser.add_argument('--gesture', required=True, help='File containing the gesture descriptor.')
    parser.add_argument('hostname', help='The hostname or ip address of the robot we are working with.')
    parser.add_argument('--initial-state', help='Start from this state (states).', default='init')
    parser.add_argument('--hand', help='The default hand. This is important if we start from a state other than init.', default='left')
    parser.add_argument('--objects', default='objects.cfg', help='File containing object descriptors.')
    args = parser.parse_args()

    im = Imitation(args.gesture, args.hostname, args.objects, args.initial_state, args.hand)
    #im.graph.draw('state_diagram.png',prog='dot')

    if args.initial_state == 'init':
        im.start()
    else:
        im.success()