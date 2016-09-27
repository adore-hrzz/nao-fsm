#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transitions import Machine
#from transitions.extensions import GraphMachine as Machine

from naoqi import ALProxy

import argparse

states = ['init','invite','grab','introduce','demo',
          'release','encourage','recognize','recourage', 
          'bravo','end']

class Imitation(Machine):

    def __init__(self,initial='init',interactive=True):

        # Matching transitions are searched for sequentially
        # Therefore, the wildcard transition '*' has to be defined last
        transitions = [ {'trigger': 'start', 'source': 'init', 'dest': 'invite'},
                        {'trigger': 'success', 'source': 'invite', 'dest': 'grab', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'grab', 'dest': 'introduce', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'introduce', 'dest': 'demo', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'demo', 'dest': 'release', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'release', 'dest': 'encourage', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'encourage', 'dest': 'recognize', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'recognize', 'dest': 'bravo', 'unless': 'user_quit'},
                        {'trigger': 'fail', 'source': 'recognize', 'dest': 'recourage', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'recourage', 'dest': 'grab', 'unless': 'user_quit'},
                        {'trigger': 'success', 'source': 'bravo', 'dest': 'init',  'unless': 'user_quit'},
                        {'trigger': 'success', 'source': '*', 'dest': 'end'}
                      ]

        Machine.__init__(self,states=states,transitions=transitions,initial=initial)

        self.interactive = interactive
        
        self.behaviors = {'invite': '', 'introduce': '', 'demo': '', 
                          'encourage': '', 'recourage': '', 'bravo': ''}

        self.ip = 'helga.local'
        self.port = 9559
        self.behavior_proxy = ALProxy('ALBehaviorManager',self.ip,self.port)

    def on_enter_invite(self):
        """
        Invite person to approach the robot.
        """
        print("Inviting...")
        self.success()

    def on_enter_grab(self):
        """
        Grab the object.
        """
        print('Grabbing...')
        self.success()

    def on_enter_introduce(self):
        """
        Introduce the task.
        """
        print('Introducing...')
        self.success()

    def on_enter_demo(self):
        """
        Demonstrate the gesture.
        """
        self.behavior_proxy.runBehavior('Drinking (left)')
        self.success()

    def on_enter_release(self):
        """
        Release the object.
        """
        print('Releasing...')
        self.success()

    def on_enter_encourage(self):
        """
        Encourage the person to repeat the gesture.
        """
        print('Encouraging...')
        self.success()

    def on_enter_recognize(self):
        """
        Run gesture recognition.
        """
        print('Recognizing...')
        self.success()

    def on_enter_recourage(self):
        """
        Re-encourage the person if gesture was not recognized.
        """
        print('Recouraging...')
        self.success()

    def on_enter_bravo(self):
        """
        Compliment the person if gesture was recognized.
        """
        print('Bravo!')
        
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
    parser.add_argument('--config', help='Configuration file name.')
    args = parser.parse_args()

    im = Imitation()
    #im.graph.draw('state_diagram.png',prog='dot')

    im.start()

