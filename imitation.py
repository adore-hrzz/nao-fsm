#!/usr/bin/env python
# -*- coding: utf-8 -*-

from transitions import Machine

class Imitation(Machine):

    def __init__(self,initial='init',interactive=True):

        states = ['init','invite','grab','introduce','demo',
                  'release','encourage','recognize','recourage', 
                  'bravo','end']

        transitions = [ {'trigger': 'start', 'source': 'init', 'dest': 'invite'},
                        {'trigger': 'success', 'source': 'invite', 'dest': 'grab'},
                        {'trigger': 'success', 'source': 'grab', 'dest': 'introduce'},
                        {'trigger': 'success', 'source': 'introduce', 'dest': 'demo'},
                        {'trigger': 'success', 'source': 'demo', 'dest': 'release'},
                        {'trigger': 'success', 'source': 'release', 'dest': 'encourage'},
                        {'trigger': 'success', 'source': 'encourage', 'dest': 'recognize'},
                        {'trigger': 'success', 'source': 'recognize', 'dest': 'bravo'},
                        {'trigger': 'fail', 'source': 'recognize', 'dest': 'recourage'},
                        {'trigger': 'success', 'source': 'recourage', 'dest': 'grab'},
                        {'trigger': 'repeat', 'source': 'bravo', 'dest': 'init'},
                        {'trigger': 'quit', 'source': '*', 'dest': 'end'},
                      ]

        Machine.__init__(self,states=states,transitions=transitions,initial=initial)

        self.interactive = interactive

    def on_enter_invite(self):
        """
        Invite person to approach the robot.
        """
        print("Inviting...")
        if not self.user_quit():
            self.success()

    def on_enter_grab(self):
        """
        Grab the object.
        """
        print('Grabbing...')
        if not self.user_quit():
            self.success()

    def on_enter_introduce(self):
        """
        Introduce the task.
        """
        print('Introducing...')
        if not self.user_quit():
            self.success()

    def on_enter_demo(self):
        """
        Demonstrate the gesture.
        """
        print('Demonstrating...')
        if not self.user_quit():
            self.success()

    def on_enter_release(self):
        """
        Release the object.
        """
        print('Releasing...')
        if not self.user_quit():
            self.success()

    def on_enter_encourage(self):
        """
        Encourage the person to repeat the gesture.
        """
        print('Encouraging...')
        if not self.user_quit():
            self.success()

    def on_enter_recognize(self):
        """
        Run gesture recognition.
        """
        print('Recognizing...')
        if not self.user_quit():
            self.success()

    def on_enter_recourage(self):
        """
        Re-encourage the person if gesture was not recognized.
        """
        print('Recouraging...')
        if not self.user_quit():
            self.success()

    def on_enter_bravo(self):
        """
        Compliment the person if gesture was recognized.
        """
        print('Bravo!')
        
        if not self.interactive or self.user_quit():
            # Right now, there's no way to break out of
            # non-interactive mode, so repeating is disabled
            # in that mode
            self.quit()
        else:
            self.repeat()

    def on_enter_end(self):
        """
        We're done.
        """
        print('Bye-bye!')

    def user_quit(self):
        """
        Wait for user input before continuing.
        """
        user_input = ''
        message = 'Completed state {0}.'.format(self.state)

        if self.interactive:
            user_input = raw_input(message + ' Hit <Enter> to continue or type q<Enter> to quit.')
        else:
            print(message)

        return user_input

if __name__ == '__main__':

    im = Imitation()
    im.start()

