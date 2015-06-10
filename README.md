# nao-fsm
.. Finite State Machine documentation master file, created by Luka Malovan
   sphinx-quickstart on Tue Mar 31 11:58:57 2015.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**Welcome to NAO Finite State Machine's documentation!**
========================================================

State machine
=============

Module is executing object grabbing and tracking protocol with NAO robot. 
Protocol consists of several stages which is why module is suitable to be derivaded as a finite state machine.

In first stage of protocol NAO has to find the object which is placed near him or in proximity. 
After that, NAO has to grab the object, do a gesture with it that corespondes the type of object, and put the object back to its place.
Lastly, NAO has to track the object and detect whether the same gesture he did with it has been repeated or not.

All states of state machine and their corespondance to protocol are described below.

State machine consists of 6 states:

    1) **Start**: Initial state of FSM, this is the start point of module. In Start, all initializations are conducted: parameters from configuration file are imported, communication parameters for manipulation with NAO are defined, ect.


    2) **Initial**: In this state NAO is set to its initial position, standing position.


    3) **Search**: Searching for object with NAO's camera. If object is found close to NAO, module goes to next state, if not, NAO looks for object in distance and around him, if object is found NAO moves towards it until he hits obstecale his foot bumper. If object is not found at all, module goes back to initial state.  


    4) **Image processing**: Image processing of object image taken with NAO's camera. Image processing consist of segmentation, to exclude object from surrounding, and calculating grab point on object. If image processing goes wrong, module goes back to inital state. 


    5) **Object manipulation**: In this state all object manipulation is done. NAO grabs object, does the coresponding gesture and puts object back to it's place.


    6) **Object tracking**: Module is tracking object's trajectory waiting to detect same gesture that NAO has done in previous step. After this, module goes back to initial state.

Graphical representation of state machine:

.. figure::  https://github.com/maloL/nao-fsm/blob/master/Flowchart.jpeg
   :align:   center

Module description
==================

.. automodule:: NAO_Fsm
   :members:

How to use
==========

How to start the module : "python NAO_Fsm.py Config.ini".

**NAO_Fsm** is the name of module. 

**Config.ini** is a configuration file that holds parameters that need to be changed by user depending on situation requirements for specific module behavior and users preferences. 
It's parametres are:

	- **Volume** - Volume of NAO's speakers. Values in range [0, 100]. 
	
	- **Mute** - Values 1 (True) or 0 (False) telling the module to mute NAO or to let it say certain information or instructions.

	- **Diagnostics** - Values 1 (True) or 0 (False) telling the module to show or to hide diagnostic informations concerning object searching data, image processing data, object tracking data and some other.	

	- **Height** - Height of a table that object is placed on. 

	- **IP** - NAOs IP adress on wifi network or cable connection

	- **PORT** - NAOs communication port.

Requirements
============

List of required Python modules and libraries:
	- naoqi - version 1.14
	- numpy 
	- cv2 
	- configparser - parameters importing from configuration file
	- argparser - enables argument handling when starting module from command window 
	- transition - state machine package
	- pymorph
	- almath

Module is programmed in Python 2.7.

Also, module reqires two modules made for image processing to be placed in same directory as this module.
Those modules are: **LinesAndPlanes.py** and **NaoImageProcessing.py**.

Aditional contest
=================

In order to ensure that module can work in different enviroments new image sets of object should be taken before using the module.
This is done by running **takeObjectPicture.py** script which, as the name says, takes pictures of object and environment so that NAO could find object more easily.

How to run the script: "python takeObjectPicture.py Config.ini"

This script is using configuration file as well, the same as the FSM module, but different parameters. User must define those parameters before running the script in order to
get the best images and segmentation.
