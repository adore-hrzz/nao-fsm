import time
import sys
import motion
import almath
import math
import Image, ImageDraw
from naoqi import ALProxy
import numpy as np
import cv2
import NaoImageProcessing

def getLineEquation(motionproxy,space,u,v):
    """
    Gets a line equation in arbitrary space from given coordinates in image space
    Use for intersecting with geometric primitives to obtain spatial coordinates
    """
    transform=motionproxy.getTransform("CameraBottom",space,True)
    transformList=almath.vectorFloat(transform)
    robotToCamera = almath.Transform(transformList)

    cx=319.5
    cy=239.5
    f=563.19

    vec=np.array([f,(cx-v),(cy-u)])
    vec=vec/math.sqrt(pow(f,2)+pow(cx-v,2)+pow(cy-u,2))
    pointLocationTransform=robotToCamera*almath.Transform.fromPosition(vec[0],vec[1],vec[2])
    cameraLocationTransform=robotToCamera

    k=np.array([cameraLocationTransform.r1_c4, cameraLocationTransform.r2_c4, cameraLocationTransform.r3_c4])
    v=np.array([pointLocationTransform.r1_c4, pointLocationTransform.r2_c4, pointLocationTransform.r3_c4])-k

    #line equation is x=vt+k, v, k, x 3d vectors, t scalar, t=0 for camera location
    return (v,k)

########################################################################################################################

def intersectWithPlane(lineEquation, space, planecoefficients):
    """
    Intersect line equation with plane specified by plane coefficients (A,B,C,D)
    Returns point of intersection as a numpy array
    """
    v=lineEquation[0]
    k=lineEquation[1]

    #plane defined as P*x+D=0, P=[A,B,C], x 3d point vector
    P=np.array([planecoefficients[0], planecoefficients[1], planecoefficients[2]])

    #intersection at t=-(Pk+D)/(Pv)

    t=-1*(np.dot(P,k)+ planecoefficients[3])/np.dot(P,v)
    point=v*t+k
    return point

########################################################################################################################

def planePlaneIntersection(motionproxy,h,d,centroidPoint, frontPoint):
    """
    Thing
    """

    centroidLine=getLineEquation(motionproxy,2,centroidPoint[0],centroidPoint[1])
    frontLine=getLineEquation(motionproxy,2,frontPoint[0],frontPoint[1])

    tablePlane=(0,0,1,-h)

    frontXYZ=intersectWithPlane(frontLine,2,tablePlane)
    vx=frontLine[0][0]
    vy=frontLine[0][1]
    q=math.sqrt(pow(vx,2)+pow(vy,2))
    vx=vx/q
    vy=vy/q
    D=-(vx*(frontXYZ[0]+d*vx/2)+vy*(frontXYZ[1]+d*vx/2))
    objectPlane=(vx,vy,0,D)
    objectXYZ=intersectWithPlane(centroidLine,2,objectPlane)
    return objectXYZ
