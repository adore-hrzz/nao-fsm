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

def get3Dpoint(motionproxy, space, u, v, planeCoeffs):

    cx = 322.47047623
    cy = 209.28243832
    f = (556.26572641+554.21845875)/2.0

    transform = motionproxy.getTransform("CameraBottom",space,True)
    T_camAL = np.asarray(transform)
    T_camAL = np.reshape(T_camAL, (4, 4))

    T_alCV = np.zeros((4, 4), dtype=np.float64)
    T_alCV[0, 2] = 1    # this means Z'= X
    T_alCV[1, 0] = 1    # this means X'= Y
    T_alCV[2, 1] = 1    # this means Y'= Z
    T_alCV[3, 3] = 1    # homogenous coordinates!

    T_camCv = np.dot(T_camAL, T_alCV)
    cam_point = np.asarray([T_camAL[0,3], T_camAL[1,3], T_camAL[2,3]])
    pix_point = np.asarray([cx-u, cy-v, f, 1])
    pix_point_transformed = np.dot(T_camCv, pix_point)
    pix_point_transformed = pix_point_transformed[0:3]

    pix_vec = pix_point_transformed - cam_point
    pix_vec = pix_vec / np.linalg.norm(pix_vec)

    A = planeCoeffs[0]
    B = planeCoeffs[1]
    C = planeCoeffs[2]
    D = planeCoeffs[3]

    t = (-1)*(A*T_camCv[0, 3] + B*T_camCv[1, 3] + C*T_camCv[2, 3] + D)/(A*pix_vec[0]+B*pix_vec[1]+C*pix_vec[2])
    point = cam_point + t*pix_vec

    return point



def getLineEquation(motionproxy,space,u,v):
    """
    Gets a line equation in arbitrary space from given coordinates in image space
    Use for intersecting with geometric primitives to obtain spatial coordinates
    """
    transform=motionproxy.getTransform("CameraBottom",space,True)
    transformList=almath.vectorFloat(transform)
    robotToCamera = almath.Transform(transformList)


    cx = 319.79047623
    cy = 209.28243832
    f = (555.66572641+553.81845875)/2.0

    vec=np.array([f,(cx-v),(cy-u)])
    vec=vec/math.sqrt(pow(f,2)+pow(cx-v,2)+pow(cy-u,2))

    #pointLocation = T_camCv*T_point
    pointLocationTransform=robotToCamera*almath.Transform.fromPosition(vec[0], vec[1], vec[2])
    cameraLocationTransform=robotToCamera

    k=np.array([cameraLocationTransform.r1_c4, cameraLocationTransform.r2_c4, cameraLocationTransform.r3_c4])
    v=np.array([pointLocationTransform.r1_c4, pointLocationTransform.r2_c4, pointLocationTransform.r3_c4])-k

    #line equation is x=vt+k, v, k, x 3d vectors, t scalar, t=0 for camera location
    return (v, k)

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
