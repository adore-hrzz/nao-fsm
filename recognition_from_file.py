# object tracking algorithm with trajectory points plotting on
# video stream window and gesture recognition

from naoqi import ALProxy, ALBroker, ALModule
import time
from vision_definitions import kVGA, kBGRColorSpace
import cv2 as opencv
import numpy as np
import random
from ghmm import *
import ConfigParser, argparse
import training

if __name__ == '__main__':
    filename = "gest209.txt"
    for i in range(1, 6):
        if i == 1:
            f = open('/home/luka/Documents/FER_projekt/Diplomski_rad/trained/drink/gesture_file.txt', 'r')
            state_num = int(f.readline())
            output_num = int(f.readline())
            gesture_treshold = f.readline()
            f.close()

            #[sigma, A, B, pi] = trening.matrices(state_num, output_num)
            #m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)

            cluster_data = training.cluster('/home/luka/Documents/FER_projekt/Diplomski_rad/' + filename, output_num) #trajektorija za evaluaciju
            #HMMfactory = HMMOpenFactory(GHMM_FILETYPE_XML)
            m = HMMOpen("/home/luka/Documents/FER_projekt/Diplomski_rad/trained/drink/m_file.xml")
            print m
            sigma = IntegerRange(0, output_num)

            test_seq = EmissionSequence(sigma, cluster_data.tolist())
            print test_seq
            print m.viterbi(test_seq)
            print gesture_treshold
            diff1 = abs(abs(float(m.viterbi(test_seq)[1])) - abs(float(gesture_treshold)))
            print "Diff 1: "
            print diff1
        elif i == 2:
            f = open('/home/luka/Documents/FER_projekt/Diplomski_rad/trained/frog/gesture_file.txt', 'r')
            state_num = int(f.readline())
            output_num = int(f.readline())
            gesture_treshold = f.readline()
            f.close()

            #[sigma, A, B, pi] = trening.matrices(state_num, output_num)
            #m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)

            cluster_data = training.cluster('/home/luka/Documents/FER_projekt/Diplomski_rad/' + filename, output_num) #trajektorija za evaluaciju
            #HMMfactory = HMMOpenFactory(GHMM_FILETYPE_XML)
            m = HMMOpen("/home/luka/Documents/FER_projekt/Diplomski_rad/trained/frog/m_file.xml")
            print m
            sigma = IntegerRange(0, output_num)

            test_seq = EmissionSequence(sigma, cluster_data.tolist())
            print test_seq
            print m.viterbi(test_seq)
            print gesture_treshold
            diff2 = abs(abs(float(m.viterbi(test_seq)[1])) - abs(float(gesture_treshold)))
            print "Diff 2: "
            print diff2
        elif i == 3:
            f = open('/home/luka/Documents/FER_projekt/Diplomski_rad/trained/neg1/gesture_file.txt', 'r')
            state_num = int(f.readline())
            output_num = int(f.readline())
            gesture_treshold = f.readline()
            f.close()

            #[sigma, A, B, pi] = trening.matrices(state_num, output_num)
            #m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)

            cluster_data = training.cluster('/home/luka/Documents/FER_projekt/Diplomski_rad/' + filename, output_num) #trajektorija za evaluaciju
            #HMMfactory = HMMOpenFactory(GHMM_FILETYPE_XML)
            m = HMMOpen("/home/luka/Documents/FER_projekt/Diplomski_rad/trained/neg1/m_file.xml")
            print m
            sigma = IntegerRange(0, output_num)

            test_seq = EmissionSequence(sigma, cluster_data.tolist())
            print test_seq
            print m.viterbi(test_seq)
            print gesture_treshold
            diff3 = abs(abs(float(m.viterbi(test_seq)[1])) - abs(float(gesture_treshold)))
            print "Diff 3: "
            print diff3
        elif i == 4:
            f = open('/home/luka/Documents/FER_projekt/Diplomski_rad/trained/neg2/gesture_file.txt', 'r')
            state_num = int(f.readline())
            output_num = int(f.readline())
            gesture_treshold = f.readline()
            f.close()

            #[sigma, A, B, pi] = trening.matrices(state_num, output_num)
            #m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)

            cluster_data = training.cluster('/home/luka/Documents/FER_projekt/Diplomski_rad/' + filename, output_num) #trajektorija za evaluaciju
            #HMMfactory = HMMOpenFactory(GHMM_FILETYPE_XML)
            m = HMMOpen("/home/luka/Documents/FER_projekt/Diplomski_rad/trained/neg2/m_file.xml")
            print m
            sigma = IntegerRange(0, output_num)

            test_seq = EmissionSequence(sigma, cluster_data.tolist())
            print test_seq
            print m.viterbi(test_seq)
            print gesture_treshold
            diff4 = abs(abs(float(m.viterbi(test_seq)[1])) - abs(float(gesture_treshold)))
            print "Diff 4: "
            print diff4

        elif i == 5:
            f = open('/home/luka/Documents/FER_projekt/Diplomski_rad/trained/plane1/gesture_file.txt', 'r')
            state_num = int(f.readline())
            output_num = int(f.readline())
            gesture_treshold = f.readline()
            f.close()

            #[sigma, A, B, pi] = trening.matrices(state_num, output_num)
            #m = HMMFromMatrices(sigma, DiscreteDistribution(sigma), A, B, pi)

            cluster_data = training.cluster('/home/luka/Documents/FER_projekt/Diplomski_rad/' + filename, output_num) #trajektorija za evaluaciju
            #HMMfactory = HMMOpenFactory(GHMM_FILETYPE_XML)
            m = HMMOpen("/home/luka/Documents/FER_projekt/Diplomski_rad/trained/plane1/m_file.xml")
            print m
            sigma = IntegerRange(0, output_num)

            test_seq = EmissionSequence(sigma, cluster_data.tolist())
            print test_seq
            print m.viterbi(test_seq)
            print gesture_treshold
            diff5 = abs(abs(float(m.viterbi(test_seq)[1])) - abs(float(gesture_treshold)))
            print "Diff 5: "
            print diff5

    diff_all = [diff1, diff2, diff3, diff4, diff5]
    print diff_all
    #for i in range(0, len(diff_all)):
    #        diff_all[i] += cost_list[i]
    treshold = float(7)
    min_diff = min(diff_all)
    if (diff_all[0] > treshold) and (diff_all[1] > treshold) and (diff_all[2] > treshold) and (diff_all[3] > treshold) and (diff_all[4] > treshold):
        print ("NEMA GESTE")

    if diff1 == min_diff:
        print ('Pijenje iz case !!!')
    elif diff2 == min_diff:
        print ('Zaba !!!')
    elif diff3 == min_diff or diff4 == min_diff:
        print ('Proljevanje !!!')
    elif diff5 == min_diff:
        print ("Avion baci mi bombon!")


 #       if float(m.viterbi(test_seq)[1]) > float(gesture_treshold):
 #           tts.say("I recognize the gesture, great job")
 #           print ('Gesta prepoznata !!!')
 #       else:
 #           tts.say("I did not recognize the gesture, please try again")
 #           print('Nema geste !!!')


