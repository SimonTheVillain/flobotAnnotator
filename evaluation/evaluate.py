import sys
import os
from os.path import dirname, realpath
from os import listdir
from os.path import isfile, join
import argparse
sys.path.insert(1, dirname(realpath(__file__)))
import cv2
import numpy as np
import matplotlib.pyplot as plt



parser = argparse.ArgumentParser(description='Parsing trough probabilities, groundtruth')
parser.add_argument('--input', type=str, required=False, # true
                    #default='/home/simon/datasets/flobot/carugate/freezer_section_dirt/2018-04-18-10-20-53.bag',
                    help="Folder with calculated probabilities.")
parser.add_argument('--groundtruth', type=str, required=False,
                    #default='/home/simon/datasets/flobot/carugate/freezer_section_dirt/dirt_mask_groundtruth.bag',
                    help="Thre groundtruth mask for dirt.")
parser.add_argument('--mask', type=str, required=False,
                    #default='/home/simon/datasets/flobot/carugate/freezer_section_dirt/dirt_mask_groundtruth.bag',
                    help="The floor mask Only evaluate these pixel.")
parser.add_argument('--prob', type=str, required=False,
                    #default='/home/simon/datasets/flobot/carugate/freezer_section_dirt/dirt_mask_groundtruth.bag',
                    help="The probablility of each pixel being dirt.")
parser.add_argument('--store', type=str, required=False,
                    help='Store the intermediate results')
opt = parser.parse_args()

#TODO! EVERYTHING!!!!!!


n = 200
r = range(0,n)
count_true_pos = np.zeros(n)#[1 for i in r]
count_false_pos = np.zeros(n)#[1 for i in r]
count_true_neg = np.zeros(n)#[0 for i in r]
count_false_neg = np.zeros(n)#[0 for i in r]


onlyfiles = [f for f in listdir(opt.groundtruth) if isfile(join(opt.groundtruth, f))]

running = True
count = 0
for f in onlyfiles:

    image_path = opt.groundtruth + "/" + onlyfiles[count]
    print(count)
    if os.path.isfile(image_path) and onlyfiles[count][0] != '.':
        im = cv2.imread(opt.input + "/" + onlyfiles[count])
        cv2.imshow("im", im)

        gt = cv2.imread(opt.groundtruth + "/" + onlyfiles[count], cv2.IMREAD_UNCHANGED)
        cv2.imshow("gt", gt)


        if opt.mask:

            image_path = opt.mask + "/" + onlyfiles[count]
            mask = cv2.imread(opt.mask + "/" + onlyfiles[count], cv2.IMREAD_UNCHANGED)
            cv2.imshow("mask", mask)
        else:
            mask = np.zeros([480, 640], dtype=np.uint8)*255
            mask[10:470, 10:630] = 254
            mask[np.logical_and(mask == 254,  np.logical_or(np.logical_or(im[:, :, 0] != 0, im[:, :, 1] != 0), im[:, :, 2] != 0))] = 255
            mask[mask == 254] = 0

            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            cv2.imshow("mask", mask)



        prob = cv2.imread(opt.prob + "/" + onlyfiles[count], cv2.IMREAD_UNCHANGED)
        cv2.imshow("prob", prob*128)

        for thr in r:
            #print(thr)
            cv2.imshow("result", (mask & (prob > thr))*255)
            true_pos = np.logical_and(mask, np.logical_and(prob > thr, gt))
            count_true_pos[thr] += np.sum(true_pos)

            false_pos = np.logical_and(mask, np.logical_and(prob > thr, np.logical_not(gt)))#np.logical_and(mask,np.logical_and(prob>thr,np.logical_not(gt)))#(mask & np.logical_and(prob > thr, np.logical_not(gt))
            count_false_pos[thr] += np.sum(false_pos)

            true_neg = np.logical_and(mask, np.logical_and(prob <= thr, np.logical_not(gt)))
            count_true_neg[thr] += np.sum(true_neg)

            false_neg = np.logical_and(mask, np.logical_and(prob <= thr, gt))
            count_false_neg[thr] += np.sum(false_neg)

            true_pos = true_pos.astype(float)
            cv2.imshow("true_pos", true_pos)
            false_pos = false_pos.astype(float)
            cv2.imshow("false_pos", false_pos)
            cv2.waitKey(1)
        #tpr = np.divide(true_pos, true_pos + false_neg)
        #fpr = np.divide(false_pos, false_pos + true_neg)
        #plt.plot(tpr, fpr)
        #plt.ylabel('some numbers')
        #plt.show()
        count = count + 1
    else:
        running = False

#print(count_true_pos)
#print(count_false_pos)
#print(count_false_neg)
#print(count_true_neg)
tpr = np.divide(count_true_pos, count_true_pos + count_false_neg)
fpr = np.divide(count_false_pos, count_false_pos + count_true_neg)
#print(tpr)
#print(fpr)
if opt.store:
    np.save(opt.store +'_count_true_pos.npy', count_true_pos)
    np.save(opt.store +'_count_false_pos.npy', count_false_pos)
    np.save(opt.store +'_count_false_neg.npy', count_false_neg)
    np.save(opt.store +'_count_true_neg.npy', count_true_neg)
plt.plot(fpr, tpr)#, [0, 1], [0, 1])
plt.ylabel('True positive rate (tpr)')
plt.xlabel('False positive rate (fpr)')
plt.show()

#fpr.save('lyon_tpr')
#tpr.save('lyon_fpr')
