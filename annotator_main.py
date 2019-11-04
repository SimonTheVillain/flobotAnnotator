#!/usr/bin/env python
import sys
import os
from os.path import dirname, realpath
sys.path.insert(1, dirname(realpath(__file__)))
from tf_manager import tf_manager
from annotation_manager import Annotation, AnnotationManager
import argparse

import collections
import rosbag
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError



parser = argparse.ArgumentParser(description='GroundplaneAnnotation')
parser.add_argument('--input', type=str, required=True,
                    #default='/home/simon/datasets/flobot/carugate/freezer_section_dirt/2018-04-18-10-20-53.bag',
                    help="Input rosbag file.")
parser.add_argument('--output', type=str, required=False,
                    #default='/home/simon/datasets/flobot/carugate/freezer_section_dirt/dirt_mask_groundtruth.bag',
                    help="Output rosbag file.")
parser.add_argument('--use_depth', type=bool, default=True,
                    help='Use the depth values to fit a plane. If not the x,y plane is assumed.')
parser.add_argument('--annotation_file', type=str, required=False,
                    default='annotations/annotations_carugate.yaml',
                    help='File to load annotations. Upon pressing p it stores all changes to this file.')

opt = parser.parse_args()

#print(opt)

bag_in = rosbag.Bag(opt.input, 'r')

render_results_to_bag = False
render_results_to_folder = False
if hasattr(opt, 'output'):
    if opt.output:
        if opt.output.endswith('.bag'):
            render_results_to_bag = True
        else:
            render_results_to_folder = True



#bag_in = rosbag.Bag('/home/simon/datasets/flobot/carugate/freezer_section_dirt/2018-04-18-09-55-25_dirt_crumbs_milk.bag','r')
#this is a valid rosbag at least!!!
#bag_in = rosbag.Bag('/home/simon/datasets/flobot/lyon_june/2018-06-13-15-46-35_cola_spots.bag','r')
#bag_in = rosbag.Bag('/media/simon/TOSHIBA EXT/lyon_june/2018-06-13-15-46-35_cola_spots.bag','r')
#result_path = '/home/simon/datasets/flobot/carugate/freezer_section_dirt/dirt_mask_groundtruth.bag'
#annotation_file = 'annotations_carugate.yaml'
#render_result_to_file = False
#load_annotation_from_file = True







#todo... we need to go from world -> vehicle -> velodyne -> camera_link -> camera_rgb_frame -> camera_rgb_optical_frame
tf_world_vehicle = tf_manager()
tf_vehicle_velodyne = tf_manager()
tf_velodyne_camera_link = tf_manager()
tf_camera_link_camera_rgb_frame = tf_manager()
tf_camera_rgb_frame_crof = tf_manager()

def getTransformation(stamp):
    mat1 = tf_world_vehicle.getMat(stamp)
    mat2 = tf_vehicle_velodyne.getMat(stamp)
    mat3 = tf_velodyne_camera_link.getMat(stamp)
    mat4 = tf_camera_link_camera_rgb_frame.getMat(stamp)
    mat5 = tf_camera_rgb_frame_crof.getMat(stamp)
    return mat1.dot(mat2).dot(mat3).dot(mat4).dot(mat5)

def generate_new_rosbag(bag_out_path , bag_in, annotator):
    bridge = CvBridge()
    bag_out = rosbag.Bag(bag_out_path, 'w')
    for topic, msg, t in bag_in.read_messages():
        bag_out.write(topic,msg,t)
        if(topic == "/camera/depth_registered/points"):
            mat = np.zeros((480, 640), np.uint8)
            stamp = msg.header.stamp
            #render existing annotations
            pose = getTransformation(msg.header.stamp)
            annotator.set_stamp_pose(stamp, pose)
            color_mat = np.zeros((480, 640, 3), np.uint8)
            color_mat, annotation = annotator.render_annotations(color_mat)
            #cv2.imshow("annotation",annotation * 10000)
            #todo: add the markings here!!!!!!!!
            #mat[100:200,100:200] = 255
            mat[annotation != 0 ] = 255

            cv2.imshow("mask", mat)
            cv2.waitKey(1)

            img = bridge.cv2_to_imgmsg(mat, 'passthrough')
            img.header.stamp = stamp
            bag_out.write('/camera/dirt_mask', img, t)#stamp)
            #https://answers.ros.org/question/11537/creating-a-bag-file-out-of-a-image-sequence/
            pass
    pass
    bag_out.close()


def render_dataset(path, bag_in, annotator):
    bridge = CvBridge()
    if not os.path.exists(path):
        os.mkdir(path)
    path_img = path + '/img'
    path_label = path + '/truth'
    if path.endswith('/'):
        path_img = path + 'img'
        path_label = path + 'truth'
    if not os.path.exists(path_img):
        os.mkdir(path_img)

    if not os.path.exists(path_label):
        os.mkdir(path_label)
    count = 0
    for topic, msg, t in bag_in.read_messages():
        if topic == "/camera/depth_registered/points":
            mat = np.zeros((480, 640), np.uint8)
            stamp = msg.header.stamp
            # render existing annotations
            pose = getTransformation(msg.header.stamp)
            annotator.set_stamp_pose(stamp, pose)
            color_mat = np.zeros((480, 640, 3), np.uint8)
            color_mat, annotation = annotator.render_annotations(color_mat)
            mat[annotation != 0] = 255

            image = np.fromstring(msg.data, dtype=np.uint8)
            image = image.reshape((msg.height, msg.width, 8 * 4))
            image = image[:, :, 16:(16 + 3)].copy()
            cv2.imshow("color_image", image)
            cv2.imshow("mask", mat)
            cv2.imwrite(path_img + '/' + str(count) + '.png', image)
            cv2.imwrite(path_label + '/' + str(count) + '.png', mat)
            cv2.waitKey(1)
            # https://answers.ros.org/question/11537/creating-a-bag-file-out-of-a-image-sequence/
            count = count + 1
            pass



#todo: integrate all of this into a class
print("building up a tf register")
for topic, msg, t in bag_in.read_messages(topics=['/tf']):
    for transform in msg.transforms:
        if transform.header.frame_id == "world" and transform.child_frame_id == "vehicle": #and child_frame_id == "vehicle"
            tf_world_vehicle.add(transform, t)
        elif transform.header.frame_id == 'vehicle' and transform.child_frame_id == 'velodyne':
            tf_vehicle_velodyne.add(transform, t)
        elif transform.header.frame_id == 'velodyne' and transform.child_frame_id == 'camera_link':
            tf_velodyne_camera_link.add(transform, t)
        elif transform.header.frame_id == '/camera_link' and transform.child_frame_id == '/camera_rgb_frame':
            tf_camera_link_camera_rgb_frame.add(transform, t)
        elif transform.header.frame_id == '/camera_rgb_frame' and transform.child_frame_id == '/camera_rgb_optical_frame':
            tf_camera_rgb_frame_crof.add(transform, t)
        elif transform.header.frame_id == "world" and transform.child_frame_id == "map":
            pass
            #print(transform)

print("done")

#TODO: delete these
#print(tf_world_vehicle.size())
#print(tf_vehicle_velodyne.size())
#print(tf_velodyne_camera_link.size())
#print(tf_camera_link_camera_rgb_frame.size())
#print(tf_camera_rgb_frame_crof.size())


def project():
    pass

def backProject(p):
    proj = np.array([
        p[0, :] / p[2, :] * 540.0 + 320.0,
        p[1, :] / p[2, :] * 540.0 + 240.0
    ])
    return proj



#ptest = np.array([[1,1,1,1],[2,2,2,2],[3,3,3,3]],np.float32)
#print(ptest)
#print(backProject(ptest))
#backProject(np.array([1,1,1,1],np.float32))

mouseClicked = False
clickedPos = []

firstFrame = True
point = []


running = True
#todo maybe a dequeue is better

messageBuffer = collections.deque()


bufferSize = 100


messageGenerator = bag_in.read_messages(topics=['/camera/depth_registered/points'])

topic, msg, t = messageGenerator.next()
messageBuffer.append((msg,t))





cv2.namedWindow("color")

annotator = AnnotationManager("color")

if os.path.isfile(opt.annotation_file):
    annotator.load(opt.annotation_file)

annotations = collections.OrderedDict()

if render_results_to_bag:
    generate_new_rosbag(opt.output, bag_in, annotator)
    exit()

if render_results_to_folder:
    render_dataset(opt.output, bag_in, annotator)
    exit()


running = True
freeRun = True
currentIndex = 0
while running:
    #fetching next frame from list
    msg, t = messageBuffer[currentIndex]

    pose = getTransformation(msg.header.stamp)
    pose_ = np.linalg.inv(pose)


    img = np.fromstring(msg.data, dtype=np.float32)
    imgBytes = np.fromstring(msg.data, dtype=np.uint8)
    img = img.reshape((msg.height, msg.width, 8))
    imgBytes = imgBytes.reshape((msg.height, msg.width, 8 * 4))
    point_image = img[:, :, 0:3]
    color_image = imgBytes[:, :, 16:(16 + 3)].copy()

    annotator.set_images(color_image,point_image,pose,t)

    color_annotated, annotation_mat = annotator.render_annotations(color_image)
    cv2.imshow("color",color_annotated)


    if freeRun:
        nextFrame = True
        key = cv2.waitKey(1)
    else:
        nextFrame = False
        key = cv2.waitKey()

    #print(key)
    if key == 27: # esc
        annotator.esc_pressed()
    if key == 32:  # space
        freeRun = not freeRun
    elif key == 97 or key == 81: #a go back
        currentIndex = currentIndex - 1
        if currentIndex < 0:
            currentIndex = 0
    elif key == 100 or key == 83: #d go forward
        nextFrame = True
    elif key == 13:#return
        annotator.return_pressed()
    elif key == 8:#backspace
        annotator.backspace_pressed()
    elif key == 255: #entf
        #todo: delete the currently selected thingy
        annotator.delete_current_selection()
        pass
    elif key == 112: #p
        annotator.store(opt.annotation_file)
        pass
    elif key == 101: #e
        #todo: end the current annotation from this timeframe on
        annotator.set_end_time_of_selection()
        pass
    elif key == 115: #s
        #todo: start the current annotation from this timeframe on
        annotator.set_start_time_of_selection()
        pass


    if nextFrame:

        currentIndex = currentIndex + 1

        if currentIndex >= len(messageBuffer):# we need to add a new frame to the buffer
            searching = True
            try:
                topic, msg, t = messageGenerator.next()
                messageBuffer.append((msg, t))
                if len(messageBuffer) > bufferSize:
                    messageBuffer.popleft()
                    currentIndex = currentIndex - 1
            except StopIteration:
                print("end of rosbag")
                running = False
                pass





