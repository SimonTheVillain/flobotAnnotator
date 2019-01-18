import numpy as np
import numpy.matlib
import collections
import bisect

import cv2
from tf_manager import tf_manager
from enum import Enum
import math


#import sys
#import ruamel.yaml
#all of this for serialization
import yaml
from collections import namedtuple



def key_from_stamp(stamp):
    return stamp.secs * 1000000000 + stamp.nsecs


AnnotationSerializeable = namedtuple("AnnotationSerializeable", "stamp_from stamp_to polygons")


AnnotationManagerSerializeable = namedtuple("AnnotationManagerSerializeable", "annotation_list")

class Annotation:

## what should be possible with this class:
## creation of a new polygon interpolating the polygon
## changing the polygon over the frames
## no adding of new points to the polygon

## setting start time for polygon at frame by button 's'
## setting stop time for polygon at frame by button 'e'?
## deleting the polygon 'del/entf'?

    def __init__(self,id):

        #self.id = id#todo: if not necessary get rid of this id (propably not necessary)
        self.polygons = collections.OrderedDict()
        #key = key_from_stamp(stamp)
        #self.polygons[key] = polygon
        #let the polygon go from beginning to end!!
        self.stamp_from =  -1
        self.stamp_to = -1

    def get_polygon(self, stamp):
        key = stamp.secs * 1000000000 + stamp.nsecs
        keys = self.polygons.keys()
        keys.sort()
        ind = bisect.bisect_left(keys, key)

        #bisect gives an insertion point which sometimes is higher than the amount of stored elements
        if ind >= keys.__len__():
            return self.polygons[keys[ind-1]].copy()

        if ind == 0:
            return self.polygons[keys[0]].copy()

        #print("between two")
        polygon1 = self.polygons[keys[ind-1]].copy()
        polygon2 = self.polygons[keys[ind]].copy()

        dkey_full = keys[ind] - keys[ind-1]
        dkey_local = key - keys[ind-1]
        l = float(dkey_local)/float(dkey_full)
        #todo interpolate

        return polygon2 * l + polygon1 * (1.0 -l)

    def draw(self,stamp,color_image,index_image):

        pass

    def draw_selected(self,stamp,color_image,index,new_pos):
        vertices = self.get_polygon(stamp)
        pass

    def set_polygon(self, polygon, stamp):
        key = key_from_stamp(stamp)
        self.polygons[key] = polygon
        pass

    def set_polygon_with_key(self, polygon, key):
        self.polygons[key] = polygon

        pass
    def get_serializeable(self):
        s_polygons = collections.OrderedDict()
        for key in self.polygons.keys():
            s_polygons[key] = self.polygons[key].tolist()
        serializeable = AnnotationSerializeable(stamp_from=self.stamp_from,stamp_to=self.stamp_to,polygons=s_polygons)
        return serializeable

class Mode(Enum):
    Edit_None = 1
    Edit_New = 2
    Edit_Existing = 3
    Edit_Vertex = 4
    Edit_Move = 5

class AnnotationManager:



    def __init__(self,window_name):
        #todo: actually this should be a vector with annotations each of the annotations needs to handle its own timestamps
        #todo: check if the annotation manager really is supposed to handle the annotations. (maybe only use it to handle rendering)
        #todo: really a ordered dict?
        #self.annotations = collections.OrderedDict()
        self.vertex_radius = 2.0
        self.stamp = []
        self.pose = []
        self.pose_ = []
        self.color_image = np.zeros((480, 640, 3), np.uint8)
        self.point_image = np.zeros((480, 640, 3), np.float32)
        res = (480, 640)
        self.label_map = np.zeros(res, np.int32)

        self.pulling_vertex = False
        self.current_vertices = []
        self.annotation_list = list()
        self.finished_polyline = False

        cv2.setMouseCallback(window_name, self.click)

        self.mode = Mode.Edit_None
        self.lbutton_pressed = False
        self.current_vertex_ind = -1
        self.current_annotation_ind = -1

        self.cx = 320
        self.cy = 240
        self.fx = 540
        self.fy = 540

    def store(self,yaml_path):
        #todo: setup a "serializable class" and dump it with yaml
        s_annotation_list = list()

        for annotation in self.annotation_list:
            s_annotation_list.append(annotation.get_serializeable())

        s_annotation_manager = AnnotationManagerSerializeable(s_annotation_list)
        with open(yaml_path, 'w') as f:
            yaml.dump(s_annotation_manager, f)
        pass

    def load(self,yaml_path):
        #todo: load it with yaml and fill the necessary containers
        with open(yaml_path) as f:
            loaded = yaml.load(f)
            print(loaded)
            self.annotation_list = list()
            for annotation_loaded in loaded.annotation_list:
                polygons = collections.OrderedDict()
                annotation = Annotation(-1)
                for key in annotation_loaded.polygons.keys():
                    annotation.set_polygon_with_key(np.array(annotation_loaded.polygons[key]),key)
                    pass

                self.annotation_list.append(annotation)

                pass
        pass

    def pix_to_global(self, x, y, pose):
        #first step is getting vector v
        #camera pos
        v_local = np.array([x-self.cx,y-self.cy,self.fx])
        v = np.dot(pose[0:3,0:3],v_local)
        v = v / np.linalg.norm(v)
        pos = pose[0:3,3]

        p = pos - v* (pos[2]/v[2])
        return p

    def click(self,event, x, y, flags, param):
        color_with_markings = self.color_image.copy()
        #index_image = np.zeros((480,640),np.int32)
        unused, index_image = self.render_annotations(color_with_markings,flags & cv2.EVENT_FLAG_SHIFTKEY)
        # https://www.pyimagesearch.com/2015/03/09/capturing-mouse-click-events-with-python-and-opencv/
        if event == cv2.EVENT_LBUTTONDOWN:
            self.lbutton_pressed = True
            if self.mode == Mode.Edit_None:
                index = index_image[y,x]
                if index != 0:
                    self.mode = Mode.Edit_Existing
                    print("clicked on something existing")
                    index = index-1
                    self.current_annotation_ind = index
                    self.current_vertices = self.annotation_list[index].get_polygon(self.stamp)#TODO: need to update stamp
                else:
                    self.mode = Mode.Edit_New
                    print("clicked on nothing")

            if self.mode == Mode.Edit_New:
                print("adding a point to new mask")
                pos = np.append(self.point_image[y, x], [1])
                #todo: the plane projection here
                pos = self.pix_to_global(x, y, self.pose)
                pos = np.append(pos, [1])
                pos = np.reshape(pos, [4, 1])
                try:
                    self.current_vertices = np.append(self.current_vertices, pos, axis=1)
                except IndexError:
                    self.current_vertices = pos
                pass
            if self.mode == Mode.Edit_Existing:
                print("todo: check if we clicked a vertex to modify it")

                #print((x, y))
                #print(self.current_vertices)
                verts = np.dot(self.pose_, self.current_vertices)
                screen_pts = self.local_to_pix( verts)
                #print(screen_pts)
                clicked_vertex = False
                if not flags & cv2.EVENT_FLAG_SHIFTKEY:
                    for index in range(0,screen_pts.shape[1]):
                        screen_pt = screen_pts[:,index]
                        dx = x - screen_pt[0]
                        dy = y - screen_pt[1]
                        dist = math.sqrt(dx*dx + dy*dy)
                        if dist<self.vertex_radius:
                            self.mode = Mode.Edit_Vertex
                            self.current_vertex_ind = index
                            clicked_vertex = True
                            print("editing vertex position")
                if not clicked_vertex:
                    index = index_image[y,x]
                    if index-1 == self.current_annotation_ind:
                        self.mode = Mode.Edit_Move
                        self.initial_move_point = self.pix_to_global(x, y, self.pose)
                        self.initial_move_vertices = self.current_vertices
                        print("moving whole annotation")

                pass



        elif event == cv2.EVENT_LBUTTONUP:
            self.lbutton_pressed = False
            if self.mode == Mode.Edit_Vertex:
                self.mode = Mode.Edit_Existing
            if self.mode == Mode.Edit_Move:
                self.mode = Mode.Edit_Existing
                #editing of vertex has ended back to editing the whole label
            # only useful when pulling a vertex
            pass
        else:
            if self.lbutton_pressed:
                if self.mode == Mode.Edit_Vertex:
                    #todo: update position of vertex
                    pos = np.append(self.point_image[y, x], [1])
                    pos = np.dot(self.pose,pos)

                    pos = self.pix_to_global(x, y, self.pose)
                    pos = np.append(pos, [1])
                    pos = np.reshape(pos, [4, 1])
                    #print(self.current_vertices)
                    #print("pos ")
                    #print(pos)
                    #print(self.current_vertices)
                    self.current_vertices[:,self.current_vertex_ind] = pos.reshape([4 ])
                    pass

                if self.mode == Mode.Edit_Move:
                    #todo: update of all vertex positions
                    point = self.pix_to_global(x, y, self.pose)
                    delta = point - self.initial_move_point
                    delta = np.append(delta, [0])#
                    delta = np.matlib.repmat(delta.reshape([4,1]),1,self.initial_move_vertices.shape[1])
                    self.current_vertices = self.initial_move_vertices + delta
                    pass

            # print("just movin the cursor")
            pass


        color_with_markings,index_image = self.render_annotations(color_with_markings,flags & cv2.EVENT_FLAG_SHIFTKEY)
        cv2.imshow("color", color_with_markings)
        cv2.imshow("annotations", index_image*10000)


    def set_images(self, image,point_image,pose,stamp):
        self.color_image = image
        self.point_image = point_image
        self.pose = pose
        self.pose_ = np.linalg.inv(self.pose)
        self.stamp = stamp
        pass


    def set_stamp_pose(self,stamp,pose):
        self.pose = pose
        self.pose_ = np.linalg.inv(self.pose)
        self.stamp = stamp#key_from_stamp(stamp)

    def render_annotations(self,color_image,suppress_circles = False):
        annotation_image = np.zeros((480,640),np.int32)
        color_with_markings = color_image.copy()
        self.render_existing_annotations(color_with_markings,annotation_image)
        try:
            if self.current_vertices.shape[1] > 0:
                verts = np.dot(self.pose_, self.current_vertices)

                #todo: implement test for no vertex to be negative
                #print(verts)
                if not np.any(verts[2,:] < 0):
                    screen_pts = self.local_to_pix(verts)
                    cv2.fillPoly(color_with_markings, [screen_pts.transpose().astype(np.int)], (0, 255, 255))
                    cv2.polylines(color_with_markings, [screen_pts.transpose().astype(np.int)], 1, (0, 0, 255), thickness=1)
                    cv2.fillPoly(annotation_image, [screen_pts.transpose().astype(np.int)], self.current_annotation_ind + 1)#todo! find out why this is not shown

                if self.mode == Mode.Edit_Existing and not suppress_circles:
                    for i in range(0,verts.shape[1]):
                        cv2.circle(color_with_markings, (screen_pts[0,i].astype(np.int),screen_pts[1,i].astype(np.int)), 3, (0,255,255))

        except AttributeError:
            pass
        return color_with_markings, annotation_image

    def render_existing_annotations(self,color_image,index_image):
        i = 0
        for annotation in self.annotation_list:
            i = i + 1
            verts = np.dot(self.pose_, annotation.get_polygon(self.stamp))
            if not np.any(verts[2, :] < 0):
                if i != self.current_annotation_ind + 1 or self.mode == Mode.Edit_None or self.mode == Mode.Edit_New:
                    screen_pts = self.local_to_pix(verts)
                    cv2.fillPoly(color_image, [screen_pts.transpose().astype(np.int)], (0, 255, 255))
                    cv2.fillPoly(index_image, [screen_pts.transpose().astype(np.int)], i)



    def render_selected_annotation(self,color_image):
        #todo: we have to handle selected existing annotations and novel annotations differently
        pass
    #def get_annotations(self, stamp):
    #    #todo: find the annotation with the lowest key
    #    #
    #    pass

    def return_pressed(self):
        try:
            if self.current_vertices.shape[1] > 0:
                #todo: store
                if self.mode == Mode.Edit_New:
                    id=1
                    annotation = Annotation(id)#,self.current_vertices,self.stamp)
                    annotation.set_polygon(self.current_vertices,self.stamp)

                    self.annotation_list.append(annotation)
                    self.current_vertices = []
                    self.mode = Mode.Edit_None
                elif self.mode == Mode.Edit_Vertex or self.mode == Mode.Edit_Existing:
                    self.annotation_list[self.current_annotation_ind].set_polygon(self.current_vertices,self.stamp)
                    self.current_vertices = []
                    self.mode = Mode.Edit_None
                    pass
        except AttributeError:
            pass


    def esc_pressed(self):
        self.mode = Mode.Edit_None
        self.current_vertices = []
        pass

    def backspace_pressed(self):
        #remove one element from the currently edited annotation
        #print(self.current_vertices)
        if self.mode == Mode.Edit_New:
            self.current_vertices = self.current_vertices[:,:-1]
        #print(self.current_vertices)



    def local_to_pix(self,p):
        proj = np.array([
            p[0, :] / p[2, :] * self.fx + self.cx,
            p[1, :] / p[2, :] * self.fy + self.cy
        ])
        return proj

    #backProject = staticmethod(backProject)


    def delete_current_selection(self):
        if self.mode == Mode.Edit_New:
            self.current_vertices = []
            self.mode = Mode.Edit_None
        if self.mode == Mode.Edit_Existing:
            self.annotation_list.remove(self.annotation_list[self.current_annotation_ind])
            self.mode = Mode.Edit_None
            self.current_vertices = []

        pass

    def set_end_time_of_selection(self):
        pass
    def set_start_time_of_selection(self):
        pass