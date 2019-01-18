import collections
import bisect
import numpy as np
#import tf
import pyquaternion


class tf_manager:

    def __init__(self):
        self.tfMessages = []
        self.messages = collections.OrderedDict();

    def add(self,transformMsg,t):
        transform = transformMsg
        key = transform.header.stamp.secs * 1000000000 + transform.header.stamp.nsecs
        #key=t
        #print(transform)
        self.messages[key] = transform
        #print("everything is awful")
        #tfMessages.extend(tfMessage)


    def get(self, stamp):
        key = stamp.secs * 1000000000 + stamp.nsecs
        #key=t
        ind = bisect.bisect_left(self.messages.keys(), key)
        #key = self.messages.keys()[ind]
        #return self.messages[key]
        if ind == 0 or ind == len(self.messages)-1:
            return self.messages.values()[ind]
        else:
            return self.messages.values()[ind]

    def size(self):
        return len(self.messages)

    def getMat(self,stamp):
        msg = self.get(stamp)

        quaternion = pyquaternion.Quaternion(msg.transform.rotation.w,
                                                       msg.transform.rotation.x,
                                                       msg.transform.rotation.y,
                                                       msg.transform.rotation.z)

        R = np.eye(4)
        R[0:3,0:3] = quaternion.rotation_matrix
        T = np.eye(4)
        T[0:3,3] = np.array([msg.transform.translation.x,
                             msg.transform.translation.y,
                             msg.transform.translation.z])


        #return R.dot(T)
        return T.dot(R) #first rotate, then translate
        #now do some numpy stuff.
