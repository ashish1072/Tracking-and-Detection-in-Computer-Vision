import random
import math
import numpy as np


class BatchGen:

    def __init__(self, train_set, test_set, db):
        self.train_set = train_set
        self.test_set = test_set
        self.db = db

    def get_anchor(self, train_set):
        rand_obj = random.choice(list(train_set.keys()))
        rand_obj_pose = random.choice(train_set[rand_obj])  # img path and its pose
        return rand_obj, rand_obj_pose

    def get_all_anchors(self, train_set):
        anchors = list()
        objects = list(train_set.keys())
        for obj in objects:
            for img in self.train_set[obj]:
                anchors.append([obj, img])
        return anchors

    def get_puller(self, obj, anchor_g, db):
        """
        Puller - the most similar (quaternion-wise) to anchor_g sample of the same
        object taken from the db_q set
        """

        min = math.inf
        min_index = 0
        for i in range(len(db[obj])):
            dist = self.quat_angular_metric(anchor_g, db[obj][i][1])
            if dist < min:
                min = dist
                min_index = i
        return db[obj][min_index]

    def quat_angular_metric(self, anchor_q, puller_q):
        """anchor_q, puller_q - quarterions"""
        dot_prod_abs = abs(np.dot(anchor_q, puller_q))
        if dot_prod_abs > 1:
            dot_prod_abs = 1
        return 2 * math.acos(dot_prod_abs)

    def get_pusher(self, db):
        rand_obj = random.choice(list(db.keys()))
        rand_obj_pose = random.choice(db[rand_obj])
        return rand_obj_pose

    def get_triplet(self):
        obj_name, anchor = self.get_anchor(self.train_set)
        puller = self.get_puller(obj_name, anchor[1], self.db)
        pusher = self.get_pusher(self.db)
        return anchor, puller, pusher

    def get_all_triplets(self):
        triplets = list()
        anchors = self.get_all_anchors(self.train_set)
        for a in anchors:
            puller = self.get_puller(a[0], a[1][1], self.db)  # pass name and only pose
            pusher = self.get_pusher(self.db)
            triplets.append([a[1], puller, pusher])
        return triplets
