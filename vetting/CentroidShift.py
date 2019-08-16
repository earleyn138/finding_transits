import matplotlib.pyplot as plt
import scipy.ndimage.measurements as sp
import numpy as np
import pickle


__all__ = ['CentroidShift']

class CentroidShift(object):
    '''
    '''

    def __init__(self, time, x_cent, y_cent, path, run):

        self.time = time
        self.x_cent = x_cent
        self.y_cent = y_cent
        self.path = path
        self.run = run

        self.load_data()
        self.get_trns_dur()
        self.find_points()


    def load_data(self):
        model = pickle.load(open(self.path, "rb"))
        self.t0 = model['Planet Solns'][self.run]['t0']
        self.per = model['Planet Solns'][self.run]['period']
        self.dur = model['BLS Duration'][self.run]


    def get_trns_dur(self):
        trns = (self.per-((min(self.time)-self.t0)%self.per))+min(self.time)

        self.trns_points = [trns]
        for n in range(10):
            trns = trns + self.per
            if trns < max(self.time):
                self.trns_points.append(trns)

        self.start_times = []
        self.end_times = []
        for t in self.trns_points:
            self.start_times.append(t - (self.dur/2))
            self.end_times.append(t + (self.dur/2))


    def find_points(self):
        '''
        '''
        self.time_trns = []
        self.x_cent_trns = []
        self.y_cent_trns = []

        for i, t in enumerate(self.start_times):
            pos = np.where((self.time > t) & (self.time < (self.end_times[i])))
            for p in pos:
                for i in p:
                    self.time_trns.append(self.time[i])
                    self.x_cent_trns.append(self.x_cent[i])
                    self.y_cent_trns.append(self.y_cent[i])
