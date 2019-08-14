import matplotlib.pyplot as plt
import scipy.ndimage.measurements as sp
import numpy as np
import pickle


__all__ = ['CentroidShift']

class CentroidShift(object):
    '''Runs through the box-least-squares method to find transits and
    performs GP modeling'''

    def __init__(self, time, xcent, ycent, pickle_path, polyorder):

        self.time = time
        self.xcent = xcent
        self.ycent = ycent
        self.path = pickle_path
        self.order = polyorder

        self.load_data()
        self.get_trns_dur()
        self.find_points()
        self.fit()
        self.plot()

    #
    #
    # def compute_com(self):
    #     '''
    #     '''
    #     xcent = []
    #     ycent = []
    #     for f in self.tpf:
    #         com = sp.center_of_mass(f*self.aperture)
    #         xcom = com[0]
    #         ycom = com[1]
    #         xcent.append(xcom)
    #         ycent.append(ycom)
    #
    #     xcent = np.array(xcent)
    #     ycent = np.array(ycent)
    #
    #     self.xcent = xcent[self.slice]
    #     self.ycent = ycent[self.slice]


    def load_data(self):
        model = pickle.load(open(self.path, "rb"))
        self.t0 = model['Planet Solns'][0]['t0']
        self.per = model['Planet Solns'][0]['period']
        self.dur = model['BLS Duration'][0]


    def get_trns_dur(self):
        trns = (self.per-((min(self.time)-self.t0)%self.per))+min(self.time)

        self.trns_points = [trns]
        for n in range(10):
            trns = trns + self.per
            if trns < max(self.time):
                self.trns_points.append(trns)

        self.start_times = []
        self.end_times = []
        for t in trns_points:
            self.start_times.append(t - (self.dur/2))
            self.end_times.append(t + (self.dur/2))


    def find_points(self):
        '''
        '''
        self.time_trns = []
        self.xcent_trns = []
        self.ycent_trns = []

        for i, t in enumerate(start_times):
            pos = np.where((self.time > t) & (self.time < (end_times[i])))
            for p in pos:
                for i in p:
                    self.time_trns.append(self.time[i])
                    self.xcent_trns.append(self.xcent[i])
                    self.ycent_trns.append(self.ycent[i])

    def fit(self):
        '''
        '''
        coeffx = np.polyfit(self.time, self.xcent, self.order)
        coeffy = np.polyfit(self.time, self.ycent, self.order)

        self.fitx = np.polyval(coeffx, self.time)
        self.fity = np.polyval(coeffy, self.time)


    def plot(self):
        '''
        '''
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

        ax0 = axes[0][0]
        ax0.scatter(self.time, self.xcent)
        ax0.scatter(self.time_trns, self.xcent_trns)
        ax0.plot(self.time, self.fitx, 'r', linewidth=3)
        ax0.set_ylabel('x centroid')

        ax1 = axes[0][1]
        ax1.scatter(self.time, self.ycent)
        ax1.scatter(self.time_trns, self.ycent_trns)
        ax1.plot(self.time, self.fity, 'r', linewidth=3)
        ax1.set_ylabel('y centroid')

        ax2 = axes[1][0]
        ax2.plot(self.time, self.xcent-self.fitx)
        ax2.set_ylabel('x residuals')
        ax2.set_ylim(-0.01, 0.01)
        #ax2.set_xlim(1440, 1442)

        ax3 = axes[1][1]
        ax3.plot(self.time, self.ycent-self.fity)
        ax3.set_ylabel('y residuals')
        ax3.set_ylim(-0.01, 0.01)
        #ax3.set_xlim(1440.5, 1441.5)

        fig.tight_layout()
        plt.show()
        plt.close()
