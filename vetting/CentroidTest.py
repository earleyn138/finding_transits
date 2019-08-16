import numpy as np
import scipy.ndimage.measurements as sp
import matplotlib.pyplot as plt
import pickle
import vetting

__all__ = ['CentroidTest']

class CentroidTest(object):
    """docstring for ."""

    def __init__(self, time, x_cent, y_cent, path, run, polyorder):

        self.time = time
        self.x_cent = x_cent
        self.y_cent = y_cent
        self.path = path
        self.run = run
        self.polyorder = polyorder

        self.find_trns()
        self.fit()
        self.plt_shift()


    def find_trns(self):
        '''docstring'''

        model = pickle.load(open(self.path, "rb"))

        #for time, x_cent, y_cent in zip(self.time_list, self.x_cent_list, self.y_cent_list):
        self.time_trns_list = []
        self.x_cent_trns_list = []
        self.y_cent_trns_list = []

        for run in range(len(model['Run'])-1):
            cents = vetting.CentroidShift(self.time, self.x_cent, self.y_cent, self.path, self.run)
            self.time_trns_list.append(cents.time_trns)
            self.x_cent_trns_list.append(cents.x_cent_trns)
            self.y_cent_trns_list.append(cents.y_cent_trns)


    def fit(self):
          '''
          '''
          self.coeffx = np.polyfit(self.time, self.x_cent, self.polyorder)
          self.coeffy = np.polyfit(self.time, self.y_cent, self.polyorder)
          self.fitx = np.polyval(self.coeffx, self.time)
          self.fity = np.polyval(self.coeffy, self.time)

          self.residx = self.x_cent - self.fitx
          self.residy = self.y_cent - self.fity


    def plt_shift(self):

        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)

        ax0 = axes[0][0]
        ax0.scatter(self.time, self.x_cent)
        for i, t in enumerate(self.time_trns_list):
            ax0.scatter(t, self.x_cent_trns_list[i])
        ax0.plot(self.time, self.fitx, 'r', linewidth=3)
        ax0.set_ylabel('x centroid')

        ax1 = axes[0][1]
        ax1.scatter(self.time, self.y_cent)
        for i, t in enumerate(self.time_trns_list):
            ax1.scatter(t, self.y_cent_trns_list[i])
        ax1.plot(self.time, self.fity, 'r', linewidth=3)
        ax1.set_ylabel('y centroid')

        ax2 = axes[1][0]
        ax2.plot(self.time, self.residx)
        ax2.set_ylabel('x residuals')
        ax2.set_ylim(-0.01, 0.01)
        #ax2.set_xlim(1440, 1442)

        ax3 = axes[1][1]
        ax3.plot(self.time, self.residy)
        ax3.set_ylabel('y residuals')
        ax3.set_ylim(-0.01, 0.01)
        #ax3.set_xlim(1440.5, 1441.5)

        fig.tight_layout()
        plt.show()
        plt.close()
