import os
import numpy as np
import braingeneers.datasets
import braingeneers as bgr
import matplotlib.pyplot as plt
import plotly.express as px

from scipy  import sparse
from scipy  import signal

from scipy.sparse import csr_matrix

class Neuron:
    def __init__(self, uuid, spike_sorted=True):

        #Set
        self.uuid = uuid
        self.spike_sorted = spike_sorted


        #To be implemented
        self.fs = None
        self.spikes = None
        self.time_series = None



        #Initially load raw data through bgr package

    def load_spikes_test(self,neurons=10,length=1000,fs=1000):
        #Other should load spikes from PRP, for testing we will create np array
        #and set
        x = np.random.rand(neurons,length)
        x[x>.8] = 1
        x[x<=.8] = 0

        
        self.spikes = x
        self.neurons = self.spikes.shape[0]
        self.spike_times = []

        st = []
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if x[i,j] == 1:
                    st.append(j)

            self.spike_times.append(st)
            st = []
        # self.spike_times 
        self.fs = fs
        self.sp_sparse = sparse.coo_matrix(self.spikes).tocsr()


    def load_map(self, map_arr=None):
        #Take in list of coordinates with (x,y,z) as positions of neurons
        #For unknown map, argument is none, creating a grid for visualization purposes
        if map_arr is None:
            size = int(np.ceil(np.sqrt(self.neurons)))
            x = np.arange(0, size)
            y = np.arange(0, size)
            z = np.zeros((size,))
            self.map = np.zeros((3,size))
            print(self.map.shape)
            self.map[0] = np.array(x)
            self.map[1] = y
            self.map[2] = z
                    

        else:
            for i,x,y,z in enumerate(map_arr):
                #Probably a better way to do this, but this works
                self.map = np.zeros((3,len(map_arr)))
                self.map[0,i] = x
                self.map[1,i] = y
                self.map[2,i] = z
                

    def plot_spikes(self):
        #Plots spikes that have been loaded
        print(self.spikes.shape)
        plt.eventplot(self.spike_times)

        plt.title('Spikes through time')
        plt.xlabel('Time')
        plt.ylabel('Spikes')
        plt.show()
        
    def spike_correlation(self):
        #Requires non-sparse encoding of spikes, set as self.spikes in load methods
        #Computes correlation of each neuron combination

        self.corr_arr = np.zeros((self.neurons,self.neurons,self.spikes.shape[1]))

        for i in range(self.neurons):
            for j in range(self.neurons):
                self.corr_arr[i,j] = signal.correlate(self.spikes[i],self.spikes[j],"same")
                


    def plot_correlation(self,lag_time=0):
        #Plots the correlation heatmap created in spike_correlation
        offset = self.spikes.shape[1]//2
        print(offset)
        labels=dict(x="Neuron Index", y="Neuron Index", color="Correlation")

        fig = px.imshow(self.corr_arr[...,500],labels=labels)

        fig.show()




if __name__ == "__main__":

    n = Neuron('test_recording')

    n.load_spikes_test()
   

    n.load_map()

    n.spike_correlation()
    n.plot_correlation(lag_time = 0)
    n.plot_spikes()
    