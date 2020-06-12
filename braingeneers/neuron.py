import os
import numpy as np
import braingeneers.datasets
import braingeneers as bgr
import matplotlib.pyplot as plt
import plotly.express as px
import time

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
        x[x>.95] = 1
        x[x<=.95] = 0

        
        self.spikes = x
        self.neurons = self.spikes.shape[0]
        self.spike_times = []

        st = []

        #TODO optimize later with this logic
        # n, t = np.nonzero(self.spikes)

        # for num, ind in n:
        #     if len(self.spike_times < 
        #     self.spike_times.append

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
        
        
    def spike_correlation(self,steps):
        #Requires non-sparse encoding of spikes, set as self.spikes in load methods
        #Computes correlation of each neuron combination

        self.corr = np.zeros((self.spikes.shape[0],self.spikes.shape[0],steps+1))
        self.corr_steps = steps

        for i in range(self.neurons):
            for j in range(self.neurons):
                self.corr[i,j] = signal.correlate(self.spikes[j],self.spikes[i,:-steps],"valid")

    def window_correlation(self,spikes,steps):
        #Requires non-sparse encoding of spikes
        #Spike shape (n_neurons,n_timesteps)
        #Requires steps, useful to use maximum hebbian learning time ~30-50ms
        #i,j means i occurs before j
        #Computes correlation of each neuron combination

        self.corr_w = np.zeros((spikes.shape[0],spikes.shape[0],steps+1))
        self.corr_w_steps = steps

        for i in range(spikes.shape[0]):
            for j in range(self.neurons):
                self.corr_w[i,j] = signal.correlate(spikes[j],spikes[i,:-steps],"valid")
                


    def plot_correlation(self,lag_time=0):
        #Plots the correlation heatmap created in spike_correlation
        
        labels=dict(x="Neuron Index", y="Neuron Index", title='Correlation at {:.3f}(s)'.format(lag_time/self.fs), color="Correlation")

        fig = px.imshow(self.corr[...,lag_time],labels=labels)
        fig.show()

    def plot_correlation_bin(self,bin=0):
        #Plots the correlation heatmap created in spike_correlation
        #Does NOT run in real time, or on windowed correlations
        
        chunk = self.corr_steps//self.corr_bin.shape[2]
        title='Correlation at {:.3f}(s)-{:.3f}(s)'.format(chunk*bin/self.fs,chunk*(bin+1)/self.fs)

        labels=dict(x="Neuron Index", y="Neuron Index",title = title,color="Correlation")

        fig = px.imshow(self.corr_bin[...,bin],labels=labels)

        fig.show()


    def bin_correlation(self, bins):
        #Takes the correlation array and bins it
        self.corr_bin =  np.zeros((self.spikes.shape[0],self.spikes.shape[0],bins))

        for i in range(self.neurons):
            for j in range(self.neurons):
                self.corr_bin[i,j],_ = np.histogram(self.corr[i,j], bins = bins,weights=self.corr[i,j])


    def bin_correlation_window(self, bins):
        #Takes the correlation array and bins it
        self.corr_w_bin =  np.zeros((self.spikes.shape[0],self.spikes.shape[0],bins))

        for i in range(self.neurons):
            for j in range(self.neurons):
                self.corr_w_bin[i,j],_ = np.histogram(self.corr_w[i,j], bins = bins,weights=self.corr_w[i,j])



if __name__ == "__main__":

    n = Neuron('test_recording')

    n.load_spikes_test()
   

    n.load_map()

    # n.spike_correlation()
    # n.plot_correlation(lag_time = 0)
    # n.plot_spikes()


    #Timing for 30,0000 timesteps, 10 neurons
    print('Time to load: ',end='')
    t = time.time()
    n.load_spikes_test(length=30000,neurons=15)
    print('{:.3f} seconds'.format(time.time() - t))


    print('Time to process one second of correlation:')
    t = time.time()
    n.window_correlation(n.spikes,steps=30)
    print('{:.3f} seconds'.format(time.time() - t))



    #Now we bin the correlations, this can be optimized later(bin before corr)
    n.bin_correlation_window(3)


    #Test loop to pretend that we are recieving data from
    timesteps = int(1e6)
    n.load_spikes_test(length=timesteps,neurons=5,fs=150)

    #Split data so we can act like we are recieving

    #Receiving 1 per second
    slices = timesteps/n.fs
    corr_steps = 59
    bins = 15

    data = np.array_split(n.spikes,slices,axis=1)

    tot_corr_b = np.zeros((n.spikes.shape[0],n.spikes.shape[0],bins))
    tot_corr = np.zeros((n.spikes.shape[0],n.spikes.shape[0],corr_steps+1))
    n.spike_correlation(corr_steps)
    n.bin_correlation(bins)
    
    for num,d in enumerate(data):
        print(num)
        #Calculate correlations
        n.window_correlation(d,steps=corr_steps)
        n.bin_correlation_window(bins)

        tot_corr += n.corr_w
        tot_corr_b += n.corr_w_bin



    print('Total correlation',tot_corr_b[1,0])
    print('Actual correlation',n.corr_bin[1,0])

    
    fig,ax = plt.subplots(5,1)

    for i,a in enumerate(ax):
        a.plot(tot_corr[0,i])
        a.plot(n.corr[0,i])
        # a.set_yticklabels([])
    plt.show()
    
    fig,ax = plt.subplots(5,1)

    for i,a in enumerate(ax):
        a.plot(tot_corr_b[0,i])
        a.plot(n.corr_bin[0,i],alpha=.5)
        # a.set_yticklabels([])
    plt.show()
