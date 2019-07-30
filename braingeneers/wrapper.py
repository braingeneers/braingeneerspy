import numpy as np
import scipy.signal as spsig
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation
from tqdm import tqdm_notebook as tqdm

import .drylab
#from colorednoise import powerlaw_psd_gaussian as gen_noise


class OrganoidWrapper():

    # -------------- __init__()-------------------
    # N : number of cells in organoid == size of input vector
    # noise : fraction of noise added to input
    # input_scale : default is 100 if normalized input to 1
    # dt : (ms) is the dicretized slice of time of simulation (granularity?)
    # org : is instance Alex's Organoid() class

    def __init__(self, N, input_scale=100, noise=0.1, dt=1):

        # Number of neurons, followed by the number which are excitatory.
        Ne = int(0.8 * N)
        #So, inhibitory == N-Ne

        # Used for constructing nonhomogeneous neural populations,
        # interpolated between two types based on the value of
        # r ∈ [0,1]. Excitatory neurons go from Regular Spiking
        # to Chattering, while inhibitory neurons go from
        # Low-Threshold Spiking to Late Spiking models over the
        # same range. Adapted from Izhikevich's writings.
        r = np.random.rand(N) # unitless
        l = np.ones(N) # unitless

        # a : 1/ms recovery time constant of membrane leak currents
        a = np.hstack((0.03*l[:Ne], 0.03 + 0.14*r[Ne:]))
        # b : nS recovery conductivity
        b = np.hstack((-2 + 3*r[:Ne]**2, 8 - 3*r[Ne:]))
        # c : mV voltage of the downstroke
        c = np.hstack((-50 + 10*r[:Ne]**2, -53 + 8*r[Ne:]))
        # d : pA instantaneous increase in leakage during downstroke
        d = np.hstack((100 + 50*r[:Ne]**2, 20 + 80*r[Ne:]))
        # C : pF membrane capacitance
        C = np.hstack((100 - 50*r[:Ne]**2, 100 - 80*r[Ne:]))
        # k : nS/mV Na+ voltage-gated channel conductivity parameter
        k = np.hstack((0.7 + 0.8*r[:Ne]**2, 1 - 0.7*r[Ne:]))
        # mV : resting membrane voltage
        Vr = np.hstack((-60*l[:Ne], -56 - 10*r[Ne:]))
        # mV : threshold voltage at u=0
        Vt = np.hstack((-40*l[:Ne], -42 + 2*r[Ne:]))
        # mV : peak cutoff of action potentials
        Vp = 30

        # tau : ms time constant of synaptic current
        tau = 20 # np.hstack((5*l[:Ne], 20*l[Ne:]))

        #------------------------------------------

        # Sij : fC total postsynaptic charge injected into
        #       neuron i when neuron j fires. Song (2005)
        #       provide an empirical distribution for EPSPs.
        mu, sigma = -0.702, 0.9355
        S = np.random.lognormal(mean=mu, sigma=sigma, size=(N,N))
        # Then convert the EPSPs to injected synaptic charge.
        S *= np.median(C / tau)
        S[:,Ne:] *= -10

        # XY : um planar positions of the cells,
        # dij : um distances between cells
        XY = np.random.rand(2,N) * 75


        # Create the actual Organoid.
        org = drylab.Organoid(XY=XY, S=S*5, tau=tau,
                              a=a, b=b, c=c, d=d,
                              k=k, C=C, Vr=Vr, Vt=Vt, Vp=Vp)

        org.initialize_stdp()

        self.org = org
        self.N = N
        self.noise = noise
        self.dt = dt
        self.input_scale = input_scale




    # -------------- step()-------------------
    # input : inside step(), we multiply input array by ~100 because cells are excited ~100pA
    # if constant input (200ms, couple hunderd) activation of ~50pA is enough to excite
    # a noise factor is added to the input
    # Check: input must be same dimensions as organoid N (no error checking)
    # interval :  (ms) is duration to run simulation

    def step(self, input, interval):
        org = self.org
        num_inputs = self.N

        Iin = self.input_scale * input + self.noise * np.random.rand(self.N)

        # Run the loop.
        while interval > self.dt:
            org.step(self.dt, Iin)
            interval -= self.dt
        org.step(interval, Iin)

        #return arry of outputs
        return org.Isyn
