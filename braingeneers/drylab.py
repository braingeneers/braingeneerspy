import numpy as np
import matplotlib.colors as colors
import matplotlib.pyplot as plt


# A map from neuron type abbreviation to ordered list of parameters
# a, b, c, d, C, k, Vr, Vt from Dynamical Systems in Neuroscience.
# NB: many of these models have some extra bonus features in the book,
# used to more accurately reproduce traces from electrophysiological
# experiments in the appropriate model organisms. In particular,
#  - LTS caps the value of u but (along with a few other types) allows
#     it to influence the effective value of spike threshold and c.
#  - I can't implement FS yet because its u nullcline is nonlinear.
#  - Several other types have PWL u nullclines.
#  - Different cell types have different spike thresholds.
NEURON_TYPES = {
    'rs':  [0.03, -2, -50, 100, 100, 0.7, -60, -40],
    'ib':  [0.01,  5, -56, 130, 150, 1.2, -75, -45],
    'ch':  [0.03,  1, -40, 150,  50, 1.5, -60, -40],
    'lts': [0.03,  8, -53,  20, 100, 1.0, -56, -42],
    'ls':  [0.17,  5, -45, 100,  20, 0.3, -66, -40]}


class Organoid():
    """
    A simulated 2D culture of cortical cells using models from
    Dynamical Systems in Neuroscience, with synapses implemented as
    exponential PSPs for both excitatory and inhibitory cells.

    The model represents the excitability of a neuron using three
    phase variables: the membrane voltage v : mV, the "recovery" or
    "leakage" current u : pA, and the total synaptic input to each
    cell Isyn : pA.

    Additionally, the excitation model contains the following static
    parameters, all of which can be set globally by providing scalars,
    or on a per cell basis by providing arrays of size (N,):
     a : 1/ms time constant of recovery current
     b : nS steady-state conductance for recovery current
     c : mV membrane voltage after a downstroke
     d : pA bump to recovery current after a downstroke
     C : pF membrane capacitance
     k : nS/mV voltage-gated Na+ channel conductance
     Vr: mV resting membrane voltage when u=0
     Vt: mV threshold voltage when u=0
     Vp: mV action potential peak, after which reset happens

    The cells are assumed to be located at physical positions contained
    in the variable XY : um, but this is not used for anything other than
    display (simulated Ca2+ imaging etc) unless you use it to generate
    the weight matrix S or something.
    """
    def __init__(self, *args, XY, S,
                 a=0.03, b=-2, c=-50, d=100, C=100, k=0.7,
                 Vr=-60, Vt=-40, Vp=30, tau=3):
        S = np.asarray(S)
        XY = np.asarray(XY)

        self.N = S.shape[0]

        self.S = S
        self.XY = XY
        self.a, self.b = a, b
        self.c = c * np.ones(self.N)
        self.d = d * np.ones(self.N)
        self.C, self.k, = C, k
        self.Vr, self.Vt, self.Vp = Vr, Vt, Vp
        self.tau = tau
        self.VUI = np.zeros((3, self.N))
        self.reset()

    def reset(self):
        self.VUI[0,:] = self.Vr
        self._fired = self.V >= self.Vp
        self.VUI[1:,:] = 0

    def VUIdot(self, Iin):
        NAcurrent = self.k*(self.V - self.Vr)*(self.V - self.Vt)
        Vdot = (NAcurrent - self.U + self.Isyn + Iin) / self.C
        Udot = self.a * (self.b*(self.V - self.Vr) - self.U)
        Idot = -self.Isyn / self.tau
        return np.array([Vdot, Udot, Idot])

    def step(self, Iin=0, dt=1):
        """
        Update the state of the organoid by 1ms, and return the current
        organoid state and a boolean array indicating which cells fired.

        The ODE is integrated by forward Euler since the dynamics are
        so chaotic and the parameters so noisy that that degree of
        numerical accuracy is completely redundant.
        """

        # Apply the correction to any cells that crossed the AP peak
        # in the last update step, so that this step puts them into
        # the start of the refractory period.
        fired = self._fired
        self.V[fired] = self.c[fired]
        self.U[fired] += self.d[fired]

        # Note that we store the total synaptic input to each cell and
        # let that decay over time.  The reason is just so that this
        # matmul has to happen only once per firing rather than once
        # per update.  The disadvantage is that the synaptic time
        # constant must be a global constant rather than per
        # presynaptic cell (per postsynaptic cell would be possible,
        # but doesn't make any sense).
        self.Isyn += self.S @ fired

        # Actually do the stepping, using the midpoint method for
        # integration. This costs as much as halving the timestep
        # would in forward Euler, but increases the order to 2.
        k1 = self.VUIdot(Iin)
        self.VUI += k1 * dt/2
        k2 = self.VUIdot(Iin)
        self.VUI += k2*dt - k1*dt/2


        self._fired = self.V >= self.Vp
        self.V[self._fired] = self.Vp

        # Return a pointer to the current state plus an array of which
        # cells fired during this update (NOT which cells are just about
        # to fire, which is the current content of self._fired).
        return self.VUI, fired

    @property
    def V(self):
        return self.VUI[0,:]

    @V.setter
    def V(self, value):
        self.VUI[0,:] = value

    @property
    def U(self):
        return self.VUI[1,:]

    @U.setter
    def U(self, value):
        self.VUI[1,:] = value

    @property
    def Isyn(self):
        return self.VUI[2,:]

    @Isyn.setter
    def Isyn(self, value):
        self.VUI[2,:] = value



class Ca2tCamera():
    """
    Generate a Pyplot illustration of an Organoid, approximately simulating
    Ca2+ imaging.

    The simulated camera averages the number n of firing events per ms
    over some period, smooths it using a moving-average filter, and
    activation of each cell grows logarithmically in firing rate.
    This gives activation that corresponds to firing frequency, without
    being able to directly measure the membrane voltage, and it fluctuates
    only slowly.
    """
    def __init__(self, n, *args,
                 tick=None, frameskip=0, window_size=1, reactivity=30,
                 Iin=lambda *args: 0, scatterargs={},
                 **kwargs):
        """
        Create a Ca2+ imaging figure! Pass in a figure and an Organoid
        object.  Also takes input current as a function of time
        Iin(t), and a function tick(n,t,*) to run on the Organoid
        at each frame. Then some parameters control the frames.

        You can set the amount of simulation time and real time per frame
        by combining this frameskip argument with the animator's frame
        interval argument: the real-time interval between simulation
        frames is interval/(frameskip + 1) ms, or in reverse, the video
        is (frameskip + 1)/interval times faster than real-time.

        The moving average filter is controlled by the parameter
        window_size: this is the number of the last internal frames
        which are averaged to produce each frame you actually see.

        Reactivity determines what is considered a "long time"
        between spikes: a cell lights up 60% if its average
        firing interval is equal to the reactivity.
        """
        self.window_size = window_size
        self.ticks_per_update = frameskip + 1
        self.n = n
        self.Iin = Iin
        self.reactivity = reactivity
        self._tick = tick

        self.X = np.zeros((window_size, n.N))
        self.scatterargs = scatterargs

    def tick(self, t, *args):
        if self._tick is not None:
            self._tick(t, *args)

    def init(self, fig=None):
        "Creates the scatter plot, must call before starting to record."
        self.fig = fig or plt.figure()
        self.ax = self.fig.gca()
        self.ax.set_aspect('equal')
        self.ax.patch.set_facecolor((0,0,0))
        self.scat = self.ax.scatter(self.n.XY[0,:], self.n.XY[1,:],
                                    s=25, c=self.X.mean(axis=0),
                                    cmap='gray',
                                    norm=colors.Normalize(vmax=1, vmin=0,
                                                          clip=True),
                                    **self.scatterargs)

    def update(self, T, *args, show=True):
        """
        Calculate one frame forward, for the Tth sampling period.
        Additional arguments can be passed to the tick() method
        """
        Tmod = T % self.window_size
        self.X[Tmod, :] = 0
        for dt in range(self.ticks_per_update):
            t = T*self.ticks_per_update + dt
            _, fired = self.n.step(self.Iin(t))
            self.X[Tmod, fired] += 1 / self.ticks_per_update
            self.tick(t, *args)

        if show:
            xavg = self.X.mean(axis=0)
            self.scat.set_array(1 - np.exp(-xavg * self.reactivity))
            return self.scat


class ElectrodeArray():
    """
    An electrical microelectrode array: a rectangular grid where
    each point stimulates nearby cells in a Neurons object.

    You pass a specification of the grid geometry, then an amount
    of activation per pin (output should have the same shape as the
    grid), and the array becomes a callable that can be
    """
    def __init__(self, *args,
                 spacing=None, shape=None, dimensions=None,
                 points=None, offset=(0,0), radius=10,
                 activation):
        if points is None:
            if dimensions is None:
                px, py = np.mgrid[:shape[0], :shape[1]] * spacing
            elif shape is None:
                try: spacing[0]
                except TypeError as _:
                    spacing = [spacing, spacing]
                px, py = np.mgrid[:dimensions[0]:spacing[0],
                                  :dimensions[1]:spacing[1]]
            elif spacing is None:
                px, py = np.mgrid[:dimensions[0]:shape[0]*1j,
                                  :dimensions[1]:shape[1]*1j]
            px = px.flatten() # - px.mean()
            py = py.flatten() # - py.mean()
            points = np.array((px, py))

        self.points = points + np.asarray(offset).reshape((2,1))
        self.activation = activation
        self.radius = radius

    def insert(self, n, vr, alpha):
        """
        Insert this array into an organoid. Precomputes the connectivity
        matrix from the array's inputs to the cells. You need to provide
        the resting voltages vr and a per-cell scaling alpha as well.
        """
        # The distance from the ith cell to the jth probe.
        dij = n.XY.reshape((2,-1,1)) - self.points.reshape((2,1,-1))
        dij = np.linalg.norm(dij / self.radius, axis=0, ord=2)
        self.M = 1 / np.maximum(1, dij)
        self.M = np.diag(alpha) @ self.M
        self.n = n
        self.vr = vr

    def Vprobe(self):
        return self.M.T @ (self.vr - self.n.V)

    def Iout(self, t):
        return self.M @ self.activation(t)
