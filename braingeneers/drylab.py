from warnings import warn
from functools import partial

import numpy as np

_mpl_import_error = None
try: # If we don't have mpl, silently ignore it.
    import matplotlib as mpl
except ImportError as e:
    mpl = None
    _mpl_import_error = e

_torch_import_error = None
try:
    import torch
except ImportError as e:
    torch = None
    _torch_import_error = e

from scipy import sparse, ndimage

import braingeneers.analysis as _analysis


# A map from neuron type abbreviation to ordered list of parameters
# a, b, c, d, C, k, Vr, Vt, Vp, Vn, and tau from Dynamical Systems in
# Neuroscience.  NB: many of these models have some extra bonus
# features in the book, used to more accurately reproduce traces from
# electrophysiological experiments in the appropriate model
# organisms. In particular,
#  - LTS caps the value of u but (along with a few other types) allows
#     it to influence the effective value of spike threshold and c.
#  - Several other types have PWL u nullclines.
NEURON_TYPES = {
    'rs':  [0.03, -2, -50, 100, 100, 0.7, -60, -40, 35,   0,  5],
    'ib':  [0.01,  5, -56, 130, 150, 1.2, -75, -45, 50,   0,  5],
    'ch':  [0.03,  1, -40, 150,  50, 1.5, -60, -40, 25,   0,  5],
    'lts': [0.03,  8, -53,  20, 100, 1.0, -56, -42, 20, -70, 20],
    'ls':  [0.17,  5, -45, 100,  20, 0.3, -66, -40, 30, -70, 20]}


class Organoid():
    """
    A simulated 2D culture of cortical cells using models from
    Dynamical Systems in Neuroscience, with synapses implemented as
    exponential PSPs for both excitatory and inhibitory cells.

    The model represents the excitability of a neuron using three
    phase variables: the membrane voltage v : mV, the "recovery" or
    "leakage" current u : pA, and the synaptic activation at each
    cell, a unitless A.

    The synapses in the updated model are conductance-type, with
    synaptic conductance following the alpha function of the synaptic
    actiavtion times the peak conductance G[i,j]. This adds another
    parameter to each cell: the Nernst reversal potential Vn of its
    neurotransmitter. Synaptic activation pulls the membrane voltage
    of the postsynaptic cell towards the reversal potential of the
    presynaptic cell.

    Additionally, the excitation model contains the following static
    parameters, on a per cell basis by providing arrays of size (N,):
     a : 1/ms time constant of recovery current
     b : nS steady-state conductance for recovery current
     c : mV membrane voltage after a downstroke
     d : pA bump to recovery current after a downstroke
     C : pF membrane capacitance
     k : nS/mV voltage-gated Na+ channel conductance
     Vr: mV resting membrane voltage when u=0
     Vt: mV threshold voltage when u=0
     Vp: mV action potential peak, after which reset happens
    tau: ms time constant for synaptic activation
     Vn: mV Nernst potential of the cell's neurotransmitter

    Finally, there is an optional triplet STDP rule for unsupervised
    learning, from Pfister and Gerstner, J. Neurosci. 26(38):9673.
    An arbitrary weight maximum has been introduced, but the proper
    way to do this is through synaptic scaling or homeostatic
    modulation of intrinsic excitability (TODO).

    The default parameters for STDP are taken from the same source,
    which derived them from a fit to V1 data recorded by Sjoström.
    """
    def __init__(self, *, XY=None, G,
                 a, b, c, d, C, k, Vr, Vt, Vp, Vn, tau,
                 # Whether to use torch or numpy arrays.
                 usetorch=False,
                 # STDP parameters.
                 do_stdp=False, stdp_tau_plus=15, stdp_tau_minus=35,
                 stdp_tau_y=115, stdp_Aplus=6.5e-3, stdp_Aminus=7e-3):

        try:
            if usetorch:
                conv = torch.FloatTensor
                stack = torch.stack
            else:
                conv = partial(np.asarray, dtype=np.float32)
                stack = np.vstack
        except AttributeError:
            raise _torch_import_error

        # Awkwardly, we have to save these in order to deal with the
        # differences between torch and numpy backends.
        self._conv = conv
        self._stack = stack

        self.G = conv(G)
        self.N = G.shape[0]
        if XY is not None:
            self.XY = conv(XY)
        self.a = conv(a)
        self.b = conv(b)
        self.c = conv(c)
        self.d = conv(d)
        self.C = conv(C)
        self.k = conv(k)
        self.Vr = conv(Vr)
        self.Vt = conv(Vt)
        self.Vp = conv(Vp)
        self.Vn = conv(Vn)
        self.tau = conv(tau)
        self.VUA = conv(np.zeros((4,self.N)))

        # STDP by the triplet model of Pfister and Gerstner (2006).
        # We store three synaptic traces at three different time
        # constants.
        self.do_stdp = do_stdp
        self.traces = conv(np.zeros((3,self.N)))
        stdp_taus = [stdp_tau_plus, stdp_tau_minus, stdp_tau_y]
        self.tau_stdp = conv([[tau] for tau in stdp_taus])
        self.Aplus = stdp_Aplus
        self.Aminus = stdp_Aminus

        self.reset()

    def reset(self):
        self.VUA[0,:] = self.Vr
        self.fired = self.V >= self.Vp
        self.VUA[1:,:] = 0

    def VUAdot(self, Iin):
        NAcurrent = self.k*(self.V - self.Vr)*(self.V - self.Vt)
        syncurrent = self.G@(self.A * self.Vn) - (self.G@self.A) * self.V
        Vdot = (NAcurrent - self.U + syncurrent + Iin) / self.C
        Udot = self.a * (self.b*(self.V - self.Vr) - self.U)
        Adot = self.Adot / self.tau
        Addot = -(self.A + 2*self.Adot) / self.tau
        return self._stack([Vdot, Udot, Adot, Addot])

    def step(self, dt, Iin):
        """
        Simulate the organoid for a time dt, subject to an input
        current Iin.
        """

        # Apply the correction to any cells that crossed the AP peak
        # in the last update step, so that this step puts them into
        # the start of the refractory period.
        fired = self.fired
        self.V[fired] = self.c[fired]
        self.U[fired] += self.d[fired]
        self.Adot[fired] += 1

        if self.do_stdp:
            any = fired.any()

            if any:
                # Update for presynaptic spikes.
                pre_mod = self.Aminus*self.traces[1,fired]
                self.G[:,fired] -= pre_mod

                # Update for postsynaptic spikes.
                post_mod = self.traces[0,:] * self.traces[2,fired,None]
                self.G[fired,:] += self.Aplus * post_mod

            # Even if no cells fired, the traces decay
            self.traces *= np.exp(dt / self.tau_stdp)

            # Cells which fired increment their traces.
            if any: self.traces[:,fired] += 1

        # Actually do the stepping, using the midpoint method for
        # integration. This costs as much as halving the timestep
        # would in forward Euler, but increases the order to 2.
        Iin = self._conv(Iin)
        k1 = self.VUAdot(Iin)
        self.VUA += k1 * dt/2
        k2 = self.VUAdot(Iin)
        self.VUA += k2*dt - k1*dt/2

        # Make a note of which cells this step has caused to fire,
        # then correct their membrane voltages down to the peak.  This
        # can make some of the traces look a little weird; it may be
        # prettier to adjust the previous point UP to self.Vp and set
        # this point to self.c, but that's not possible here since we
        # don't save all states.
        self.fired = self.V >= self.Vp
        self.V[self.fired] = self.Vp[self.fired]


    @property
    def V(self):
        return self.VUA[0,:]

    @V.setter
    def V(self, value):
        self.VUA[0,:] = value

    @property
    def U(self):
        return self.VUA[1,:]

    @U.setter
    def U(self, value):
        self.VUA[1,:] = value

    @property
    def A(self):
        return self.VUA[2,:]

    @A.setter
    def A(self, value):
        self.VUA[2,:] = value

    @property
    def Adot(self):
        return self.VUA[3,:]

    @Adot.setter
    def Adot(self, value):
        self.VUA[3,:] = value


class ChargedMedium():
    """
    Models the evolution of charge distribution with time in a medium
    full of diffusing ionic charge carriers, making two assumptions:
     1) charges are interchangeable and have identical properties,
     2) and they do not interact, i.e. an electron gas model.
    """

    def __init__(self, Xgrid, Ygrid,
                 D=2, eps_rel=80):
        self.D = D
        self.X, self.Y = Xgrid, Ygrid
        self._grid = np.array(np.meshgrid(Xgrid, Ygrid, indexing='ij'))
        self.dx, self.dy = Xgrid[1] - Xgrid[0], Ygrid[1] - Ygrid[0]
        self.rho = np.zeros(Xgrid.shape + Ygrid.shape)
        self.sigma = np.sqrt(2*D) / np.array([self.dx, self.dy])

        # Somehow, this code is off by a factor of 10^6 unless
        # I make the units of the free space permittivity wrong
        # in the following way. (It should be 10^-6 times this.)
        self._eps_factor = 4*np.pi* eps_rel * 8.854187

    def immerse(self, organoid):
        "Immerse an organoid in this medium for measurement."
        self.org = organoid
        self.Vprev = self.org.V.copy()

        # Quauntize neurons to grid points. If any neurons are
        # outside the grid, an error will be thrown later...
        self._neuron_grid = np.array(
            [np.argmin(abs(self.org.XY[0] - self.X[:,None]), axis=0),
             np.argmin(abs(self.org.XY[1] - self.Y[:,None]), axis=0)])

    def step(self, dt):
        """
        Run one forward simulation step. This will not work if the
        size of the filter is too small!
        """

        # Take the change in medium charge density due to the
        # membrane currents of all cells. Implementation note:
        # coordinate-form sparse matrices sum the contributions
        # of duplicate coordinates, i.e. this has the semantics
        # of looping over the cell grid but does it in C rather
        # than Python for a bit of a speed boost. The equivalent
        # Python loop is commented out below in case anyone cares.
        charge = self.org.C*(self.Vprev - self.org.V) / (self.dx*self.dy)
        self.rho += sparse.coo_matrix((charge, self._neuron_grid),
                                      shape=self.rho.shape)

        # for i,(x,y) in enumerate(zip(*self._neuron_grid)):
        #     self.rho[x,y] += charge[i]

        # Take one timestep of the diffusion process. Raise a warning
        # if the timestep is too small for diffusion to appear.
        sigma = self.sigma * np.sqrt(dt)
        if np.any(4*sigma < 1):
            warn('Timestep too small or grid too coarse;'
                 ' diffusion step has no effect.',
                 RuntimeWarning)
        self.rho = ndimage.gaussian_filter(self.rho, mode='constant',
                                           sigma=sigma)

        # Copy a new previous voltage.
        self.Vprev = self.org.V.copy()

    def probe_at(self, points):
        "Set probe point locations."

        # Save the points for caching.
        self._points = points

        # Distance from the probe points to each grid point.
        self._r = np.linalg.norm(points[:,...,None,None] -
                                 self._grid[:,None,...], axis=0)

        # Distance from the probe points to each cell.
        self._d = np.linalg.norm(points[:,...,None] -
                                 self.org.XY[:,None,...], axis=0)

    def probe(self, points=None):
        """
        Probe the voltage at the currently selected probe points.
        Optionally, you can provide the points, and it will select
        them by calling probe_at() for you. The last value is cached
        since usually you'll probe the same point set repeatedly.
        """

        if points not in (None, self._points):
            self.probe_at(points)

        # Contribution from the medium charge distribution: the
        # integral of charge distribution divided by distance.
        dist = np.trapz(np.trapz(self.rho/self._r, self.Y), self.X)

        # Contribution from the charge trapped inside the cells:
        # the sum per cell of the difference from resting potential.
        cells = self.org.C * (1/self._d) @ (self.org.V - self.org.Vr)

        # Divide by 4 pi epsilon_0 to get the actual potential.
        return (dist + cells) / self._eps_factor


class Ca2tImage():
    """
    Generate a single image without taking ownership of the Organoid's
    simulation stepping. Each firing creates fluorescence which initially
    displays as a pixel of intensity REACTIVITY (with 1 being fully
    saturated) and decays exponentially from there at rate TAU.
    """
    def __init__(self, cell_position, cell_size,
                 tau, reactivity, fig=None, **kwargs):

        if _mpl_import_error is not None:
            raise _mpl_import_error

        self.tau = tau
        self.X = np.zeros(cell_position.shape[1])

        # Create the scatter plot...
        self.fig = fig
        self.ax = self.fig.gca(aspect='equal')
        self.ax.patch.set_facecolor((0,0,0))
        self.scat = self.ax.scatter(*cell_position,
                                    s=cell_size, c=self.X,
                                    cmap='gray', alpha=0.5,
                                    norm=mpl.colors.Normalize(0,1),
                                    **kwargs)

    def animate(self, dt, events, **kwargs):
        func = partial(self.step, dt)
        frames = _gen_frame_events(dt, events)
        return mpl.animation.FuncAnimation(self.fig, func=func,
                                           interval=dt,
                                           frames=frames,
                                           **kwargs)

    def step(self, dt, events):
        """
        Given a timestep (which is used to determine the decay of the
        fluorescence level), plus the delta-t and cell index of each
        firing event since the last call, updates the fluorescence
        state and scatter plot.
        """
        self.X *= np.exp(-dt / self.tau)
        for time, cell in events:
            self.X[cell] += np.exp(-time / self.tau)

        self.scat.set_array(self.X)


def _gen_frame_events(dt, events):
    """
    Given a list of firing events in the form (time, cell index),
    groups them into batches of all events in time intervals of
    length dt.
    """
    T, evs = dt, []
    for time, cell in events:
        if time > T:
            yield evs
            T += dt
            evs = []
        evs.append((T - time, cell))


class ElectrodeArray():
    """
    An electrical microelectrode array: a rectangular grid where
    each point stimulates nearby cells in a Neurons object.

    You pass a specification of the grid geometry, then an amount
    of activation per pin (input should have the same shape as the
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

    def insert(self, org, vr, alpha):
        """
        Insert this array into an organoid. Precomputes the connectivity
        matrix from the array's inputs to the cells. You need to provide
        the resting voltages vr and a per-cell scaling alpha as well.
        """
        # The distance from the ith cell to the jth probe.
        dij = org.XY.reshape((2,-1,1)) - self.points.reshape((2,1,-1))
        dij = np.linalg.norm(dij / self.radius, axis=0, ord=2)
        self.M = 1 / np.maximum(1, dij)
        self.M = np.diag(alpha) @ self.M
        self.org = org
        self.vr = vr

    def Vprobe(self):
        return self.M.T @ (self.vr - self.org.V)

    def Iout(self, t):
        return self.M @ self.activation(t)


class OrganoidWrapper():
    def __init__(self, N, usetorch=True, input_scale=200, noise=0.1, dt=1, do_stdp=False):
        """
        Wraps an Organoid with easier initialization and timestepping
        for machine learning applications. An Organoid with N cells is
        constructed; this has been tested mostly at N=1000, so if you
        have a different number of inputs, use an input matrix that
        assigns each input to a random subset of neurons.

        Inputs are prescaled to convert from unitless values to
        currents; the default value of 200 is tuned towards inputs in
        the range (0,1). Additionally, random noise is added to the
        input before injecting it to the Organoid, with SNR=noise.

        Simulations are run with the timestep dt, and you can select
        whether to use the torch backend and whether to attempt to
        ``learn'' using STDP by passing keyword arguments.
        """

        # Let 80% of neurons be excitatory as observed in vivo.
        Ne = int(0.8 * N)

        # We're going to assign cells to four different types:
        # excitatory cells linearly interpolate between RS and Ch, and
        # inhibitory cells between LTS and LS. The weights are random,
        # but with different distributions: inhibitory identity is
        # uniform, whereas excitatory identity is squared to create a
        # bias towards RS cells, which are more common in vivo.
        identity = np.random.rand(N)
        celltypes = np.zeros((4,N))
        celltypes[0,:Ne] = identity[:Ne]**2
        celltypes[1,:Ne] = 1 - celltypes[0,:Ne]
        celltypes[2,Ne:] = identity[Ne:]
        celltypes[3,Ne:] = 1 - celltypes[2,Ne:]

        # Stack the parameters of each type into one array.
        typeparams = np.array([
            NEURON_TYPES['rs'],
            NEURON_TYPES['ch'],
            NEURON_TYPES['lts'],
            NEURON_TYPES['ls']])

        # Compute the parameters by interpolation.
        a, b, c, d, C, k, Vr, Vt, Vp, Vn, tau = typeparams.T @ celltypes

        # Synaptic conductances are lognormally distributed; the
        # parameters were originally derived from a biological source
        # describing the distribution in rat V1, but in theory the
        # actual values shouldn't be very important.
        G = np.random.lognormal(mean=-1.8, sigma=0.94, size=(N,N))
        # Inhibitory synapses are stronger because there are 4x fewer.
        G[:,Ne:] *= 8

        # XY : um planar positions of the cells,
        XY = np.random.rand(2,N) * 75

        # Use those positions to generate random small-world
        # connectivity using the modified Watts-Strogatz algorithm
        # from the braingeneers.analysis sublibrary. I've chosen
        # a characteristic length scale of 10μm and local connection
        # probability within that region of 50%. Then, only excitatory
        # synapses have a 2.5% chance of rewiring to a distant neighbor.
        # This is a boolean connectivity matrix, which is used to
        # delete most of the synapses.
        beta = np.zeros((1,N))
        beta[:Ne] = 2.5e-2
        G *= _analysis.small_world(XY/10, plocal=0.5, beta=beta)

        self.org = Organoid(XY=XY, G=G, tau=tau, a=a, b=b, c=c, d=d,
                            k=k, C=C, Vr=Vr, Vt=Vt, Vp=Vp, Vn=Vn,
                            usetorch=usetorch, do_stdp=do_stdp)
        self.N = N
        self.noise = noise
        self.dt = dt
        self.input_scale = input_scale


    def total_firings(self, input, interval):
        """
        Simulates the Organoid for a time interval subject to a fixed
        input current, and returns an array containing the number of
        firings for each cell during that time.
        """
        sigma = self.noise * self.input_scale
        Iin = lambda: (self.input_scale * input
                       + sigma * np.random.rand(self.N))

        firings = np.zeros(self.N)
        while interval > self.dt:
            self.org.step(self.dt, Iin())
            firings[self.org.fired] += 1
            interval -= self.dt
        self.org.step(interval, Iin())

        return firings


    def activation_after(self, input, interval):
        """
        Simulates the Organoid for a time interval subject to a fixed
        input current, and returns the array of presynaptic
        activations at the end of that time.
        """
        org = self.org
        num_inputs = self.N

        sigma = self.noise * self.input_scale
        Iin = lambda: (self.input_scale * input
                       + sigma * np.random.rand(self.N))

        while interval > self.dt:
            org.step(self.dt, Iin())
            interval -= self.dt
        org.step(interval, Iin())

        return org.A

    def synapses(self):
        "Retrieves the synaptic strengths from the organoid."
        return self.org.G
