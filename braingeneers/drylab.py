from warnings import warn
from functools import partial

import numpy as np
from scipy import sparse, ndimage, spatial
import braingeneers.analysis as _analysis
import braingeneers.buildnn as _buildnn


# If we don't have matplotlib, create a dummy object instead that
# raises an error if any attribute is requested, so that there is only
# an ImportError if matplotlib is actually requested.
try:
    import matplotlib as mpl
except ImportError as e:
    class mpl():
        def __init__(self, e):
            self.e = e
        def __getattr__(self, attr):
            raise self.e
    mpl = mpl(e)

# Create the default numpy backend.
class backend_numpy():
    array = partial(np.asarray, dtype=np.float32)
    stack = np.stack
    exp = np.exp
    sign = np.sign

# Try to import torch; if successful, create the torch backend,
# otherwise create a fake backend that will error if used, like mpl.
try:
    import torch
    class backend_torch():
        _has_cuda = torch.cuda.is_available()
        array = torch.cuda.FloatTensor if torch.cuda.is_available() \
            else torch.FloatTensor
        stack = torch.stack
        exp = torch.exp
        sign = torch.sign
except ImportError as e:
    class backend_torch():
        _has_cuda = False
        def __init__(self, e):
            self.e = e
        def __getattr__(self, attr):
            raise self.e
    backend_torch = backend_torch(e)


class Organoid(_buildnn.IzhikevichNeurons):
    """
    A simulated 2D culture of cortical cells as Izhikevich neurons
    with all-to-all connectivity through conductance synapses with
    exponential time activation.

    These features are implemented by the classes IzhikevichNeurons
    and ExponentialSynapses in the braingeneers.buildnn package.
    """
    def __init__(self, *, XY=None, G, dt=1,
                 a, b, c, d, C, k, Vr, Vt, Vp, Vn, tau, noise_rate,
                 backend=backend_numpy,
                 # STDP parameters, updated slightly.
                 do_stdp=False,
                 stdp_tau_pre=15, stdp_tau_post1=35,
                 stdp_tau_post2=115, stdp_Aplus2=0,
                 stdp_Aplus3=6.5e-3, stdp_Aminus2=7.1e-3,
                 # Synaptic scaling parameters, independent of STDP.
                 do_scaling=False, scaling_A=1e-3,
                 scaling_rate_target=0.05, scaling_tau=1000):

        # The torch backend is not yet supported by buildnn simulation
        # code, so error if it is requested.
        if backend is not backend_numpy:
            raise NotImplementedError

        # Just pass the parameters on.
        super().__init__(N=G.shape[0], dt=dt, a=a, b=b, c=c, d=d,
                         C=C, k=k, Vr=Vr, Vt=Vt, Vp=Vp)
        self.syn = _buildnn.ExponentialSynapses(
            G, self, Vn=Vn, tau=tau, noise_rate=noise_rate)

        if do_stdp:
            _buildnn.MinimalTripletSTDP(
                self.syn, tau_pre=stdp_tau_pre, tau_post1=stdp_tau_post1,
                tau_post2=stdp_tau_post2, A_plus2=stdp_Aplus2,
                Aplus3=stdp_Aplus3, Aminus2=stdp_Aminus2)

        if do_scaling:
            _buildnn.SynapticScaling(self.syn, tau=scaling_tau,
                                     rate_target=scaling_rate_target,
                                     Ascaling=scaling_A)

    @property
    def A(self):
        return self.syn.a

    @A.setter
    def A(self, value):
        self.syn.a[:] = value


class DipoleOrganoid(Organoid):
    """
    An extension o the above Organoid class for fast computation of
    extracellular potentials: each cell is modeled as a pure dipole,
    with all currents entering at the soma and exiting at the end.
    """
    def __init__(self, *, XY, dXdY, **kw):
        super().__init__(XY=XY, **kw)
        self.dXdY = dXdY

    def probe_at(self, points, radius=5):
        dijA = radius + spatial.distance_matrix(points, self.XY)
        dijB = radius + spatial.distance_matrix(points, self.XY
                                                + self.dXdY)
        Itot = self.Idyn + self.Isyn
        return -1/(4*np.pi*0.3) * (1/dijA - 1/dijB) @ Itot


class TripoleOrganoid(Organoid):
    """
    An extension of the above Organoid class for fast computation of
    extracellular potentials: each cell is modeled as a pair of
    dipoles with dynamical currents entering at the soma and exiting
    at the "axon" end, whereas synaptic currents enter at the dendrites
    but exit at the soma.
    """
    def __init__(self, *, XY, dXdY_axon, dXdY_dend, **kw):
        super().__init__(XY=XY, **kw)
        self.dax = dXdY_axon
        self.dde = dXdY_dend

    def probe_at(self, points, radius=5):
        dijS = radius + spatial.distance_matrix(points, self.XY)
        dijA = radius + spatial.distance_matrix(points, self.XY
                                                + self.dax)
        dijD = radius + spatial.distance_matrix(points, self.XY
                                                + self.dde)
        VA = (1/dijA - 1/dijS) @ self.Idyn
        VD = (1/dijS - 1/dijD) @ self.Isyn
        return -1/(4*np.pi*0.3) * (VA + VD)






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
        self._r = np.linalg.norm(points[:,...,None,None]
                                 - self._grid[:,None,...], axis=0)

        # Distance from the probe points to each cell.
        self._d = np.linalg.norm(points[:,...,None]
                                 - self.org.XY[:,None,...], axis=0)

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



class OrganoidWrapper(Organoid):
    def __init__(self, N, *, use_torch=None, do_scaling=False,
                 input_scale=100, dt=1, do_stdp=False,
                 noise_rate=5, p_rewire=2.5e-2,
                 world_bigness=10, scale_inhibitory=8,
                 G_mu=-1.8, G_sigma=0.94):
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
        learn using STDP by passing keyword arguments.
        """

        # In my experiments, CPU torch was always slower than numpy,
        # and GPU torch is faster only for very large networks, so
        # this chooses between torch and numpy by that heuristic.
        if use_torch is None:
            use_torch = backend_torch._has_cuda and N > 2500

        # Let 80% of neurons be excitatory as observed in vivo.
        Ne = int(0.8 * N)

        # We're going to assign cells to four different types:
        # excitatory cells linearly interpolate between RS and Ch, and
        # inhibitory cells between LTS and LS. The weights are random,
        # but with different distributions: inhibitory identity is
        # uniform, whereas excitatory identity is squared to create a
        # bias towards RS cells, which are more common in vivo.
        identity = np.random.rand(N)
        celltypes = np.zeros((N,4))
        celltypes[:Ne,0] = identity[:Ne]**2
        celltypes[:Ne,1] = 1 - celltypes[:Ne,0]
        celltypes[Ne:,2] = identity[Ne:]
        celltypes[Ne:,3] = 1 - celltypes[Ne:,2]
        params = _buildnn.IzhikevichNeurons.cell_types('RS CH LTS LS'.split(),
                                                       celltypes)

        # You also need to provide the synaptic parameters here.
        tau = np.zeros(N)
        tau[:Ne] = 5
        tau[Ne:] = 20
        Vn = np.ones(N)
        Vn[:Ne] = 0
        Vn[Ne:] = -100

        # Synaptic conductances are lognormally distributed; the
        # parameters were originally derived from a biological source
        # describing the distribution in rat V1, but in theory the
        # actual values shouldn't be very important.
        G = np.random.lognormal(mean=G_mu, sigma=G_sigma, size=(N,N))
        # Inhibitory synapses are stronger because there are 4x fewer.
        G[:,Ne:] *= scale_inhibitory

        # XY : µm planar positions of the cells.
        # These positions are supposed to approximately fit the areal
        # density of neurons in the chimpanzee cerebral cortex based
        # on a paper I looked up once, but the exact value is really
        # not important. This works together with the world_bigness
        # parameter to determine the locality of connectivity.
        XY = np.random.rand(2,N) * np.sqrt(N) * 2.5

        # Use those positions to generate random small-world
        # connectivity using the modified Watts-Strogatz algorithm
        # from the braingeneers.analysis sublibrary. I've chosen
        # a characteristic length scale of 10μm and local connection
        # probability within that region of 50%. Then, only excitatory
        # synapses have a 2.5% chance of rewiring to a distant neighbor.
        # This is a boolean connectivity matrix, which is used to
        # delete most of the synapses.
        beta = np.zeros((1,N))
        beta[:Ne] = p_rewire
        G *= _analysis.small_world(XY/world_bigness,
                                   plocal=0.5, beta=beta)

        super().__init__(XY=XY, G=G, **params, tau=tau, Vn=Vn,
                         noise_rate=noise_rate,
                         do_stdp=do_stdp, do_scaling=do_scaling,
                         backend=backend_torch if use_torch
                         else backend_numpy)
        self.dt = dt
        self.input_scale = input_scale

    def synapses(self):
        """
        Returns a copy of the synaptic connectivity matrix for
        external analysis.
        """
        return self.syn.G.copy()

    def activation_after(self, inp, interval):
        """
        Simulates the Organoid for a time interval subject to a fixed
        input current, and returns the array of presynaptic
        activations at the end of that time.
        """
        self.list_firings(inp, interval)
        return self.syn.a

    def list_firings(self, inp, interval):
        return super().list_firings(self.input_scale*inp, interval)

    def measure_criticality(self, duration):
        events = self.list_firings(0, duration)
        if len(events) == 0:
            return np.inf
        return _analysis.criticality_metric([t for (t,i) in events])
