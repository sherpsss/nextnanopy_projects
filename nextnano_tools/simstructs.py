import numpy as np
import matplotlib.pyplot as plt

_DEFAULT_BAND_COLORS = {
    'CB':    'dodgerblue',
    'VB':    'crimson',
    'Gamma': 'dodgerblue',
    'X':     'steelblue',
    'L':     'cornflowerblue',
    'HH':    'crimson',
    'LH':    'darkorange',
    'SO':    'mediumpurple',
}

class Eigenstate:
    def __init__(self, index:int, energy:float, prob_dist:np.ndarray, nn_index:int=None):
        #for 1 eigenenstate of whatever band is relevant

        self.index = index
        self.nn_index = nn_index if nn_index is not None else index  # original nextnano 1-based index; preserved after sort_subbands
        self.energy = energy
        self.probab_dist = prob_dist

    def __repr__(self):
        return f"<Eigenstate #index={self.index}, energy={self.energy} eV>"
    
class BandEdge:
    def __init__(self, name, energy, x):
        #stores band edge data for specific edge (e.g., Gamma, HH, LH)
        self.name = name
        self.energy = energy  #energy profile
        self.x = x  #x coords

    def plot(self, ax=None, show=True, color=None, label=None, offset=0.0,
             fontsizebase=18, **kwargs):
        """
        Plot this band edge profile.

        Parameters
        ----------
        offset : float
            Subtract this value from energy before plotting (used by plot_band
            for normalize_y). Default 0.0.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        c = color or _DEFAULT_BAND_COLORS.get(self.name, 'gray')
        lbl = label if label is not None else self.name
        ax.plot(self.x, self.energy - offset, color=c, label=lbl, **kwargs)

        ax.set_xlabel("Growth direction [nm]", fontsize=fontsizebase)
        ax.set_ylabel("Energy relative to $E_F$ [eV]", fontsize=fontsizebase)
        ax.tick_params(axis='both', labelsize=fontsizebase)

        if show:
            plt.tight_layout()
            plt.show()

        return ax


class BandStructure:
    def __init__(self, name:str,subbands = None, bandedges = None,x=None):
        #stores eigenstates in CB or VB subband
        self.name = name
        self.subbands = [] if subbands is None else subbands
        self.bandedges = [] if bandedges is None else bandedges
        self.x = x  # spatial axis in nm
        self.dipole_moments = {}  # {polarization: {(nn_i, nn_j): |d|^2 [e^2·nm^2]}}
    
    def add_subband(self, eigenstate:Eigenstate):
        if not isinstance(eigenstate, Eigenstate):
            raise TypeError("eigenstate must be an instance of Eigenstate class")
        self.subbands.append(eigenstate)
    
    def get_energies(self):
        return np.array([subband.energy for subband in self.subbands])
    
    def sort_subbands(self,decreasing=True):
        self.subbands.sort(key=lambda x: x.energy, reverse=decreasing)
        for new_index, subband in enumerate(self.subbands,start=1):
            subband.index = new_index   # Update index to reflect new order
    
    def remove_subband(self, subband=None,index=None):
        if subband is not None:
            self.subbands = [s for s in self.subbands if s is not subband]
        elif index is not None:
            self.subbands = [s for s in self.subbands if s.index != index]
        else:
            raise ValueError("must provide subband object or index to remove")
        
        for new_index, subband in enumerate(self.subbands,start=1):
            subband.index = new_index
    
    def add_bandedge(self, edge:BandEdge):
            if not isinstance(edge, BandEdge):
                raise TypeError("bandedges must be instances of BandEdge class")
            self.bandedges.append(edge)

    def add_dipole_moments(self, polarization: str, data: dict):
        """
        Store intraband dipole matrix elements for one polarization.

        Parameters
        ----------
        polarization : str
            Label matching the nextnano output filename suffix (e.g. 'TM_z', 'component_x').
        data : dict
            {(nn_i, nn_j): |<i|eps·d|j>|^2  [e^2·nm^2]}  keyed by nextnano 1-based indices.
        """
        self.dipole_moments[polarization] = data

    def get_dipole_matrix(self, polarization: str) -> np.ndarray:
        """
        Return an (N x N) matrix of |dipole|^2 [e^2·nm^2] in the current subband order.
        Rows/cols follow self.subbands order (respects any prior sort_subbands call).
        """
        if polarization not in self.dipole_moments:
            raise KeyError(f"No dipole moments stored for polarization '{polarization}'. "
                           f"Available: {list(self.dipole_moments.keys())}")
        n = len(self.subbands)
        mat = np.zeros((n, n))
        d = self.dipole_moments[polarization]
        for row, si in enumerate(self.subbands):
            for col, sj in enumerate(self.subbands):
                mat[row, col] = d.get((si.nn_index, sj.nn_index), 0.0)
        return mat

    def get_dipole_vs_energy(self, polarization: str, upward_only=True):
        """
        Return dipole transition data for all subband pairs.

        Returns
        -------
        dE_vals : np.ndarray  transition energies [eV]
        d_vals  : np.ndarray  |dipole|^2 [e²·nm²]
        labels  : list[str]   'i→j' label strings
        """
        d_mat = self.get_dipole_matrix(polarization)
        dE_vals, d_vals, labels = [], [], []
        for i, si in enumerate(self.subbands):
            for j, sj in enumerate(self.subbands):
                if i == j:
                    continue
                dE = sj.energy - si.energy
                if upward_only and dE <= 0:
                    continue
                dE_vals.append(dE)
                d_vals.append(d_mat[i, j])
                labels.append(f"{si.index}→{sj.index}")
        return np.array(dE_vals), np.array(d_vals), labels

    def plot_dipole_vs_energy(self, polarization: str, upward_only=True, label_transitions=False,
                              ax=None, show=True, fontsizebase=18, fontsizetitle=22,
                              title_diff=None, **kwargs):
        """
        Stem plot of |dipole|^2 vs. transition energy for all subband pairs.

        Parameters
        ----------
        polarization : str
            Key matching a stored dipole polarization (e.g. 'TM_z', 'component_x').
        upward_only : bool
            If True (default), only show transitions i→j where E_j > E_i.
        label_transitions : bool
            If True, annotate each stem with 'i→j' subband labels.
        """
        dE_vals, d_vals, labels = self.get_dipole_vs_energy(polarization, upward_only=upward_only)

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        markerline, _, _ = ax.stem(dE_vals, d_vals, **kwargs)
        markerline.set_markersize(5)

        if label_transitions:
            for dE, d, lbl in zip(dE_vals, d_vals, labels):
                ax.annotate(lbl, (dE, d), textcoords="offset points",
                            xytext=(0, 4), ha='center', fontsize=fontsizebase - 6)

        ax.set_xlabel("Transition energy ΔE (eV)")
        ax.set_ylabel(r"$|d_{ij}|^2$ [e²·nm²]")
        ax.set_title(title_diff or f"{self.name} dipole moments — {polarization}")
        ax.xaxis.get_label().set_fontsize(fontsizebase)
        ax.yaxis.get_label().set_fontsize(fontsizebase)
        ax.title.set_fontsize(fontsizetitle)
        ax.tick_params(axis='both', labelsize=fontsizebase)

        if show:
            plt.tight_layout()
            plt.show()

        return ax
    
    def calc_intersubband_transitions(self, upward_only=False):
        Ei, Ef = np.meshgrid(self.get_energies(), self.get_energies())
        T = Ef - Ei
        if upward_only:
            mask = np.triu(np.ones_like(T, dtype=bool), k=1)
            return T[mask], mask
        return T

    def display_intersubband_transitions(self, upward_only=False, sort_by_deltaE=None):
        """
        Pretty-print intersubband transition table.

        Parameters
        ----------
        upward_only : bool
            If True, include only transitions where j > i.
        sort_by_deltaE : None, "ascending", or "descending"
            If provided, sorts flattened transitions by ΔE.
        """

        T = self.calc_intersubband_transitions(upward_only=False)
        n = T.shape[0]

        # -------------------------------------------------
        # Collect all transitions into a list
        # -------------------------------------------------
        transitions = []
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue  # skip zero ΔE transitions
                if upward_only and j <= i:
                    continue
                transitions.append((i+1, j+1, T[i, j]))  # store 1-based indices

        # -------------------------------------------------
        # Optional sorting
        # -------------------------------------------------
        if sort_by_deltaE is not None:
            reverse_flag = (sort_by_deltaE == "descending")
            transitions.sort(key=lambda x: x[2], reverse=reverse_flag)

        # -------------------------------------------------
        # Print nicely
        # -------------------------------------------------
        print(f"\nIntersubband transition energies — {self.name}")
        print("-----------------------------------------------------")
        print(" i → j | ΔE (eV)")
        print("-----------------------------------------------------")

        for i, j, dE in transitions:
            print(f" {i:2d} → {j:2d} | {dE: .4f}")

        print("-----------------------------------------------------\n")
    
    def define_x(self, x:np.ndarray): #if I don't define when I pass it in
        self.x = x

    def calc_ldos(self, eVbias_values: np.ndarray, deltaE: float = 0.0) -> np.ndarray:
        """
        Compute the local density of states as a function of position and bias energy.

        LDOS(eVbias, z) = sum of probab_dist for all subbands with energy <= eVbias.

        Parameters
        ----------
        eVbias_values : array-like
            Applied bias energies in eV.

        Returns
        -------
        ldos : np.ndarray, shape (len(eVbias_values), len(self.x))
            Rows correspond to eVbias, columns to z (growth direction).
        """
        if self.x is None:
            raise ValueError("Spatial axis x is not defined for this BandStructure.")
        # eVbias_values = np.asarray(eVbias_values) #already passed in as an array
        n_bias = len(eVbias_values)
        n_z = len(self.x)
        ldos = np.zeros((n_bias, n_z))
        for i, eVbias in enumerate(eVbias_values):
            for subband in self.subbands:
                if subband.energy <= (eVbias + deltaE):  # the deltaE allows for broadening from Vmod
                    ldos[i] += subband.probab_dist
        return ldos

    def plot_band(self, scale=0.05, fontsizebase=18, fontsizetitle=22, color=None, prob_alpha=0.45,
                 ax=None, show=True, show_legend=True, show_grid=False, title_diff=None, normalize_y=False):
        """
        Plot band edges and subband probability distributions.

        Parameters
        ----------
        scale : float
            Vertical scale factor for normalized probability amplitudes.
        color : str or None
            If None (default), subbands are colored by increasing energy using the
            'plasma' colormap. Pass an explicit color string to override.
        prob_alpha : float
            Alpha (opacity) for probability distribution lines. Default 0.45 so
            they are visually distinct from the solid energy-level lines.
        ax : matplotlib.axes.Axes
            Optional Axes object to plot into.
        show : bool
            Whether to call plt.show().
        normalize_y : bool
            If True, shift all energies so that 0 = minimum band-edge energy.
            (Useful for comparing band structures on a shared vertical scale.)
        """

        if self.x is None:
            raise ValueError("Spatial axis x is not defined for this BandStructure.")

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        # ------------------------------------------------------------------
        # Build per-subband colors
        # ------------------------------------------------------------------
        use_winter = color is None
        if use_winter:
            cmap = plt.colormaps['winter']
            subbands_by_energy = sorted(self.subbands, key=lambda s: s.energy)
            n = len(subbands_by_energy)
            energy_rank = {id(s): i for i, s in enumerate(subbands_by_energy)}

        # --------------------------------------------
        # Compute energy offset if normalize_y=True
        # --------------------------------------------
        if normalize_y:
            all_edge_vals = []
            for edge in self.bandedges:
                all_edge_vals.extend(edge.energy)

            if len(all_edge_vals) == 0:
                raise ValueError("normalize_y=True requested but no band edges exist.")

            E_min = np.min(all_edge_vals)
        else:
            E_min = 0.0

        # --------------------------------------------
        # Plot band edges (with optional normalization)
        # --------------------------------------------
        for edge in self.bandedges:
            edge.plot(ax=ax, show=False, offset=E_min, fontsizebase=fontsizebase)

        # --------------------------------------------
        # Plot each eigenstate energy + wavefunction
        # --------------------------------------------
        for subband in self.subbands:
            E = subband.energy - E_min

            if use_winter:
                rank = energy_rank[id(subband)]
                c = cmap(rank / max(n - 1, 1))
            else:
                c = color

            # energy level as solid horizontal line
            ax.plot(self.x, np.full_like(self.x, E), ls='--', color=c,
                    label=f'{self.name} {subband.index}', lw=2.0)

            # probability distribution — same color, semi-transparent
            psi2 = subband.probab_dist
            psi2_norm = psi2 / np.max(np.abs(psi2)) if np.max(np.abs(psi2)) != 0 else psi2
            ax.plot(self.x, psi2_norm * scale + E, color=c, alpha=prob_alpha, lw=1.5)

        if normalize_y:
            ax.set_ylabel("Energy relative to Band Edge [eV]")
        # else:
        #     ax.set_ylabel("Energy relative to $E_F$ [eV]")

        # ax.set_xlabel("Growth direction [nm]")
        if title_diff is not None:
            ax.set_title(title_diff)
        else:
            ax.set_title(f"{self.name} band structure")

        if show_legend:
            ax.legend(loc="best")
            ax.legend(fontsize=fontsizebase)

        ax.xaxis.get_label().set_fontsize(fontsizebase)
        ax.yaxis.get_label().set_fontsize(fontsizebase)
        ax.title.set_fontsize(fontsizetitle)
        ax.tick_params(axis='both', labelsize=fontsizebase)

        if show_grid:
            ax.grid(True)

        if show:
            plt.tight_layout()
            plt.show()

        return fig, ax

    def __repr__(self):
        n = len(self.subbands)
        if n==0:
            return f"<BandStructure name={self.name}, no subbands>"
        energies = ", ".join([f"{subband.energy:.4f} eV" for subband in self.subbands])
        return f"<BandStructure name={self.name}, {n} subbands: [{energies}]>"


class Spectrum:
    def __init__(self, x, y, x_unit="eV", y_label="Absorption (cm⁻¹)",
                 polarization=None, axis=None, bias=None, well_width=None):
        self.x = np.array(x)
        self.y = np.array(y)
        self.x_unit = x_unit
        self.y_label = y_label
        self.polarization = polarization
        self.axis = axis

    def plot(self, ax=None, show=True,show_grid=False, diff_title=None, fontsizebase=18,fontsizetitle=22, **kwargs):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        label = f"{self.polarization}-{self.axis}" if self.axis else self.polarization
        ax.plot(self.x, self.y, label=label, **kwargs)
        xlabel = "Photon Energy (eV)" if self.x_unit.lower() == "ev" else "Wavelength (nm)"
        ax.set_xlabel(xlabel)
        ax.set_ylabel(self.y_label)

        if diff_title is not None:
            ax.set_title(diff_title)
        ax.legend()

        #fontsize formatting
        ax.xaxis.get_label().set_fontsize(fontsizebase)
        ax.yaxis.get_label().set_fontsize(fontsizebase)
        ax.legend(fontsize=fontsizebase)
        ax.title.set_fontsize(fontsizetitle)
        ax.tick_params(axis='both', labelsize=fontsizebase)

        if show:
            plt.tight_layout()
            if show_grid:
                plt.grid()
            plt.show()

        return ax

    def normalize(self):
        """Return a normalized copy (max=1)."""
        y_norm = self.y / np.max(self.y)
        return Spectrum(self.x, y_norm, x_unit=self.x_unit,
                        y_label=self.y_label, polarization=self.polarization,
                        axis=self.axis, bias=self.bias, well_width=self.well_width)

    def subset(self, xmin=None, xmax=None):
        """Return a subset between xmin and xmax."""
        mask = np.ones_like(self.x, dtype=bool)
        if xmin is not None:
            mask &= self.x >= xmin
        if xmax is not None:
            mask &= self.x <= xmax
        return Spectrum(self.x[mask], self.y[mask],
                        x_unit=self.x_unit, y_label=self.y_label,
                        polarization=self.polarization, axis=self.axis,
                        bias=self.bias, well_width=self.well_width)

    def __repr__(self):
        pol = f"{self.polarization}-{self.axis}" if self.axis else self.polarization
        return f"<Spectrum {pol}, {len(self.x)} points, x_unit={self.x_unit}>"


class OpticalAbsorption:
    def __init__(self):
        self.spectra = {}  # dict[str, Spectrum]

    def add_spectrum(self, spectrum: Spectrum):
        if not isinstance(spectrum, Spectrum):
            raise TypeError("spectrum must be a Spectrum instance")
        label = f"{spectrum.polarization}-{spectrum.axis}" if spectrum.axis else spectrum.polarization
        self.spectra[label] = spectrum

    def get_spectrum(self, label: str):
        return self.spectra[label]

    def plot(self, labels=None, ax=None, show=True,fontsizebase=18,fontsizetitle=22):
        if ax is None:
            fig, ax = plt.subplots(figsize=(10,6))
        if labels is None:
            labels = list(self.spectra.keys())
        for label in labels:
            self.spectra[label].plot(ax=ax, show=False)
        ax.legend(title="Polarization")

        #fontsize formatting
        ax.xaxis.get_label().set_fontsize(fontsizebase)
        ax.yaxis.get_label().set_fontsize(fontsizebase)
        ax.legend(fontsize=fontsizebase)
        ax.title.set_fontsize(fontsizetitle)
        ax.tick_params(axis='both', labelsize=fontsizebase)

        
        if show:
            plt.tight_layout()
            plt.show()
        return ax

    def __repr__(self):
        return f"<OpticalAbsorption: {list(self.spectra.keys())}>"

    
_KNOWN_DENSITY_COLORS = {
    'Electron_density':         'mediumblue',
    'Hole_density':             'crimson',
    'Ionized_donor_density':    'darkorange',
    'Ionized_acceptor_density': 'mediumpurple',
}


class Density:
    def __init__(self, density_name: str, x: np.ndarray, density: np.ndarray, units: str = '1e18 cm⁻³'):
        self.density_name = density_name
        self.x = x
        self.density = density
        self.units = units

    def plot(self, log_scale=False, ax=None, show=True,
             fontsizebase=18, fontsizetitle=22, title_diff=None):
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))

        color = _KNOWN_DENSITY_COLORS.get(self.density_name, 'gray')

        ax.plot(self.x, self.density, color=color, lw=2.0)

        if log_scale:
            ax.set_yscale('log')

        ax.set_xlabel("Growth direction [nm]", fontsize=fontsizebase)
        ax.set_ylabel(f"{self.density_name} [{self.units}]", fontsize=fontsizebase)
        ax.set_title(title_diff or self.density_name, fontsize=fontsizetitle)
        ax.tick_params(axis='both', labelsize=fontsizebase)

        if show:
            plt.tight_layout()
            plt.show()

        return ax

    def __repr__(self):
        return f"<Density name={self.density_name}, {len(self.x)} points, units={self.units}>"


class SimOut:
    def __init__(self, simname:str):
        #represents 1 full sim result
        self.simname = simname
        self.electron_Fermi_level = None
        self.hole_Fermi_level = None
        self.bands ={}
        self.optical_absorption = OpticalAbsorption() #only included in some sims
        self.interband_dipole_moments = {}  # {polarization: {(nn_i, nn_j): |d|^2 [e^2·nm^2]}}
        self.variables = {}  # {name: int or float} from variables_input.txt
        self.densities = {}  # {density_name: Density} — populated from all density .dat files
    
    def add_band(self, band):
        #pass in either bandstructure type or just new string type
        if isinstance(band, BandStructure):
            self.bands[band.name] = band
        elif isinstance(band, str):
            if band not in self.bands:
                self.bands[band] = BandStructure(band)
        else:
            raise TypeError("band must be a BandStructure instance or a string")

    def add_subband(self, band_name:str, subband:Eigenstate = None, index:int = None, energy:float = None, prob_dist:np.ndarray = None):
        #pass eigenstate obj directly or pass in energy, prob dist, and subband index and then create obj
        if band_name not in self.bands:
            self.add_band(band_name)
        
        if subband is not None:
            if not isinstance(subband, Eigenstate):
                raise TypeError("subband must be an instance of Eigenstate class")
            self.bands[band_name].add_subband(subband)
        elif all(v is not None for v in [index, energy, prob_dist]):
            new_subband = Eigenstate(index, energy, prob_dist)
            self.bands[band_name].add_subband(new_subband)
        else:
            raise ValueError("Either subband or all of index, energy, and prob_dist must be provided")
        
    def sort_all_bands(self,decreasing=True):
        for band in self.bands.values():
            band.sort_subbands(decreasing=decreasing)
    
    def calc_interband_transitions(self, upper='CB', lower='VB'):
        """
        Compute interband transition energies between two bands.

        Parameters
        ----------
        upper : str
            Name of the upper (higher energy) band. Default 'CB'.
            For 1-band model use e.g. 'Gamma'.
        lower : str
            Name of the lower (lower energy) band. Default 'VB'.
            For 1-band model use e.g. 'HH' or 'LH'.

        Returns
        -------
        transitions : 2D numpy array
            transitions[i, j] = upper_i - lower_j
        """
        if upper not in self.bands or lower not in self.bands:
            raise ValueError(f"Both '{upper}' and '{lower}' bands must exist. Available: {list(self.bands.keys())}")

        lower_mesh, upper_mesh = np.meshgrid(self.bands[lower].get_energies(), self.bands[upper].get_energies())
        return upper_mesh - lower_mesh

    
    def add_interband_dipole_moments(self, polarization: str, data: dict):
        """
        Store interband (CB-VB) dipole matrix elements for one polarization.

        Parameters
        ----------
        polarization : str
            Label matching the nextnano output filename suffix.
        data : dict
            {(nn_i, nn_j): |<i|eps·d|j>|^2  [e^2·nm^2]}  where i and j may be
            from different bands; nextnano 1-based indices.
        """
        self.interband_dipole_moments[polarization] = data

    def get_interband_dipole_matrix(self, polarization: str, upper='CB', lower='VB') -> np.ndarray:
        """
        Return an (N_upper x N_lower) matrix of |dipole|^2 [e^2·nm^2].
        Rows = upper band subbands, cols = lower band subbands, both in current subband order.
        Looks up both (nn_i_upper, nn_j_lower) and (nn_j_lower, nn_i_upper) orderings.

        Parameters
        ----------
        upper : str
            Name of the upper band. Default 'CB'. For 1-band use e.g. 'Gamma'.
        lower : str
            Name of the lower band. Default 'VB'. For 1-band use e.g. 'HH' or 'LH'.
        """
        if upper not in self.bands or lower not in self.bands:
            raise ValueError(f"Both '{upper}' and '{lower}' bands must exist. Available: {list(self.bands.keys())}")
        if polarization not in self.interband_dipole_moments:
            raise KeyError(f"No interband dipole moments for polarization '{polarization}'. "
                           f"Available: {list(self.interband_dipole_moments.keys())}")
        cb_subbands = self.bands[upper].subbands
        vb_subbands = self.bands[lower].subbands
        mat = np.zeros((len(cb_subbands), len(vb_subbands)))
        d = self.interband_dipole_moments[polarization]
        for row, si in enumerate(cb_subbands):
            for col, sj in enumerate(vb_subbands):
                val = d.get((si.nn_index, sj.nn_index),
                            d.get((sj.nn_index, si.nn_index), 0.0))
                mat[row, col] = val
        return mat

    def add_absorption_spectrum(self, photon_energy: np.ndarray, absorption: np.ndarray, polarization: str = None):
        self.optical_absorption.add_spectrum(photon_energy, absorption, polarization)
    
    # def plot_probabilities(self, band_name:str):
    def plot_all_bands(self, scale=0.05, fontsizebase=18,fontsizetitle=22,title_diff = None, colors=None, show=True,add_title=True,add_legend=True):
        """
        Plot all band structures and their subband wavefunctions in one figure.

        Parameters
        ----------
        scale : float
            Scaling factor for probability amplitudes.
        colors : dict
            Optional dict like {'CB': 'blue', 'VB': 'red'}.
        show : bool
            Whether to call plt.show().
        """
        colors = colors or {}
        fig, ax = plt.subplots(figsize=(10, 6))

        for name, band in self.bands.items():
            color = colors.get(name) or _DEFAULT_BAND_COLORS.get(name, 'gray')
            band.plot_band(scale=scale, color=color, ax=ax, show=False,show_legend=add_legend)

        if add_title:
            if title_diff is not None:
                ax.set_title(title_diff)
            else:
                ax.set_title(f"Band Structures — {self.simname}")

        if add_legend:
            ax.legend()
            ax.legend(fontsize=fontsizebase)

        ax.xaxis.get_label().set_fontsize(fontsizebase)
        ax.yaxis.get_label().set_fontsize(fontsizebase)

        ax.title.set_fontsize(fontsizetitle)
        ax.tick_params(axis='both', labelsize=fontsizebase)
        if show:
            plt.tight_layout()
            plt.show()
        return fig, ax

    def plot_density(self, band_names, density, log_scale=False,
                     ax=None, show=True, fontsizebase=18, fontsizetitle=22,
                     title_diff=None):
        """
        Plot band edge(s) on the left axis and a named density on the right axis.

        Parameters
        ----------
        band_names : str or list of str
            Band name(s) whose edges to plot on the left axis.
        density : str
            Key in self.densities (e.g. 'Electron_density', 'n-Si-in-AlAs').
        log_scale : bool
            If True, right axis uses log scale.
        """
        if isinstance(band_names, str):
            band_names = [band_names]

        if density not in self.densities:
            raise ValueError(f"Density '{density}' not found. Available: {list(self.densities.keys())}")
        d = self.densities[density]

        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))

        # Left axis: band edges
        for name in band_names:
            if name not in self.bands:
                raise ValueError(f"Band '{name}' not found. Available: {list(self.bands.keys())}")
            for edge in self.bands[name].bandedges:
                edge.plot(ax=ax, show=False, fontsizebase=fontsizebase)

        ax.tick_params(axis='both', labelsize=fontsizebase)
        ax.legend(fontsize=fontsizebase)

        # Right axis: density
        ax2 = ax.twinx()
        color = _KNOWN_DENSITY_COLORS.get(d.density_name, 'gray')
        ax2.plot(d.x, d.density, color=color, lw=2.0, alpha=0.7)
        ax2.set_ylabel(f"{d.density_name} [{d.units}]", fontsize=fontsizebase, color=color)
        ax2.tick_params(axis='y', labelsize=fontsizebase, labelcolor=color)
        if log_scale:
            ax2.set_yscale('log')

        ax.set_title(title_diff or f"Band edges & {d.density_name} — {self.simname}", fontsize=fontsizetitle)

        if show:
            plt.tight_layout()
            plt.show()

        return ax, ax2

    def __repr__(self):
        lines = [f"<SimOut simname={self.simname}>"]
        if not self.bands:
            lines.append("  No bands available")
        for name,band in self.bands.items():
            E=band.get_energies()
            if len(E):
                lines.append(f" {name}:{len(E)} subbands, Energies = {E.round(5).tolist()}")
            else:
                lines.append(f" {name}: No subbands available")
        return "\n".join(lines)
    