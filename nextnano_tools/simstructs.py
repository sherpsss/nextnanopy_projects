import numpy as np
import matplotlib.pyplot as plt 

class Eigenstate:
    def __init__(self, index:int,energy:float, prob_dist:np.ndarray):
        #for 1 eigenenstate of whatever band is relevant

        self.index = index
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
    
class BandStructure:
    def __init__(self, name:str,subbands = None, bandedges = None,x=None):
        #stores eigenstates in CB or VB subband
        self.name = name
        self.subbands = [] if subbands is None else subbands
        self.bandedges = [] if bandedges is None else bandedges
        self.x = x  # spatial axis in nm
    
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

    # def remove_bandedge(self, *edge_names):
    #     for n in edge_names:
    #         if n in self.bandedges:
    #             self.bandedges.pop(n)
    
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

    def plot_band(self, scale=0.05, fontsizebase = 18,fontsizetitle = 22,color=None, ax=None, show=True, show_legend =True, show_grid=False,title_diff=None,normalize_y=False):
        """
        Plot band edges and subband probability distributions.

        Parameters
        ----------
        scale : float
            Vertical scale factor for normalized probability amplitudes.
        color : str
            Optional color for subband wavefunctions.
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

        color = color or ('dodgerblue' if self.name == 'CB' else 'crimson')

        # --------------------------------------------
        # Compute energy offset if normalize_y=True
        # --------------------------------------------
        if normalize_y:
            # collect all band edge energies for this band
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
            y = edge.energy - E_min
            ax.plot(edge.x, y, label=f"{edge.name}")

        # --------------------------------------------
        # Plot each eigenstate energy + wavefunction
        # --------------------------------------------
        for subband in self.subbands:
            E = subband.energy - E_min

            # plot energy as dashed horizontal line
            ax.plot(self.x, np.full_like(self.x, E), ls='--',
                    label=f'{self.name} {subband.index}')

            # probability distribution
            psi2 = subband.probab_dist
            psi2_norm = psi2 / np.max(np.abs(psi2)) if np.max(np.abs(psi2)) != 0 else psi2
            ax.plot(self.x, psi2_norm * scale + E, color=color, lw=1.2)

        if normalize_y:
            ax.set_ylabel("Energy relative to Band Edge (eV)")
        else:
            ax.set_ylabel("Energy (eV)")

        ax.set_xlabel("Growth direction (nm)")
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

        if show:
            plt.tight_layout()
            plt.show()
        
        if show_grid:
            plt.grid()
            
        return ax

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


    
class SimOut:
    def __init__(self, simname:str):
        #represents 1 full sim result
        self.simname = simname
        self.electron_Fermi_level = None
        self.hole_Fermi_level = None
        self.bands ={}
        self.optical_absorption = OpticalAbsorption() #only included in some sims
    
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
    
    def calc_interband_transitions(self):
        """
        Compute interband transition energies between VB and CB.

        Returns
        -------
        transitions : 2D numpy array
            transitions[i, j] = CB_i - VB_j
        """
        if 'CB' not in self.bands or 'VB' not in self.bands:
            raise ValueError("Both CB and VB bands must exist for interband transitions.")

        VB_mesh, CB_mesh = np.meshgrid(self.bands['VB'].get_energies(), self.bands['CB'].get_energies())
        return CB_mesh - VB_mesh

    
    def add_absorption_spectrum(self, photon_energy: np.ndarray, absorption: np.ndarray, polarization: str = None):
        self.optical_absorption.add_spectrum(photon_energy, absorption, polarization)
    
    # def plot_probabilities(self, band_name:str):
    def plot_all_bands(self, scale=0.05, fontsizebase=18,fontsizetitle=22,title_diff = None, colors=None, show=True):
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
        colors = colors or {'CB': 'dodgerblue', 'VB': 'crimson'}
        fig, ax = plt.subplots(figsize=(10, 6))

        for name, band in self.bands.items():
            band.plot_band(scale=scale, color=colors.get(name, None), ax=ax, show=False)

        if title_diff is not None:
            ax.set_title(title_diff)
        else:
            ax.set_title(f"Band Structures — {self.simname}")
        ax.legend()

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
    