import os
import numpy as np
import matplotlib.pyplot as plt
import nextnanopy as nn
from .simstructs import SimOut, BandStructure,Eigenstate,BandEdge,OpticalAbsorption,Spectrum

def build_optical_absorption(optics_path,absorption_files): #built to assume only eV input for now
    absorption_obj = OpticalAbsorption()
    for f in absorption_files:
        filepath = os.path.join(optics_path, f)
            # try:
        data = nn.DataFile(filepath, "nextnano++")

        photonin = data.coords['Energy'].value
        alpha = data.variables['Absorption_Coefficient'].value
        # if x_var is None or alpha_var is None:
        #     continue

        # Infer polarization and axis
        fname = f.lower()
        pol = "TE" if "te" in fname else "TM" if "tm" in fname else "unknown"
        axis = None
        if "_x_" in fname:
            axis = "x"
        elif "_y_" in fname:
            axis = "y"
        elif "_z_" in fname:
            axis = "z"

        # Build Spectrum object
        spectrum = Spectrum(
            x=photonin,
            y=alpha,
            x_unit='eV',
            polarization=pol,
            axis=axis
        )

        absorption_obj.add_spectrum(spectrum)

            # except Exception as e:
                # print(f"Warning: Could not load absorption file '{f}': {e}")
    return absorption_obj
                    

_EV_TO_WAVENUM = 8065.544  # cm⁻¹ per eV


def plot_absorption_with_dipoles(band, absorption_spectrum, polarization,
                                  upward_only=True, label_transitions=False,
                                  numin=None, numax=None,
                                  ax=None, show=True, fontsizebase=18, fontsizetitle=22,
                                  title_diff=None):
    """
    Dual-axis plot: absorption spectrum (left axis) and ISB dipole |d|^2 stems (right axis).
    X-axis is in wavenumbers [cm⁻¹].

    Parameters
    ----------
    band : BandStructure
    absorption_spectrum : Spectrum
        Must have x in eV (x_unit='eV') or already in cm⁻¹.
    polarization : str
        Dipole polarization key (e.g. 'TM_z').

    Returns
    -------
    ax, ax2 : left and right Axes
    """
    dE_eV, d_vals, labels = band.get_dipole_vs_energy(polarization, upward_only=upward_only)
    dE_wavenum = dE_eV * _EV_TO_WAVENUM

    spec_x = absorption_spectrum.x
    if absorption_spectrum.x_unit.lower() == 'ev':
        spec_x_wavenum = spec_x * _EV_TO_WAVENUM
    else:
        spec_x_wavenum = spec_x  # assume already cm⁻¹

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    # Left axis: absorption spectrum
    pol_label = f"{absorption_spectrum.polarization}-{absorption_spectrum.axis}" if absorption_spectrum.axis else absorption_spectrum.polarization
    ax.plot(spec_x_wavenum, absorption_spectrum.y, color='steelblue', label=pol_label)
    ax.set_xlabel("Wavenumber (cm⁻¹)")
    ax.set_ylabel(absorption_spectrum.y_label, color='steelblue')
    ax.tick_params(axis='y', labelcolor='steelblue')

    # Right axis: dipole strengths
    ax2 = ax.twinx()
    markerline, _, _ = ax2.stem(dE_wavenum, d_vals, linefmt='C1-', markerfmt='C1o', basefmt=' ')
    markerline.set_markersize(5)
    ax2.set_ylabel(r"$|d_{ij}|^2$ [e²·nm²]", color='C1')
    ax2.tick_params(axis='y', labelcolor='C1')

    if label_transitions:
        for dE, d, lbl in zip(dE_wavenum, d_vals, labels):
            ax2.annotate(lbl, (dE, d), textcoords="offset points",
                         xytext=(0, 4), ha='center', fontsize=fontsizebase - 6)

    ax.set_title(title_diff or f"{band.name} absorption & dipole strengths — {polarization}")
    ax.xaxis.get_label().set_fontsize(fontsizebase)
    ax.yaxis.get_label().set_fontsize(fontsizebase)
    ax2.yaxis.get_label().set_fontsize(fontsizebase)
    ax.title.set_fontsize(fontsizetitle)
    ax.tick_params(axis='both', labelsize=fontsizebase)
    ax2.tick_params(axis='y', labelsize=fontsizebase)

    if numin is not None or numax is not None:
        ax.set_xlim(numin, numax)
        # Autoscale both y-axes to the visible x range
        xlo = numin if numin is not None else spec_x_wavenum.min()
        xhi = numax if numax is not None else spec_x_wavenum.max()

        mask_spec = (spec_x_wavenum >= xlo) & (spec_x_wavenum <= xhi)
        if mask_spec.any():
            ylo = absorption_spectrum.y[mask_spec].min()
            yhi = absorption_spectrum.y[mask_spec].max()
            pad = 0.05 * abs(yhi - ylo) if yhi != ylo else 0.1 * abs(yhi)
            ax.set_ylim(ylo - pad, yhi + pad)

        mask_dip = (dE_wavenum >= xlo) & (dE_wavenum <= xhi)
        if mask_dip.any():
            ax2.set_ylim(0, d_vals[mask_dip].max() * 1.2)

    if show:
        plt.tight_layout()
        plt.show()

    return ax, ax2


def plot_ldos(band: BandStructure, ldos: np.ndarray, eVbias_values: np.ndarray, ax=None, show=True,
              cmap='inferno', fontsizebase=18, fontsizetitle=22,
              title_diff=None, colorbar_label='LDOS [a.u.]', **kwargs):
    """
    Plot a heatmap of a pre-computed LDOS matrix.

    x-axis: growth direction (nm)
    y-axis: eVbias (eV)
    color:  LDOS value

    Parameters
    ----------
    band : BandStructure
        Provides the spatial axis (band.x).
    ldos : np.ndarray, shape (len(eVbias_values), len(band.x))
        Output of band.calc_ldos.
    eVbias_values : array-like
        Bias energy values in eV corresponding to rows of ldos.
    ax : matplotlib.axes.Axes, optional
    show : bool
    cmap : str
    title_diff : str, optional
        Override the default title.
    **kwargs
        Passed to ax.pcolormesh.

    Returns
    -------
    ax : matplotlib.axes.Axes
    """
    eVbias_values = np.asarray(eVbias_values)

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    else:
        fig = ax.get_figure()

    mesh = ax.pcolormesh(band.x, eVbias_values, ldos, cmap=cmap, shading='auto', **kwargs)
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label(colorbar_label, fontsize=fontsizebase)
    cbar.ax.tick_params(labelsize=fontsizebase)

    ax.set_xlabel("Growth direction [nm]", fontsize=fontsizebase)
    ax.set_ylabel("eV [eV]", fontsize=fontsizebase)
    ax.set_title(title_diff or f"LDOS — {band.name}", fontsize=fontsizetitle)
    ax.tick_params(axis='both', labelsize=fontsizebase)

    if show:
        plt.tight_layout()
        plt.show()

    return ax


def _load_dipole_moments(interactions_path, non_degen_E):
    """
    Parse dipole moment matrix element .txt files from the quantum interactions folder.
    Only pairs where both indices are in non_degen_E (kept states) are included.

    Returns
    -------
    isb_dipoles : dict  {polarization: {(nn_i, nn_j): |d|^2 [e^2·nm^2]}}
        Intraband pairs (both states in the same band).
    interband_dipoles : dict  {polarization: {(nn_i, nn_j): |d|^2 [e^2·nm^2]}}
        Interband pairs (states in different bands).
    """
    if not os.path.isdir(interactions_path):
        return {}, {}

    isb_dipoles = {}
    interband_dipoles = {}

    for fname in os.listdir(interactions_path):
        if 'dipole_moment_matrix_elements_k00000' not in fname.lower() or not fname.endswith('.txt'):
            continue

        polarization = fname.replace('dipole_moment_matrix_elements_k00000_', '').replace('.txt', '')
        filepath = os.path.join(interactions_path, fname)

        try:
            data = np.loadtxt(filepath, skiprows=1)
        except Exception as e:
            print(f"Warning: could not load dipole file '{fname}': {e}")
            continue

        if data.ndim == 1:
            data = data.reshape(1, -1)

        isb_dipoles[polarization] = {}
        interband_dipoles[polarization] = {}

        for row in data:
            i, j = int(row[0]), int(row[1])
            abs2 = float(row[3])

            if i not in non_degen_E or j not in non_degen_E:
                continue

            band_i = non_degen_E[i][0]
            band_j = non_degen_E[j][0]

            if band_i == band_j:
                isb_dipoles[polarization][(i, j)] = abs2
            else:
                interband_dipoles[polarization][(i, j)] = abs2

    return isb_dipoles, interband_dipoles


def build_output(outpath, quantum_region, quantum_band, quantum_band_interactions, bias, VB_cutoff, well_w, model='kp'):
    """
    Parses Nextnano output files and organizes results into SimOut -> BandStructure -> Eigenstate hierarchy.

    Parameters
    ----------
    model : str
        'kp'    : k·p model — eigenvalues are doubly degenerate at k=0 (spin), so every
                  other eigenvalue is skipped.
        'gamma' : Gamma-band model — no spin degeneracy; all eigenvalues are used directly.
    """

    # Create the top-level simulation container
    sim = SimOut(simname=f"sweep_w{well_w}")

    # Load band edge data
    band_edge = nn.DataFile(os.path.join(outpath, bias, 'bandedges.dat'), 'nextnano++')
    quantum_sims_path = os.path.join(outpath, bias, quantum_region)
    energy_spectrum = nn.DataFile(os.path.join(quantum_sims_path, quantum_band, 'energy_spectrum_k00000.dat'), 'nextnano++')
    probabilities = nn.DataFile(os.path.join(quantum_sims_path, quantum_band, 'probabilities_k00000.dat'), 'nextnano++')
    # transition_energies = nn.DataFile(os.path.join(quantum_sims_path, quantum_band_interactions, 'transition_energies_k00000.txt'), 'nextnano++')

    x_edge = band_edge.coords['x'].value
    x_prob = probabilities.coords['x'].value

    CB = BandStructure('CB',x=x_prob)
    VB = BandStructure('VB',x=x_prob)

    Gamma = BandEdge('Gamma', band_edge.variables['Gamma'].value, x_edge)
    HH = BandEdge('HH', band_edge.variables['HH'].value, x_edge)
    LH = BandEdge('LH', band_edge.variables['LH'].value, x_edge)

    
    CB.add_bandedge(Gamma)
    VB.add_bandedge(HH)
    VB.add_bandedge(LH)

    sim.electron_Fermi_level = np.mean(band_edge.variables['electron_Fermi_level'].value)
    sim.hole_Fermi_level = np.mean(band_edge.variables['hole_Fermi_level'].value)

    # Dictionary to store energy classification: {psi_index: ('CB' or 'VB', energy_value)}
    non_degen_E = {}

    #load in energies and then use those energy indices to load in probabilities so I can have the non-shifted probabilities

    energies = energy_spectrum.variables['Energy'].value
    if model == 'kp':
        # kp: spin-degenerate pairs at k=0 — keep one from each pair (1-based, every other)
        selected_inds = list(range(1, len(energies), 2))
    elif model == 'gamma':
        # Gamma: no spin degeneracy — use all eigenvalues (1-based to match psi file naming)
        selected_inds = list(range(1, len(energies) + 1))
    else:
        raise ValueError(f"Unknown model '{model}'. Use 'kp' or 'gamma'.")

    for ind in selected_inds:
        Eeig = energies[ind - 1]  # energy array is 0-based; psi indices are 1-based
        if Eeig < VB_cutoff:
            non_degen_E[ind] = ('VB', Eeig)
        else:
            non_degen_E[ind] = ('CB', Eeig)

    # Loop again to collect corresponding probability distributions
    for psi in probabilities.variables:
        idx = int(psi.name.split('_')[-1])
        if idx in non_degen_E:
            band_name, energy = non_degen_E[idx]
            if band_name == 'CB':
                CB.add_subband(Eigenstate(idx, energy, psi.value))
            else:
                VB.add_subband(Eigenstate(idx, energy, psi.value))

    # Load dipole moment matrix elements if present in the interactions folder
    interactions_path = os.path.join(quantum_sims_path, quantum_band_interactions)
    isb_dipoles, interband_dipoles = _load_dipole_moments(interactions_path, non_degen_E)

    for polarization, d in isb_dipoles.items():
        cb_d = {(i, j): v for (i, j), v in d.items() if non_degen_E[i][0] == 'CB' and non_degen_E[j][0] == 'CB'}
        vb_d = {(i, j): v for (i, j), v in d.items() if non_degen_E[i][0] == 'VB' and non_degen_E[j][0] == 'VB'}
        if cb_d:
            CB.add_dipole_moments(polarization, cb_d)
        if vb_d:
            VB.add_dipole_moments(polarization, vb_d)

    for polarization, d in interband_dipoles.items():
        if d:
            sim.add_interband_dipole_moments(polarization, d)

    # Add band structures to simulation output
    sim.add_band(CB)
    sim.add_band(VB)

        # --- Optical absorption spectra ---
    optics_path = os.path.join(outpath, bias, "OpticsQuantum", "quantum_region")
    print(optics_path)

    if os.path.isdir(optics_path):
        print("reached optics path")
        all_files = os.listdir(optics_path)
        print(all_files)
        absorption_files =[]
        for f in all_files:
            lower_case = f.lower()
            print(lower_case)
            rules = ["absorption_coeff_spectrum" in lower_case,"ev" in lower_case,lower_case.endswith(".dat")]
            print(rules)
            if all(rules):
                absorption_files.append(f)

        print(absorption_files)

        if absorption_files:
            absorption_populated = build_optical_absorption(optics_path,absorption_files)

            sim.optical_absorption = absorption_populated

    return sim
