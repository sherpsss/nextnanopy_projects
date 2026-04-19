import os
import numpy as np
import matplotlib.pyplot as plt
import nextnanopy as nn
import re
from .simstructs import SimOut, BandStructure, Eigenstate, BandEdge, OpticalAbsorption, Spectrum, Density

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

# 1-band model: bands classified as CB or VB by directory name
_ONEBAND_CB_BANDS = ['Gamma', 'X', 'L']
_ONEBAND_VB_BANDS = ['HH', 'LH', 'SO']


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


def build_output(outpath, quantum_region, bias, model='kp8', VB_cutoff=None):
    """
    Parses Nextnano output files and organizes results into SimOut -> BandStructure -> Eigenstate hierarchy.

    Parameters
    ----------
    outpath : str
        Path to the simulation output directory. The last folder name is used as the SimOut simname.
    quantum_region : str
        Relative path to the quantum region folder (e.g. r'Quantum\\quantum_region').
    bias : str
        Bias subfolder name (e.g. 'bias_00000').
    model : str
        'kp8'   : 8-band k·p model — spin-degenerate at k=0, every other eigenvalue skipped.
                  CB/VB split by VB_cutoff. Produces sim.bands['CB'] and sim.bands['VB'].
                  Requires VB_cutoff.
        'kp6'   : 6-band k·p VB-only model. kp6/ folder → all states → sim.bands['VB'] (spin-
                  degenerate skip applied). Gamma/ folder → sim.bands['CB'] (no degeneracy).
                  No VB_cutoff needed.
        '1band' : Single-band model. Per-band directories are auto-discovered from:
                  CB bands: Gamma, X, L.  VB bands: HH, LH, SO.
                  No spin degeneracy. Produces one BandStructure per discovered band
                  (e.g. sim.bands['Gamma'], sim.bands['HH'], sim.bands['LH']).
    VB_cutoff : float, optional
        Energy threshold (eV) separating CB from VB states. Required for model='kp8'.
    """
    if model == 'kp8' and VB_cutoff is None:
        raise ValueError("model='kp8' requires VB_cutoff.")

    # Create the top-level simulation container
    simname = os.path.basename(os.path.normpath(outpath))
    sim = SimOut(simname=simname)

    # Load input variables
    vars_path = os.path.join(outpath, 'variables_input.txt')
    if os.path.isfile(vars_path):
        with open(vars_path) as vf:
            for line in vf:
                line = line.strip()
                if not line or line.startswith('#') or '=' not in line:
                    continue
                name, _, raw = line.partition('=')
                name = name.strip().lstrip('$')
                raw = raw.strip().strip('"')
                try:
                    val = int(raw)
                except ValueError:
                    try:
                        val = float(raw)
                    except ValueError:
                        val = raw
                sim.variables[name] = val

    # Load band edge data (shared across all models)
    band_edge = nn.DataFile(os.path.join(outpath, bias, 'bandedges.dat'), 'nextnano++')
    x_edge = band_edge.coords['x'].value
    quantum_sims_path = os.path.join(outpath, bias, quantum_region)

    sim.electron_Fermi_level = np.mean(band_edge.variables['electron_Fermi_level'].value)
    sim.hole_Fermi_level = np.mean(band_edge.variables['hole_Fermi_level'].value)

    # Classical band edges from bandedges.dat — available in all models
    gamma_edge = BandEdge('Gamma', band_edge.variables['Gamma'].value, x_edge)
    hh_edge = BandEdge('HH', band_edge.variables['HH'].value, x_edge)
    lh_edge = BandEdge('LH', band_edge.variables['LH'].value, x_edge)
    so_edge = BandEdge('SO', band_edge.variables['SO'].value, x_edge)

    # ------------------------------------------------------------------
    # MODEL: kp8 — 8-band k·p, CB/VB split by VB_cutoff
    # ------------------------------------------------------------------
    if model == 'kp8':
        probabilities = nn.DataFile(os.path.join(quantum_sims_path, 'kp8', 'probabilities_k00000.dat'), 'nextnano++')
        energy_spectrum = nn.DataFile(os.path.join(quantum_sims_path, 'kp8', 'energy_spectrum_k00000.dat'), 'nextnano++')
        x_prob = probabilities.coords['x'].value

        CB = BandStructure('CB', x=x_prob)
        VB = BandStructure('VB', x=x_prob)
        CB.add_bandedge(gamma_edge)
        VB.add_bandedge(hh_edge)
        VB.add_bandedge(lh_edge)
        VB.add_bandedge(so_edge)

        energies = energy_spectrum.variables['Energy'].value
        # kp: spin-degenerate pairs at k=0 — keep one from each pair
        selected_inds = list(range(1, len(energies), 2))

        non_degen_E = {}
        for ind in selected_inds:
            Eeig = energies[ind - 1]
            non_degen_E[ind] = ('VB', Eeig) if Eeig < VB_cutoff else ('CB', Eeig)

        for psi in probabilities.variables:
            idx = int(psi.name.split('_')[-1])
            if idx in non_degen_E:
                band_label, energy = non_degen_E[idx]
                target = CB if band_label == 'CB' else VB
                target.add_subband(Eigenstate(idx, energy, psi.value))

        interactions_path = os.path.join(quantum_sims_path, 'kp8_kp8')
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

        sim.add_band(CB)
        sim.add_band(VB)

    # ------------------------------------------------------------------
    # MODEL: 1band — one BandStructure per discovered band directory
    # ------------------------------------------------------------------
    elif model == '1band':
        # Classical band edges available for these bands in bandedges.dat
        _edge_map = {'Gamma': gamma_edge, 'HH': hh_edge, 'LH': lh_edge, 'SO': so_edge}

        for band_name in _ONEBAND_CB_BANDS + _ONEBAND_VB_BANDS:
            band_dir = os.path.join(quantum_sims_path, band_name)
            if not os.path.isdir(band_dir):
                continue

            energy_spectrum = nn.DataFile(os.path.join(band_dir, 'energy_spectrum_k00000.dat'), 'nextnano++')
            probabilities = nn.DataFile(os.path.join(band_dir, 'probabilities_k00000.dat'), 'nextnano++')
            x_prob = probabilities.coords['x'].value

            band_struct = BandStructure(band_name, x=x_prob)
            if band_name in _edge_map:
                band_struct.add_bandedge(_edge_map[band_name])

            energies = energy_spectrum.variables['Energy'].value
            # Build local index→energy map (1-based, no degeneracy skip)
            local_map = {idx + 1: E for idx, E in enumerate(energies)}

            for psi in probabilities.variables:
                local_idx = int(psi.name.split('_')[-1])
                if local_idx in local_map:
                    band_struct.add_subband(Eigenstate(local_idx, local_map[local_idx], psi.value))

            # Load per-band dipoles from e.g. HH_HH/ folder
            interactions_dir = os.path.join(quantum_sims_path, f"{band_name}_{band_name}")
            if os.path.isdir(interactions_dir):
                # Pass local_map as non_degen_E — all states share the same band label
                local_non_degen = {idx: (band_name, E) for idx, E in local_map.items()}
                isb_dipoles, _ = _load_dipole_moments(interactions_dir, local_non_degen)
                for polarization, d in isb_dipoles.items():
                    if d:
                        band_struct.add_dipole_moments(polarization, d)

            sim.add_band(band_struct)

    # ------------------------------------------------------------------
    # MODEL: kp6 — 6-band VB-only k·p; CB loaded separately from Gamma/
    # ------------------------------------------------------------------
    elif model == 'kp6':
        kp6_dir = os.path.join(quantum_sims_path, 'kp6')
        prob_vb = nn.DataFile(os.path.join(kp6_dir, 'probabilities_k00000.dat'), 'nextnano++')
        espec_vb = nn.DataFile(os.path.join(kp6_dir, 'energy_spectrum_k00000.dat'), 'nextnano++')
        x_prob = prob_vb.coords['x'].value

        VB = BandStructure('VB', x=x_prob)
        VB.add_bandedge(hh_edge)
        VB.add_bandedge(lh_edge)
        VB.add_bandedge(so_edge)

        energies_vb = espec_vb.variables['Energy'].value
        # kp6: spin-degenerate at k=0 — keep one from each pair
        selected_inds_vb = list(range(1, len(energies_vb), 2))
        non_degen_vb = {ind: ('VB', energies_vb[ind - 1]) for ind in selected_inds_vb}

        for psi in prob_vb.variables:
            idx = int(psi.name.split('_')[-1])
            if idx in non_degen_vb:
                _, energy = non_degen_vb[idx]
                VB.add_subband(Eigenstate(idx, energy, psi.value))

        vb_isb_dipoles, _ = _load_dipole_moments(
            os.path.join(quantum_sims_path, 'kp6_kp6'), non_degen_vb)
        for polarization, d in vb_isb_dipoles.items():
            if d:
                VB.add_dipole_moments(polarization, d)

        # CB: Gamma folder, no spin degeneracy
        CB = BandStructure('CB', x=x_prob)
        CB.add_bandedge(gamma_edge)

        gamma_dir = os.path.join(quantum_sims_path, 'Gamma')
        if os.path.isdir(gamma_dir):
            prob_cb = nn.DataFile(os.path.join(gamma_dir, 'probabilities_k00000.dat'), 'nextnano++')
            espec_cb = nn.DataFile(os.path.join(gamma_dir, 'energy_spectrum_k00000.dat'), 'nextnano++')
            energies_cb = espec_cb.variables['Energy'].value
            local_map_cb = {idx + 1: E for idx, E in enumerate(energies_cb)}

            for psi in prob_cb.variables:
                local_idx = int(psi.name.split('_')[-1])
                if local_idx in local_map_cb:
                    CB.add_subband(Eigenstate(local_idx, local_map_cb[local_idx], psi.value))

            gamma_gamma_dir = os.path.join(quantum_sims_path, 'Gamma_Gamma')
            if os.path.isdir(gamma_gamma_dir):
                local_non_degen_cb = {idx: ('CB', E) for idx, E in local_map_cb.items()}
                cb_isb_dipoles, _ = _load_dipole_moments(gamma_gamma_dir, local_non_degen_cb)
                for polarization, d in cb_isb_dipoles.items():
                    if d:
                        CB.add_dipole_moments(polarization, d)

        sim.add_band(CB)
        sim.add_band(VB)

    else:
        raise ValueError(f"Unknown model '{model}'. Use 'kp', 'kp6', or '1band'.")

    # --- Optical absorption spectra (shared) ---
    optics_path = os.path.join(outpath, bias, "OpticsQuantum", "quantum_region")
    if os.path.isdir(optics_path):
        absorption_files = [
            f for f in os.listdir(optics_path)
            if "absorption_coeff_spectrum" in f.lower() and "ev" in f.lower() and f.lower().endswith(".dat")
        ]
        if absorption_files:
            sim.optical_absorption = build_optical_absorption(optics_path, absorption_files)

    # --- Densities (optional) ---
    density_files = [
        (os.path.join(outpath, bias, 'density_electron.dat'),          False),
        (os.path.join(outpath, bias, 'density_hole.dat'),              False),
        (os.path.join(outpath, bias, 'density_donor_ionized.dat'),     False),
        (os.path.join(outpath, bias, 'density_acceptor_ionized.dat'),  False),
        (os.path.join(outpath, 'Structure', 'density_donor.dat'),      False),
        (os.path.join(outpath, 'Structure', 'density_acceptor.dat'),   False),
    ]
    for fpath, _ in density_files:
        if not os.path.isfile(fpath):
            continue
        with open(fpath) as f:
            headers = f.readline().split()
        # strip units from each header token, e.g. "Electron_density[1e18_cm^-3]" -> "Electron_density"
        col_names = [re.sub(r'\[.*?\]', '', h) for h in headers]
        raw = np.loadtxt(fpath, skiprows=1)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)
        x = raw[:, 0]
        # col_names[0] is the x coordinate; remaining columns are densities
        for col_idx, name in enumerate(col_names[1:], start=1):
            sim.densities[name] = Density(name, x, raw[:, col_idx])

    return sim
