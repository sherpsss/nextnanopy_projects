import os
import numpy as np
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
                    

def build_output(outpath, quantum_region, quantum_band, quantum_band_interactions, bias, VB_cutoff, well_w):
    """
    Parses Nextnano output files and organizes results into SimOut -> BandStructure -> Eigenstate hierarchy.
    """

    # Create the top-level simulation container
    sim = SimOut(simname=f"sweep_w{well_w}")

    # Load band edge data
    band_edge = nn.DataFile(os.path.join(outpath, bias, 'bandedges.dat'), 'nextnano++')
    quantum_sims_path = os.path.join(outpath, bias, quantum_region)
    energy_spectrum = nn.DataFile(os.path.join(quantum_sims_path, quantum_band, 'energy_spectrum_k00000.dat'), 'nextnano++')
    probabilities = nn.DataFile(os.path.join(quantum_sims_path, quantum_band, 'probabilities_k00000.dat'), 'nextnano++')
    transition_energies = nn.DataFile(os.path.join(quantum_sims_path, quantum_band_interactions, 'transition_energies_k00000.txt'), 'nextnano++')

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

    #skip every other eigenvalue to get non-degenerate states
    degen_energies = energy_spectrum.variables['Energy'].value
    non_degen_inds = list(range(1, len(degen_energies), 2)) #adding 1 to get the ususal way subbands are indexed
    for non_degen_ind in non_degen_inds:
        Eeig = degen_energies[non_degen_ind]
        if Eeig < VB_cutoff:
            non_degen_E[non_degen_ind] = ('VB', Eeig)
        else:
            non_degen_E[non_degen_ind] = ('CB', Eeig)

    # Loop again to collect corresponding probability distributions
    for psi in probabilities.variables:
        idx = int(psi.name.split('_')[-1])
        if idx in non_degen_E:
            band_name, energy = non_degen_E[idx]
            if band_name == 'CB':
                CB.add_subband(Eigenstate(idx, energy, psi.value))
            else:
                VB.add_subband(Eigenstate(idx, energy, psi.value))

    # Add band structures to simulation output
    sim.add_band(CB)
    sim.add_band(VB)

        # --- Optical absorption spectra ---
    optics_path = os.path.join(outpath, bias, "OpticsQuantum", "quantum_region")

    if os.path.isdir(optics_path):
        absorption_files = [
            f for f in os.listdir(optics_path)
            if "absorption_coeff_spectrum" in f.lower() and "eV" in f.lower() and f.endswith(".dat")
        ]


        if absorption_files:
            absorption_populated = build_optical_absorption(optics_path,absorption_files)

            sim.optical_absorption = absorption_populated

    return sim
