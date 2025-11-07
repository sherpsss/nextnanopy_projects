import os
import numpy as np
import nextnanopy as nn
from .simstructs import SimOut

def build_output(outpath, quantum_region, quantum_band, quantum_band_interactions, bias, VB_cutoff, well_w):
    """
    Parses Nextnano output files and organizes results into SimOut -> BandStructure -> Eigenstate hierarchy.
    """

    # Create the top-level simulation container
    sim = SimOut(simname=f"sweep_w{well_w}_bias{bias}")

    # Load band edge data
    band_edge = nn.DataFile(os.path.join(outpath, bias, 'bandedges.dat'), 'nextnano++')
    Gamma = band_edge.variables['Gamma']
    HH = band_edge.variables['HH']
    LH = band_edge.variables['LH']
    sim.electron_Fermi_level = np.mean(band_edge.variables['electron_Fermi_level'].value)
    sim.hole_Fermi_level = np.mean(band_edge.variables['hole_Fermi_level'].value)

    # Define paths for quantum simulation results
    quantum_sims_path = os.path.join(outpath, bias, quantum_region)
    energy_spectrum = nn.DataFile(os.path.join(quantum_sims_path, quantum_band, 'energy_spectrum_k00000.dat'), 'nextnano++')
    probabilities = nn.DataFile(os.path.join(quantum_sims_path, quantum_band, 'probabilities_shift_k00000.dat'), 'nextnano++')
    transition_energies = nn.DataFile(os.path.join(quantum_sims_path, quantum_band_interactions, 'transition_energies_k00000.txt'), 'nextnano++')

    # Dictionary to store energy classification: {psi_index: ('CB' or 'VB', energy_value)}
    non_degen_E = {}

    # Identify non-degenerate eigenenergies and classify by band type (VB vs CB)
    for psi in probabilities.variables:
        parts = psi.name.split('_')
        if len(parts) > 1 and parts[-1].isdigit():
            num = int(parts[-1])
            if num % 2 == 1 and 'E' in psi.name:  # odd-numbered, energy-like variable
                energy = np.mean(psi.value)
                if energy < VB_cutoff:
                    non_degen_E[num] = ('VB', energy)
                else:
                    non_degen_E[num] = ('CB', energy)

    # Loop again to collect corresponding probability distributions
    for psi in probabilities.variables:
        if 'Psi^2' in psi.name:
            idx = int(psi.name.split('_')[-1])
            if idx in non_degen_E:
                band_name, energy = non_degen_E[idx]
                prob_dist = psi.value
                sim.add_subband(band_name=band_name, index=idx, energy=energy, prob_dist=prob_dist)

    # Attach auxiliary fields if desired
    sim.band_edge = {'Gamma': Gamma, 'HH': HH, 'LH': LH}

    return sim
