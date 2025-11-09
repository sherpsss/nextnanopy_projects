import os
import numpy as np
import nextnanopy as nn
from .simstructs import SimOut, BandStructure,Eigenstate,BandEdge

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

    # # Identify non-degenerate eigenenergies and classify by band type (VB vs CB)
    # for psi in probabilities.variables:
    #     parts = psi.name.split('_')
    #     if len(parts) > 1 and parts[-1].isdigit():
    #         num = int(parts[-1])
    #         if num % 2 == 1 and 'E' in psi.name:  # odd-numbered, energy-like variable
    #             energy = np.mean(psi.value)
    #             if energy < VB_cutoff:
    #                 non_degen_E[num] = ('VB', energy)
    #             else:
    #                 non_degen_E[num] = ('CB', energy)

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

    return sim
