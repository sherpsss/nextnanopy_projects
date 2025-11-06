import numpy as np

class Eigenstate:
    def __init__(self, index:int,energy:float, prob_dist:np.ndarray):
        #for 1 eigenenstate of whatever band is relevant

        self.index = index
        self.energy = energy
        self.probab_dist = prob_dist
    
    def __repr__(self):
        return f"<Eigenstate #index={self.index}, energy={self.energy} eV>"
    
class BandStructure:
    def __init__(self, name:str):
        #stores eigenstates in CB or VB subband
        self.name = name
        self.subbands = []
    
    def add_subband(self, eigenstate:Eigenstate):
        if not isinstance(eigenstate, Eigenstate):
            raise TypeError("eigenstate must be an instance of Eigenstate class")
        self.subbands.append(eigenstate)
    
    def get_energies(self):
        return np.array([subband.energy for subband in self.subbands])
    
    def sort_subbands(self,decreasing=True):
        self.subbands.sort(key=lambda x: x.energy, reverse=decreasing)
        for new_index, subband in enumerate(self.subbands):
            subband.index = new_index   # Update index to reflect new order
    
    def remove_subband(self, subband=None,index=None):
        if subband is not None:
            self.subbands = [s for s in self.subbands if s is not subband]
        elif index is not None:
            self.subbands = [s for s in self.subbands if s.index != index]
        else:
            raise ValueError("must provide subband object or index to remove")
        
        for new_index, subband in enumerate(self.subbands):
            subband.index = new_index

    def __repr__(self):
        n = len(self.subbands)
        if n==0:
            return f"<BandStructure name={self.name}, no subbands>"
        energies = ", ".join([f"{subband.energy:.4f} eV" for subband in self.subbands])
        return f"<BandStructure name={self.name}, {n} subbands: [{energies}]>"
    
class SimOut:
    def __init__(self, simname:str):
        #represents 1 full sim result
        self.simname = simname
        self.bands ={}
    
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