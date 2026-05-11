import numpy as np
import torch
import scipy.sparse
import matchms
import matchms.filtering as ms_filters
from massspecgym.data.transforms import SpecTransform, MolTransform
from mol2vec.features import mol2alt_sentence, MolSentence, sentences2vec
from gensim.models import word2vec
from rdkit import Chem


class ChemEmbedSpecTransform(SpecTransform):
    
    def __init__(self, max_mz=700, resolution=0.01, noise_threshold=0.01):
        self.max_mz = max_mz
        self.resolution = resolution
        self.noise_threshold = noise_threshold
        self.n_bins = int(max_mz / resolution)  # 70000

    def matchms_transforms(self, spec: matchms.Spectrum) -> matchms.Spectrum:
        # Noise filter — eliminar picos por debajo del 1% de la intensidad máxima
        mzs = spec.peaks.mz
        ints = spec.peaks.intensities
        
        mask = ints >= self.noise_threshold
        mzs = mzs[mask]
        ints = ints[mask]
        
        # Truncation — eliminar picos por encima de precursor_mz + 0.5
        precursor_mz = spec.metadata["precursor_mz"]
        mask2 = mzs <= precursor_mz + 0.5
        mzs = mzs[mask2]
        ints = ints[mask2]
        
        spec = matchms.Spectrum(mz=mzs, intensities=ints, metadata=spec.metadata)
        return spec

    def matchms_to_torch(self, spec: matchms.Spectrum) -> torch.Tensor:
        mzs = spec.peaks.mz
        
        # Binning binarizado a 0.01 Da → vector de 70000 dimensiones
        bin_edges = np.arange(0, self.max_mz + self.resolution, self.resolution)
        hist, _ = np.histogram(mzs, bins=bin_edges)
        hist = (hist > 0).astype(np.float32)  # binarizar
        
        return torch.from_numpy(hist)  # shape (70000,)


class ChemEmbedMolTransform(MolTransform):
    
    def __init__(self, model_path):
        self.model = word2vec.Word2Vec.load(model_path)
    
    def from_smiles(self, smiles: str) -> np.ndarray:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.zeros(300, dtype=np.float32)

        sentence = MolSentence(mol2alt_sentence(mol, radius=1))
        vec = sentences2vec([sentence], self.model)[0]
        return vec.astype(np.float32)  # numpy array shape (300,)

    # def from_smiles(self, smiles: str):
    #     mol = Chem.MolFromSmiles(smiles)
    #     if mol is None:
    #         return torch.zeros(300, dtype=torch.float32)

    #     sentence = MolSentence(mol2alt_sentence(mol, radius=1))
    #     vec = sentences2vec([sentence], self.model)[0]

    #     return torch.tensor(vec, dtype=torch.float32)  # torch tensor shape (300,)