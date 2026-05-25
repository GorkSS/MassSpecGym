import torch
from rdkit.Chem import AllChem as Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

from massspecgym.models.base import Stage
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel


class RandomRetrieval(RetrievalMassSpecGymModel):

    def step(
        self, batch: dict, stage: Stage = Stage.NONE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Generate random retrieval scores
        scores = torch.rand(batch["candidates_mol"].shape[0]).to(self.device)

        # Random baseline, so we return a dummy loss
        loss = torch.tensor(0.0, requires_grad=True)

        return dict(loss=loss, scores=scores)

    def configure_optimizers(self):
        # No optimizer needed for a random baseline
        return None


class RandomRetrievalGTFormula(RetrievalMassSpecGymModel):
    """Random retrieval restricted to candidates whose molecular formula
    matches the ground-truth query formula.

    Candidates with a matching formula receive a random score in (0, 1];
    non-matching candidates receive 0.
    """

    def step(
        self, batch: dict, stage: Stage = Stage.NONE
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def _formula(smi):
            mol = Chem.MolFromSmiles(smi)
            return CalcMolFormula(mol) if mol is not None else ""

        gt_formulas = [_formula(s) for s in batch["smiles"]]
        cand_formulas = [_formula(s) for s in batch["candidates_smiles"]]

        scores = torch.zeros(len(cand_formulas), device=self.device)
        idx = 0
        for qi, n in enumerate(batch["batch_ptr"].tolist()):
            gt_f = gt_formulas[qi]
            for ci in range(idx, idx + n):
                if cand_formulas[ci] == gt_f:
                    scores[ci] = torch.rand(1, device=self.device).item()
            idx += n

        loss = torch.tensor(0.0, requires_grad=True)
        return dict(loss=loss, scores=scores)

    def configure_optimizers(self):
        return None

