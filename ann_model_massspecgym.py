import torch
import torch.nn as nn
from massspecgym.models.retrieval.base import RetrievalMassSpecGymModel
from massspecgym.models.base import Stage
from ann_model import ANN_Class

class AnnRetrieval(RetrievalMassSpecGymModel):

    def __init__(self, mol_embedding_dim=300, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ann = ANN_Class(input_dim=70000, output_dim=mol_embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, 70000)
        return self.ann(x)  # output: (batch_size, 300)

    def step(self, batch: dict, stage: Stage) -> dict:
        x = batch["spec"]
        mol_true = batch["mol"]
        mol_pred = self.forward(x)

        loss = nn.functional.mse_loss(mol_pred, mol_true)

        scores = None
        if stage == Stage.TEST:
            cands = batch["candidates_mol"]
            batch_ptr = batch["batch_ptr"]
            mol_pred_repeated = mol_pred.repeat_interleave(batch_ptr, dim=0)
            scores = nn.functional.cosine_similarity(mol_pred_repeated, cands)

        return dict(loss=loss, scores=scores)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)