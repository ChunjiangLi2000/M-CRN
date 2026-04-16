import numpy as np
import torch
from rdkit import Chem
from lightning import pytorch as pl

from chemprop import data, featurizers, models


def predict_batch(smiles_list, checkpoint_path, batch_size=128):
    # ===== 1. canonicalize =====
    valid_smis = []
    valid_indices = []

    for i, smi in enumerate(smiles_list):
        smi = str(smi).strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        cano = Chem.MolToSmiles(mol, canonical=True)
        valid_smis.append(cano)
        valid_indices.append(i)

    if len(valid_smis) == 0:
        raise ValueError("没有有效的 SMILES")

    # ===== 2. dataset =====
    ys = np.zeros((len(valid_smis), 1), dtype=np.float32)

    datapoints = [
        data.MoleculeDatapoint.from_smi(smi, y)
        for smi, y in zip(valid_smis, ys)
    ]

    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
    dataset = data.MoleculeDataset(datapoints, featurizer)

    loader = data.build_dataloader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    # ===== 3. model =====
    model = models.MPNN.load_from_checkpoint(checkpoint_path)

    trainer = pl.Trainer(
        logger=False,
        accelerator="auto",
        devices=1,
    )

    # ===== 4. predict =====
    preds_batches = trainer.predict(model, dataloaders=loader)
    preds = torch.cat(preds_batches, dim=0).cpu().numpy().reshape(-1)
    results = [np.nan] * len(smiles_list)
    for idx, pred in zip(valid_indices, preds):
        results[idx] = float(pred)

    return results


# input
smiles_list = [
    "CCO",
    "CCN"
]

ckpt = "path/to/checkpoint.ckpt"

# output
preds = predict_batch(smiles_list, ckpt)

for smi, pred in zip(smiles_list, preds):
    print(f"{smi} -> {pred}")