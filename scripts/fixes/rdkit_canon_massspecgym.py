"""Canonicalize all SMILES in MassSpecGym with RDKit (v1.5 release).

Produces:
  - MassSpecGym1.5.tsv  (smiles column canonicalized)
  - MassSpecGym1.5.mgf  (MGF export of the above)
  - MassSpecGym1.5_retrieval_candidates_formula.json
  - MassSpecGym1.5_retrieval_candidates_mass.json

All SMILES (TSV column, JSON keys, and ALL JSON candidate values) are
canonicalized with massspecgym.utils.rdkit_canonical_smiles.
"""

import json
import logging
import random
import sys
from collections import Counter
from pathlib import Path
import os

import numpy as np
import pandas as pd
from matchms import Spectrum
from matchms.exporting import save_as_mgf
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.inchi import MolToInchi, InchiToInchiKey
from tqdm import tqdm

import massspecgym.utils as utils

RDLogger.logger().setLevel(RDLogger.CRITICAL)

BASE = Path("/scratch/project_465002061/rbushuie/DreaMS-Mol_dev")
TSV_IN = BASE / "MassSpecGym/data/MassSpecGym.tsv"
TSV_OUT = BASE / "MassSpecGym/data/MassSpecGym1.5.tsv"
MGF_OUT = BASE / "MassSpecGym/data/MassSpecGym1.5.mgf"

CAND_JSONS = [
    BASE / "MassSpecGym/data/MassSpecGym_retrieval_candidates_formula.json",
    BASE / "MassSpecGym/data/MassSpecGym_retrieval_candidates_mass.json",
]
JSON_OUT_DIR = BASE / "MassSpecGym/data"

JSON_SANITY_SAMPLE = 2000  # random keys+first-cands probed before re-canonicalization

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


canon_counts = Counter()  # "canonical_true", "canonical_fallback", "kept_original"


def rdkit_canonical(smi: str) -> str:
    result, outcome = utils.rdkit_canonical_smiles(smi, return_outcome=True)
    canon_counts[outcome] += 1
    return result


def main():
    # ── 1. Collect ALL unique SMILES from TSV + JSONs ────────────────────
    log.info("Reading %s ...", TSV_IN)
    df = pd.read_csv(TSV_IN, sep="\t")
    log.info("Loaded %d rows", len(df))

    all_smiles: set[str] = set(df["smiles"].unique())
    log.info("Unique SMILES in TSV: %d", len(all_smiles))

    json_data: dict[Path, dict[str, list[str]]] = {}
    for json_path in CAND_JSONS:
        log.info("Reading %s ...", json_path)
        with open(json_path) as f:
            cands: dict[str, list[str]] = json.load(f)
        json_data[json_path] = cands
        for key, vals in cands.items():
            all_smiles.add(key)
            all_smiles.update(vals)
        log.info("  %d keys loaded", len(cands))

    log.info("Total unique SMILES across TSV + all JSONs: %d", len(all_smiles))

    # ── 1b. JSON canonicality sanity check ───────────────────────────────
    # Documentation step: we ALWAYS proceed to full re-canonicalization in step 2.
    # This block just empirically reports how much of each input JSON would change,
    # so a reader can confirm that re-canonicalizing the JSONs is (or isn't) necessary.
    log.info("--- JSON canonicality sanity check ---")
    rng = random.Random(0)
    for json_path, cands in json_data.items():
        keys = list(cands.keys())
        n_sample = min(JSON_SANITY_SAMPLE, len(keys))
        sample_keys = rng.sample(keys, n_sample)
        n_keys_changed = 0
        n_vals_total = 0
        n_vals_changed = 0
        for k in sample_keys:
            kc = utils.rdkit_canonical_smiles(k)
            if kc != k:
                n_keys_changed += 1
            vals = cands[k]
            if vals:
                v = vals[0]
                vc = utils.rdkit_canonical_smiles(v)
                n_vals_total += 1
                if vc != v:
                    n_vals_changed += 1
        log.info(
            "  %s: %d/%d keys (%.2f%%) and %d/%d first-candidates (%.2f%%) "
            "would change under re-canonicalization.",
            json_path.name,
            n_keys_changed, n_sample,
            100.0 * n_keys_changed / max(1, n_sample),
            n_vals_changed, n_vals_total,
            100.0 * n_vals_changed / max(1, n_vals_total),
        )

    # ── 2. Build global canonical mapping ────────────────────────────────
    log.info("Canonicalizing %d unique SMILES ...", len(all_smiles))
    canon_map: dict[str, str] = {}
    for i, smi in enumerate(sorted(all_smiles)):
        if i % 500_000 == 0 and i > 0:
            log.info("  Canonicalization progress: %d / %d", i, len(all_smiles))
        canon_map[smi] = rdkit_canonical(smi)
    n_changed = sum(1 for k, v in canon_map.items() if k != v)
    log.info("--- Canonicalization summary ---")
    log.info("  Total unique SMILES: %d", len(canon_map))
    log.info("  MolToSmiles(canonical=True): %d", canon_counts["canonical_true"])
    log.info("  MolToSmiles() fallback:      %d", canon_counts["canonical_fallback"])
    log.info("  Kept original SMILES:        %d", canon_counts["kept_original"])
    log.info("  SMILES that changed:         %d", n_changed)

    # ── 3. Replace SMILES in TSV and validate properties ─────────────────
    orig_smiles_col = df["smiles"].copy()
    df["smiles"] = df["smiles"].map(canon_map)

    log.info("Validating formula, parent_mass, inchikey against canonical SMILES ...")
    error_counts: Counter = Counter()
    validation_errors: list[str] = []
    MASS_TOL = 0.1

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):
        new_smi = row["smiles"]
        orig_smi = orig_smiles_col.iloc[idx] if hasattr(orig_smiles_col, "iloc") else orig_smiles_col[idx]
        mol = Chem.MolFromSmiles(new_smi)
        if mol is None:
            error_counts["parse_failed"] += 1
            validation_errors.append(
                f"[parse_failed] Row {idx}: cannot parse new_smi='{new_smi}' orig_smi='{orig_smi}'")
            continue

        computed_formula = rdMolDescriptors.CalcMolFormula(mol)
        computed_mass = rdMolDescriptors.CalcExactMolWt(mol)
        inchi = MolToInchi(mol)
        computed_ik = InchiToInchiKey(inchi)[:14] if inchi else None

        orig_formula = str(row["formula"])
        orig_mass = float(row["parent_mass"])
        orig_ik = str(row["inchikey"])[:14]

        if computed_formula != orig_formula:
            error_counts["formula_mismatch"] += 1
            validation_errors.append(
                f"[formula_mismatch] Row {idx}: stored='{orig_formula}' computed='{computed_formula}' "
                f"orig_smi='{orig_smi}' new_smi='{new_smi}'")

        if abs(computed_mass - orig_mass) >= MASS_TOL:
            error_counts["mass_mismatch"] += 1
            validation_errors.append(
                f"[mass_mismatch] Row {idx}: stored={orig_mass} computed={computed_mass} "
                f"orig_smi='{orig_smi}' new_smi='{new_smi}'")

        if computed_ik != orig_ik:
            error_counts["inchikey_mismatch"] += 1
            validation_errors.append(
                f"[inchikey_mismatch] Row {idx}: stored='{orig_ik}' computed='{computed_ik}' "
                f"orig_smi='{orig_smi}' new_smi='{new_smi}'")

    log.info("--- Property validation summary ---")
    log.info("  Total rows: %d", len(df))
    if error_counts:
        for err_type, cnt in error_counts.most_common():
            log.info("  %s: %d", err_type, cnt)
        log.info("  All validation errors:")
        for err in validation_errors:
            log.info("    %s", err)
    else:
        log.info("  All properties match — canonicalization is safe.")

    log.info("Writing %s ...", TSV_OUT)
    df.to_csv(TSV_OUT, sep="\t", index=False)

    # ── 4. Convert to MGF ────────────────────────────────────────────────
    log.info("Converting to MGF ...")
    spectra = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building spectra"):
        metadata = {
            k: v for k, v in row.items()
            if k not in ("mzs", "intensities") and v is not np.nan
        }
        spec = Spectrum(
            mz=utils.parse_spec_array(row["mzs"]),
            intensities=utils.parse_spec_array(row["intensities"]),
            metadata=metadata,
        )
        spectra.append(spec)
    # Delete previous MGF if it exists
    if MGF_OUT.exists():
        log.info("Previous MGF file exists. Deleting %s", MGF_OUT)
        os.remove(MGF_OUT)
    log.info("Writing %s (%d spectra) ...", MGF_OUT, len(spectra))
    save_as_mgf(spectra, str(MGF_OUT))

    # ── 5. Process candidate JSONs ───────────────────────────────────────
    for json_in_path, cands in json_data.items():
        log.info("=" * 60)
        log.info("Processing %s ...", json_in_path)

        first_eq_key = 0
        first_ne_key = 0
        first_ne_key_examples: list[str] = []

        new_cands: dict[str, list[str]] = {}
        for key, vals in cands.items():
            new_key = canon_map[key]

            if vals and vals[0] == key:
                first_eq_key += 1
            else:
                first_ne_key += 1
                first_ne_key_examples.append(
                    f"key='{key}' first='{vals[0] if vals else '<empty>'}'")

            new_vals = [canon_map[v] for v in vals]
            new_cands[new_key] = new_vals

        stem = json_in_path.stem.replace("MassSpecGym_retrieval_candidates_", "")
        stem_tag = stem.split("_")[0]
        json_out_path = JSON_OUT_DIR / f"MassSpecGym1.5_retrieval_candidates_{stem_tag}.json"

        log.info("Writing %s ...", json_out_path)
        with open(json_out_path, "w") as f:
            json.dump(new_cands, f)

        log.info("--- JSON summary for %s ---", stem_tag)
        log.info("  Total keys: %d", len(cands))
        log.info("  First element == key (expected): %d", first_eq_key)
        log.info("  First element != key (UNEXPECTED): %d", first_ne_key)
        if first_ne_key_examples:
            for ex in first_ne_key_examples:
                log.info("    %s", ex)

    # ── 6. Final summary ─────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("ALL DONE")
    log.info("=" * 60)
    log.info("TSV: %s", TSV_OUT)
    log.info("MGF: %s", MGF_OUT)
    log.info("Candidate JSONs written to: %s", JSON_OUT_DIR)


if __name__ == "__main__":
    main()
