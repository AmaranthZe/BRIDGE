"""Data loading, validation, and indexer construction."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_go_csv(path: str) -> pd.DataFrame:
    """Load GO annotation CSV. Columns: ENSGID, GeneName, GOTerm."""
    df = pd.read_csv(path)
    df.columns = ["gene_id", "gene_name", "go_term"]
    df = df.dropna(subset=["gene_id", "go_term"])
    df = df.drop_duplicates(subset=["gene_id", "go_term"])
    logger.info(f"Loaded {len(df)} GO annotations from {Path(path).name}")
    return df


def load_disease_csv(path: str) -> pd.DataFrame:
    """Load disease CSV. Columns: datatypeId,datasourceId,diseaseId,targetId,score,evidenceCount."""
    df = pd.read_csv(path)
    df = df.rename(columns={"targetId": "gene_id", "diseaseId": "disease_id"})
    df = df.dropna(subset=["gene_id", "disease_id", "score"])
    # Keep max score per gene-disease pair
    df = df.groupby(["gene_id", "disease_id"], as_index=False)["score"].max()
    logger.info(f"Loaded {len(df)} disease associations from {Path(path).name}")
    return df


def validate_species(gene_ids: pd.Series, expected_prefix: str) -> pd.Series:
    """Filter gene IDs by expected species prefix."""
    mask = gene_ids.str.startswith(expected_prefix)
    n_invalid = (~mask).sum()
    if n_invalid > 0:
        logger.warning(f"{n_invalid} gene IDs don't match prefix '{expected_prefix}'")
    return mask


def build_indexers(dataframes: Dict[str, pd.DataFrame]) -> Dict:
    """Build ID-to-index mappings for all entity types."""
    # Collect unique IDs
    human_genes = set()
    mouse_genes = set()
    bp_terms = set()
    mf_terms = set()
    diseases = set()
    gene_symbols = {}  # gene_id -> symbol

    for key in ["human_gobp", "human_gomf"]:
        df = dataframes[key]
        valid = validate_species(df["gene_id"], "ENSG")
        human_genes.update(df.loc[valid, "gene_id"])
        for _, row in df[valid].iterrows():
            gene_symbols[row["gene_id"]] = row["gene_name"]

    for key in ["mouse_gobp", "mouse_gomf"]:
        df = dataframes[key]
        valid = validate_species(df["gene_id"], "ENSMUSG")
        mouse_genes.update(df.loc[valid, "gene_id"])
        for _, row in df[valid].iterrows():
            gene_symbols[row["gene_id"]] = row["gene_name"]

    bp_terms.update(dataframes["human_gobp"]["go_term"])
    bp_terms.update(dataframes["mouse_gobp"]["go_term"])
    mf_terms.update(dataframes["human_gomf"]["go_term"])
    mf_terms.update(dataframes["mouse_gomf"]["go_term"])

    for key in ["human_disease", "mouse_disease"]:
        df = dataframes[key]
        if key == "human_disease":
            valid = validate_species(df["gene_id"], "ENSG")
            human_genes.update(df.loc[valid, "gene_id"])
        else:
            valid = validate_species(df["gene_id"], "ENSMUSG")
            mouse_genes.update(df.loc[valid, "gene_id"])
        diseases.update(df.loc[valid, "disease_id"])

    # Build sorted indexers
    indexers = {
        "human_genes": {g: i for i, g in enumerate(sorted(human_genes))},
        "mouse_genes": {g: i for i, g in enumerate(sorted(mouse_genes))},
        "bp_terms": {t: i for i, t in enumerate(sorted(bp_terms))},
        "mf_terms": {t: i for i, t in enumerate(sorted(mf_terms))},
        "diseases": {d: i for i, d in enumerate(sorted(diseases))},
        # Inverse mappings
        "human_genes_inv": {i: g for g, i in zip(sorted(human_genes), range(len(human_genes)))},
        "mouse_genes_inv": {i: g for g, i in zip(sorted(mouse_genes), range(len(mouse_genes)))},
        "gene_symbols": gene_symbols,
    }

    logger.info(
        f"Indexers: {len(indexers['human_genes'])} human genes, "
        f"{len(indexers['mouse_genes'])} mouse genes, "
        f"{len(indexers['bp_terms'])} BP, {len(indexers['mf_terms'])} MF, "
        f"{len(indexers['diseases'])} diseases"
    )
    return indexers


def load_all_data(cfg) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """Load all data files and build indexers."""
    raw_dir = Path(cfg.paths.raw_dir)

    dataframes = {
        "human_gobp": load_go_csv(raw_dir / cfg.io.human_gobp),
        "human_gomf": load_go_csv(raw_dir / cfg.io.human_gomf),
        "mouse_gobp": load_go_csv(raw_dir / cfg.io.mouse_gobp),
        "mouse_gomf": load_go_csv(raw_dir / cfg.io.mouse_gomf),
        "human_disease": load_disease_csv(raw_dir / cfg.io.human_dis),
        "mouse_disease": load_disease_csv(raw_dir / cfg.io.mouse_dis),
    }

    indexers = build_indexers(dataframes)
    return dataframes, indexers
