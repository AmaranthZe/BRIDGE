"""
Data loading, validation, and indexer construction.

This module handles:
- Loading GO annotation CSVs (human/mouse GOBP/GOMF)
- Loading Disease association CSVs (human/mouse)
- Building ID indexers for genes, GO terms, and diseases
- Validating data integrity
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_go_csv(path: str, species: str) -> pd.DataFrame:
    """
    Load GO annotation file.

    Args:
        path: Path to CSV file
        species: 'human' or 'mouse'

    Returns:
        DataFrame with columns: gene_id, gene_name, go_term
    """
    logger.info(f"Loading GO data from {path} for {species}")
    df = pd.read_csv(path)

    # Standardize column names
    df.columns = ['gene_id', 'gene_name', 'go_term']

    # Validate species by checking gene ID prefixes
    if species == 'human':
        # Human genes start with ENSG but not ENSMUSG
        valid = df['gene_id'].str.startswith('ENSG') & ~df['gene_id'].str.startswith('ENSMUSG')
        if not valid.all():
            invalid_count = (~valid).sum()
            logger.warning(f"Found {invalid_count} non-human gene IDs in {path}")
            df = df[valid].copy()
    else:
        # Mouse genes start with ENSMUSG
        valid = df['gene_id'].str.startswith('ENSMUSG')
        if not valid.all():
            invalid_count = (~valid).sum()
            logger.warning(f"Found {invalid_count} non-mouse gene IDs in {path}")
            df = df[valid].copy()

    # Remove duplicates (same gene-term pair)
    df = df.drop_duplicates(subset=['gene_id', 'go_term'])

    logger.info(f"Loaded {len(df)} GO annotations for {df['gene_id'].nunique()} {species} genes")
    return df


def load_disease_csv(path: str, species: str) -> pd.DataFrame:
    """
    Load disease association file.

    Args:
        path: Path to CSV file
        species: 'human' or 'mouse'

    Returns:
        DataFrame with columns: disease_id, gene_id, score
    """
    logger.info(f"Loading disease data from {path} for {species}")
    df = pd.read_csv(path)

    # Extract relevant columns
    df = df[['diseaseId', 'targetId', 'score']].copy()
    df.columns = ['disease_id', 'gene_id', 'score']

    # Validate species by checking gene ID prefixes
    if species == 'human':
        valid = df['gene_id'].str.startswith('ENSG') & ~df['gene_id'].str.startswith('ENSMUSG')
        if not valid.all():
            invalid_count = (~valid).sum()
            logger.warning(f"Found {invalid_count} non-human gene IDs in {path}")
            df = df[valid].copy()
    else:
        valid = df['gene_id'].str.startswith('ENSMUSG')
        if not valid.all():
            invalid_count = (~valid).sum()
            logger.warning(f"Found {invalid_count} non-mouse gene IDs in {path}")
            df = df[valid].copy()

    # Handle duplicates: keep max score for same (gene, disease) pair
    df = df.groupby(['gene_id', 'disease_id'], as_index=False)['score'].max()

    logger.info(f"Loaded {len(df)} disease associations for {df['gene_id'].nunique()} {species} genes")
    return df


def build_indexers(
    human_gobp: pd.DataFrame,
    human_gomf: pd.DataFrame,
    human_disease: pd.DataFrame,
    mouse_gobp: pd.DataFrame,
    mouse_gomf: pd.DataFrame,
    mouse_disease: pd.DataFrame,
    go_roots: Optional[List[str]] = None
) -> Dict[str, Dict[str, int]]:
    """
    Build ID-to-index mappings for all entity types.

    Args:
        human_gobp: Human GOBP annotations
        human_gomf: Human GOMF annotations
        human_disease: Human disease associations
        mouse_gobp: Mouse GOBP annotations
        mouse_gomf: Mouse GOMF annotations
        mouse_disease: Mouse disease associations
        go_roots: GO root terms to exclude

    Returns:
        Dictionary containing indexers for each entity type:
        - human_genes: {gene_id: index}
        - mouse_genes: {gene_id: index}
        - bp_terms: {go_term: index}
        - mf_terms: {go_term: index}
        - diseases: {disease_id: index}
        - human_gene_names: {gene_id: gene_name}
        - mouse_gene_names: {gene_id: gene_name}
    """
    if go_roots is None:
        go_roots = ['GO:0008150', 'GO:0003674', 'GO:0005575']

    go_roots_set = set(go_roots)

    # Collect unique human genes
    human_genes = set()
    human_genes.update(human_gobp['gene_id'].unique())
    human_genes.update(human_gomf['gene_id'].unique())
    human_genes.update(human_disease['gene_id'].unique())
    human_genes = sorted(human_genes)

    # Collect unique mouse genes
    mouse_genes = set()
    mouse_genes.update(mouse_gobp['gene_id'].unique())
    mouse_genes.update(mouse_gomf['gene_id'].unique())
    mouse_genes.update(mouse_disease['gene_id'].unique())
    mouse_genes = sorted(mouse_genes)

    # Collect BP terms (from both species, excluding roots)
    bp_terms = set()
    bp_terms.update(human_gobp['go_term'].unique())
    bp_terms.update(mouse_gobp['go_term'].unique())
    bp_terms = bp_terms - go_roots_set
    bp_terms = sorted(bp_terms)

    # Collect MF terms (from both species, excluding roots)
    mf_terms = set()
    mf_terms.update(human_gomf['go_term'].unique())
    mf_terms.update(mouse_gomf['go_term'].unique())
    mf_terms = mf_terms - go_roots_set
    mf_terms = sorted(mf_terms)

    # Collect diseases (from both species)
    diseases = set()
    diseases.update(human_disease['disease_id'].unique())
    diseases.update(mouse_disease['disease_id'].unique())
    diseases = sorted(diseases)

    # Build indexers
    indexers = {
        'human_genes': {g: i for i, g in enumerate(human_genes)},
        'mouse_genes': {g: i for i, g in enumerate(mouse_genes)},
        'bp_terms': {t: i for i, t in enumerate(bp_terms)},
        'mf_terms': {t: i for i, t in enumerate(mf_terms)},
        'diseases': {d: i for i, d in enumerate(diseases)},
    }

    # Build gene name mappings
    human_gene_names = {}
    for df in [human_gobp, human_gomf]:
        for _, row in df[['gene_id', 'gene_name']].drop_duplicates().iterrows():
            if row['gene_id'] not in human_gene_names:
                human_gene_names[row['gene_id']] = row['gene_name']

    mouse_gene_names = {}
    for df in [mouse_gobp, mouse_gomf]:
        for _, row in df[['gene_id', 'gene_name']].drop_duplicates().iterrows():
            if row['gene_id'] not in mouse_gene_names:
                mouse_gene_names[row['gene_id']] = row['gene_name']

    indexers['human_gene_names'] = human_gene_names
    indexers['mouse_gene_names'] = mouse_gene_names

    # Add inverse indexers for convenience
    indexers['human_genes_inv'] = {i: g for g, i in indexers['human_genes'].items()}
    indexers['mouse_genes_inv'] = {i: g for g, i in indexers['mouse_genes'].items()}
    indexers['bp_terms_inv'] = {i: t for t, i in indexers['bp_terms'].items()}
    indexers['mf_terms_inv'] = {i: t for t, i in indexers['mf_terms'].items()}
    indexers['diseases_inv'] = {i: d for d, i in indexers['diseases'].items()}

    logger.info(f"Built indexers: {len(indexers['human_genes'])} human genes, "
                f"{len(indexers['mouse_genes'])} mouse genes, "
                f"{len(indexers['bp_terms'])} BP terms, "
                f"{len(indexers['mf_terms'])} MF terms, "
                f"{len(indexers['diseases'])} diseases")

    return indexers


def save_indexers(indexers: Dict, path: str):
    """Save indexers to JSON file."""
    # Convert int keys to strings for JSON serialization
    serializable = {}
    for key, value in indexers.items():
        if isinstance(value, dict):
            serializable[key] = {str(k): v for k, v in value.items()}
        else:
            serializable[key] = value

    with open(path, 'w') as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved indexers to {path}")


def load_indexers(path: str) -> Dict:
    """Load indexers from JSON file."""
    with open(path, 'r') as f:
        data = json.load(f)

    # Convert string keys back to appropriate types
    indexers = {}
    for key, value in data.items():
        if key.endswith('_inv'):
            # Inverse indexers have int keys
            indexers[key] = {int(k): v for k, v in value.items()}
        else:
            indexers[key] = value

    logger.info(f"Loaded indexers from {path}")
    return indexers


def validate_data(
    human_gobp: pd.DataFrame,
    human_gomf: pd.DataFrame,
    human_disease: pd.DataFrame,
    mouse_gobp: pd.DataFrame,
    mouse_gomf: pd.DataFrame,
    mouse_disease: pd.DataFrame
) -> bool:
    """
    Validate data integrity.

    Returns:
        True if all validations pass
    """
    errors = []

    # Check for empty dataframes
    for name, df in [
        ('human_gobp', human_gobp),
        ('human_gomf', human_gomf),
        ('human_disease', human_disease),
        ('mouse_gobp', mouse_gobp),
        ('mouse_gomf', mouse_gomf),
        ('mouse_disease', mouse_disease)
    ]:
        if len(df) == 0:
            errors.append(f"{name} is empty")

    # Check for null values in critical columns
    for name, df in [
        ('human_gobp', human_gobp),
        ('human_gomf', human_gomf),
        ('mouse_gobp', mouse_gobp),
        ('mouse_gomf', mouse_gomf)
    ]:
        if df['gene_id'].isnull().any():
            errors.append(f"{name} has null gene_id values")
        if df['go_term'].isnull().any():
            errors.append(f"{name} has null go_term values")

    for name, df in [
        ('human_disease', human_disease),
        ('mouse_disease', mouse_disease)
    ]:
        if df['gene_id'].isnull().any():
            errors.append(f"{name} has null gene_id values")
        if df['disease_id'].isnull().any():
            errors.append(f"{name} has null disease_id values")
        if df['score'].isnull().any():
            errors.append(f"{name} has null score values")

    # Check disease scores are in valid range
    for name, df in [
        ('human_disease', human_disease),
        ('mouse_disease', mouse_disease)
    ]:
        if (df['score'] < 0).any() or (df['score'] > 1).any():
            errors.append(f"{name} has scores outside [0, 1] range")

    # Check for shared GO terms between species (expected)
    human_bp = set(human_gobp['go_term'].unique())
    mouse_bp = set(mouse_gobp['go_term'].unique())
    shared_bp = human_bp & mouse_bp
    if len(shared_bp) == 0:
        errors.append("No shared BP terms between human and mouse (unexpected)")
    else:
        logger.info(f"Found {len(shared_bp)} shared BP terms between species")

    human_mf = set(human_gomf['go_term'].unique())
    mouse_mf = set(mouse_gomf['go_term'].unique())
    shared_mf = human_mf & mouse_mf
    if len(shared_mf) == 0:
        errors.append("No shared MF terms between human and mouse (unexpected)")
    else:
        logger.info(f"Found {len(shared_mf)} shared MF terms between species")

    # Check for shared diseases
    human_dis = set(human_disease['disease_id'].unique())
    mouse_dis = set(mouse_disease['disease_id'].unique())
    shared_dis = human_dis & mouse_dis
    logger.info(f"Found {len(shared_dis)} shared diseases between species")

    if errors:
        for err in errors:
            logger.error(f"Validation error: {err}")
        return False

    logger.info("Data validation passed")
    return True


def load_all_data(config: Dict) -> Tuple[Dict[str, pd.DataFrame], Dict]:
    """
    Load all data files and build indexers.

    Args:
        config: Configuration dictionary with paths

    Returns:
        Tuple of (dataframes dict, indexers dict)
    """
    raw_dir = Path(config['paths']['raw_dir'])
    io_cfg = config['io']
    preproc_cfg = config['preproc']

    # Load all dataframes
    human_gobp = load_go_csv(raw_dir / io_cfg['human_gobp'], 'human')
    human_gomf = load_go_csv(raw_dir / io_cfg['human_gomf'], 'human')
    human_disease = load_disease_csv(raw_dir / io_cfg['human_dis'], 'human')
    mouse_gobp = load_go_csv(raw_dir / io_cfg['mouse_gobp'], 'mouse')
    mouse_gomf = load_go_csv(raw_dir / io_cfg['mouse_gomf'], 'mouse')
    mouse_disease = load_disease_csv(raw_dir / io_cfg['mouse_dis'], 'mouse')

    # Validate
    if not validate_data(human_gobp, human_gomf, human_disease,
                         mouse_gobp, mouse_gomf, mouse_disease):
        raise ValueError("Data validation failed")

    # Build indexers
    go_roots = preproc_cfg.get('go_roots', ['GO:0008150', 'GO:0003674', 'GO:0005575'])
    indexers = build_indexers(
        human_gobp, human_gomf, human_disease,
        mouse_gobp, mouse_gomf, mouse_disease,
        go_roots=go_roots
    )

    dataframes = {
        'human_gobp': human_gobp,
        'human_gomf': human_gomf,
        'human_disease': human_disease,
        'mouse_gobp': mouse_gobp,
        'mouse_gomf': mouse_gomf,
        'mouse_disease': mouse_disease
    }

    return dataframes, indexers
