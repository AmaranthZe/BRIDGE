"""
Disease Report Module for Cross-Species Gene Mapping.

Given a query gene (mouse or human), this module computes a comprehensive
disease relevance report by combining three evidence sources:

  A. **Direct association** – diseases linked to the query gene itself
     (e.g. IMPC phenotype for mouse, Open Targets multi-source for human).
  B. **Ortholog association** – diseases linked to the cross-species ortholog
     identified via functional annotation similarity.
  C. **Functional-cluster enrichment** – diseases enriched across the top-N
     functionally similar genes in the target species, weighted by their
     similarity scores.

The combined relevance score is::

    Combined = w_A * direct_tfidf + w_B * ortholog_tfidf + w_C * enrichment_score

Results can be exported to Excel (.xlsx) or JSON.

Usage from CLI::

    python -m src.cli disease-report --ids "ENSMUSG00000022100" -o output.xlsx

Usage from Python::

    from src.disease_report import run_disease_report
    result = run_disease_report(query="ENSMUSG00000022100", config=config)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature-similarity helpers
# ---------------------------------------------------------------------------

def _compute_feature_similarity(
    source_idx: int,
    W_source: Dict[str, sp.csr_matrix],
    W_target: Dict[str, sp.csr_matrix],
    feature_weights: Dict[str, float],
) -> np.ndarray:
    """
    Compute annotation-based feature similarity between one source gene
    and all genes in the target species.

    Returns:
        1-D array of similarity scores, length = #target_genes.
    """
    sim = None
    for key in ('bp', 'mf', 'disease'):
        w = feature_weights.get(key, 0.0)
        if w == 0.0:
            continue
        row = W_source[key][source_idx]
        col_sim = (row @ W_target[key].T).toarray().flatten()
        if sim is None:
            sim = w * col_sim
        else:
            sim += w * col_sim
    if sim is None:
        return np.zeros(W_target['bp'].shape[0])
    return sim


def find_functional_matches(
    source_idx: int,
    W_source: Dict[str, sp.csr_matrix],
    W_target: Dict[str, sp.csr_matrix],
    feature_weights: Dict[str, float],
    topk: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find top-K functionally similar genes in the target species.

    Returns:
        (indices, scores) – both 1-D arrays of length topk.
    """
    sim = _compute_feature_similarity(
        source_idx, W_source, W_target, feature_weights,
    )
    topk = min(topk, len(sim))
    top_idx = np.argsort(sim)[::-1][:topk]
    top_scores = sim[top_idx]
    return top_idx, top_scores


# ---------------------------------------------------------------------------
# Disease collection helpers
# ---------------------------------------------------------------------------

def _collect_direct_diseases(
    gene_idx: int,
    W_disease: sp.csr_matrix,
    dis_inv: Dict[int, str],
    raw_df: Optional[pd.DataFrame] = None,
    gene_id: Optional[str] = None,
) -> Dict[str, Dict]:
    """
    Collect direct disease associations for one gene.

    Returns:
        {disease_id: {'tfidf': float, 'raw_score': float, 'evidence_count': int,
                       'datasources': str}}
    """
    row = W_disease[gene_idx]
    nz_cols = row.nonzero()[1]
    weights = row.toarray().flatten()

    raw_scores = {}
    raw_ev = {}
    raw_src = {}
    if raw_df is not None and gene_id is not None:
        sub = raw_df[raw_df['targetId'] == gene_id]
        raw_scores = sub.groupby('diseaseId')['score'].max().to_dict()
        raw_ev = sub.groupby('diseaseId')['evidenceCount'].sum().to_dict()
        if 'datasourceId' in sub.columns:
            raw_src = sub.groupby('diseaseId')['datasourceId'].apply(
                lambda x: ';'.join(sorted(set(x)))
            ).to_dict()

    result = {}
    for col in nz_cols:
        did = dis_inv.get(int(col), f'UNKNOWN_{col}')
        result[did] = {
            'tfidf': float(weights[col]),
            'raw_score': float(raw_scores.get(did, 0.0)),
            'evidence_count': int(raw_ev.get(did, 0)),
            'datasources': raw_src.get(did, ''),
        }
    return result


def _collect_enrichment_diseases(
    matched_indices: np.ndarray,
    matched_scores: np.ndarray,
    matched_names: List[str],
    W_disease: sp.csr_matrix,
    dis_inv: Dict[int, str],
) -> Dict[str, Dict]:
    """
    Aggregate disease associations across a set of functionally matched genes,
    weighted by their similarity scores.

    Returns:
        {disease_id: {'weighted_score': float, 'gene_count': int,
                       'genes': str}}
    """
    result: Dict[str, Dict] = {}
    for idx, score, name in zip(matched_indices, matched_scores, matched_names):
        if score <= 0:
            continue
        row = W_disease[int(idx)]
        nz_cols = row.nonzero()[1]
        vals = row.toarray().flatten()
        for col in nz_cols:
            did = dis_inv.get(int(col), f'UNKNOWN_{col}')
            if did not in result:
                result[did] = {
                    'weighted_score': 0.0,
                    'gene_names': [],
                }
            result[did]['weighted_score'] += float(score * vals[col])
            result[did]['gene_names'].append(name)

    # Finalise
    for did, info in result.items():
        names = sorted(set(info['gene_names']))
        info['gene_count'] = len(names)
        info['genes'] = ';'.join(names)
        del info['gene_names']

    return result


# ---------------------------------------------------------------------------
# Disease name resolution
# ---------------------------------------------------------------------------

def resolve_disease_names(
    disease_ids: List[str],
    batch_size: int = 20,
    timeout: int = 30,
) -> Dict[str, str]:
    """
    Resolve disease ontology IDs to human-readable names via the
    Open Targets Platform GraphQL API.

    Falls back gracefully: if the API is unreachable or a disease is not
    found the ID itself is returned as the name.
    """
    try:
        import requests  # noqa: delay import to keep module importable without requests
    except ImportError:
        logger.warning("'requests' not installed – skipping disease name resolution")
        return {}

    url = "https://api.platform.opentargets.org/api/v4/graphql"
    names: Dict[str, str] = {}
    id_list = list(dict.fromkeys(disease_ids))  # deduplicate, preserve order

    for i in range(0, len(id_list), batch_size):
        batch = id_list[i:i + batch_size]
        fragments = [
            f'd{j}: disease(efoId: "{did}") {{ id name }}'
            for j, did in enumerate(batch)
        ]
        query = "{ " + " ".join(fragments) + " }"
        try:
            resp = requests.post(url, json={"query": query}, timeout=timeout)
            if resp.status_code == 200:
                data = resp.json().get('data', {})
                for j, did in enumerate(batch):
                    entry = data.get(f'd{j}')
                    if entry and entry.get('name'):
                        names[did] = entry['name']
        except Exception as exc:
            logger.debug(f"API batch {i} failed: {exc}")

    logger.info(f"Resolved {len(names)}/{len(id_list)} disease names via Open Targets")
    return names


# ---------------------------------------------------------------------------
# Main analysis class
# ---------------------------------------------------------------------------

class DiseaseReporter:
    """
    Orchestrates the 3-source disease analysis for a query gene.
    """

    def __init__(
        self,
        sparse_matrices: Dict[str, sp.csr_matrix],
        indexers: Dict,
        config: Dict,
        raw_human_disease: Optional[pd.DataFrame] = None,
        raw_mouse_disease: Optional[pd.DataFrame] = None,
    ):
        self.sparse_matrices = sparse_matrices
        self.indexers = indexers
        self.config = config

        # Build convenience dicts
        self.W_human = {
            'bp': sparse_matrices['W_hB'],
            'mf': sparse_matrices['W_hF'],
            'disease': sparse_matrices['W_hD'],
        }
        self.W_mouse = {
            'bp': sparse_matrices['W_mB'],
            'mf': sparse_matrices['W_mF'],
            'disease': sparse_matrices['W_mD'],
        }

        # Disease inverse indexer
        self.dis_inv = {
            int(v): k for k, v in indexers['diseases'].items()
        }

        # Gene name lookups
        self.human_gene_names = indexers.get('human_gene_names', {})
        self.mouse_gene_names = indexers.get('mouse_gene_names', {})

        # Feature weights (reuse from fgw config)
        fgw_cfg = config.get('fgw', {})
        self.feature_weights = fgw_cfg.get('feature_weights', {
            'bp': 0.25, 'mf': 0.15, 'disease': 0.60,
        })

        # Disease-report specific config
        dr_cfg = config.get('disease_report', {})
        self.topk_func_genes = dr_cfg.get('topk_func_genes', 10)
        self.weight_direct = dr_cfg.get('weight_direct', 0.30)
        self.weight_ortholog = dr_cfg.get('weight_ortholog', 0.30)
        self.weight_enrichment = dr_cfg.get('weight_enrichment', 0.40)
        self.resolve_names = dr_cfg.get('resolve_names', True)

        # Raw disease dataframes (optional, for raw score / evidence detail)
        self.raw_human_disease = raw_human_disease
        self.raw_mouse_disease = raw_mouse_disease

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def report(self, gene_id: str) -> Dict:
        """
        Generate a disease report for a single gene.

        Args:
            gene_id: Ensembl gene ID (ENSG… or ENSMUSG…)

        Returns:
            Result dict ready for export.
        """
        species = self._detect_species(gene_id)
        if species is None:
            return {'error': f'Unknown gene ID format: {gene_id}'}

        if species == 'mouse':
            source_indexer = self.indexers['mouse_genes']
            target_inv_indexer = self.indexers['human_genes_inv']
            W_source = self.W_mouse
            W_target = self.W_human
            raw_source = self.raw_mouse_disease
            raw_target = self.raw_human_disease
            source_names = self.mouse_gene_names
            target_names = self.human_gene_names
            target_species = 'human'
        else:
            source_indexer = self.indexers['human_genes']
            target_inv_indexer = self.indexers['mouse_genes_inv']
            W_source = self.W_human
            W_target = self.W_mouse
            raw_source = self.raw_human_disease
            raw_target = self.raw_mouse_disease
            source_names = self.human_gene_names
            target_names = self.mouse_gene_names
            target_species = 'mouse'

        if gene_id not in source_indexer:
            return {'error': f'Gene {gene_id} not found in {species} indexer'}

        source_idx = source_indexer[gene_id]
        source_symbol = source_names.get(gene_id, gene_id)

        # --- Source A: direct ---
        direct = _collect_direct_diseases(
            gene_idx=source_idx,
            W_disease=W_source['disease'],
            dis_inv=self.dis_inv,
            raw_df=raw_source,
            gene_id=gene_id,
        )

        # --- Find functional matches in target species ---
        match_idx, match_scores = find_functional_matches(
            source_idx=source_idx,
            W_source=W_source,
            W_target=W_target,
            feature_weights=self.feature_weights,
            topk=self.topk_func_genes,
        )
        match_names = [
            target_names.get(target_inv_indexer[int(i)], target_inv_indexer[int(i)])
            for i in match_idx
        ]
        match_ids = [target_inv_indexer[int(i)] for i in match_idx]

        # Identify the best ortholog candidate (rank-1 in func matches)
        ortholog_id = match_ids[0] if len(match_ids) > 0 else None
        ortholog_symbol = match_names[0] if match_names else None
        ortholog_idx = int(match_idx[0]) if len(match_idx) > 0 else None

        # --- Source B: ortholog direct diseases ---
        ortholog_diseases: Dict[str, Dict] = {}
        if ortholog_idx is not None:
            ortholog_diseases = _collect_direct_diseases(
                gene_idx=ortholog_idx,
                W_disease=W_target['disease'],
                dis_inv=self.dis_inv,
                raw_df=raw_target,
                gene_id=ortholog_id,
            )

        # --- Source C: functional-cluster enrichment ---
        enrichment = _collect_enrichment_diseases(
            matched_indices=match_idx,
            matched_scores=match_scores,
            matched_names=match_names,
            W_disease=W_target['disease'],
            dis_inv=self.dis_inv,
        )

        # --- Merge all disease IDs ---
        all_ids = set(direct) | set(ortholog_diseases) | set(enrichment)

        # --- Resolve disease names ---
        disease_names: Dict[str, str] = {}
        if self.resolve_names and all_ids:
            disease_names = resolve_disease_names(sorted(all_ids))

        # --- Build per-disease rows ---
        rows: List[Dict] = []
        for did in sorted(all_ids):
            d_direct = direct.get(did)
            d_ortho = ortholog_diseases.get(did)
            d_enrich = enrichment.get(did)

            src_tfidf = d_direct['tfidf'] if d_direct else 0.0
            ort_tfidf = d_ortho['tfidf'] if d_ortho else 0.0
            enr_score = d_enrich['weighted_score'] if d_enrich else 0.0

            combined = (
                self.weight_direct * src_tfidf
                + self.weight_ortholog * ort_tfidf
                + self.weight_enrichment * enr_score
            )

            evidence_sources = []
            if d_direct:
                evidence_sources.append(f'{species}_{source_symbol}')
            if d_ortho:
                evidence_sources.append(f'{target_species}_{ortholog_symbol}')
            if d_enrich:
                evidence_sources.append('FuncCluster')

            rows.append({
                'Disease_ID': did,
                'Disease_Name': disease_names.get(did, ''),
                'Ontology': did.split('_')[0] if '_' in did else did,
                'Combined_Relevance': combined,
                'Num_Sources': len(evidence_sources),
                'Evidence_Sources': ';'.join(evidence_sources),
                # Source A
                'Direct_TFIDF': src_tfidf,
                'Direct_RawScore': d_direct['raw_score'] if d_direct else 0.0,
                'Direct_EvidenceCount': d_direct['evidence_count'] if d_direct else 0,
                # Source B
                'Ortholog_TFIDF': ort_tfidf,
                'Ortholog_RawScore': d_ortho['raw_score'] if d_ortho else 0.0,
                'Ortholog_EvidenceCount': d_ortho['evidence_count'] if d_ortho else 0,
                'Ortholog_DataSources': d_ortho['datasources'] if d_ortho else '',
                # Source C
                'Enrichment_WeightedScore': enr_score,
                'Enrichment_GeneCount': d_enrich['gene_count'] if d_enrich else 0,
                'Enrichment_Genes': d_enrich['genes'] if d_enrich else '',
            })

        rows.sort(key=lambda r: -r['Combined_Relevance'])

        # --- Build functional matches list ---
        func_matches = []
        for i, (mid, mname, mscore) in enumerate(
            zip(match_ids, match_names, match_scores)
        ):
            # shared GO terms
            s_bp = set(W_source['bp'][source_idx].nonzero()[1])
            s_mf = set(W_source['mf'][source_idx].nonzero()[1])
            t_bp = set(W_target['bp'][int(match_idx[i])].nonzero()[1])
            t_mf = set(W_target['mf'][int(match_idx[i])].nonzero()[1])

            bp_inv = {int(v): k for k, v in self.indexers['bp_terms'].items()}
            mf_inv = {int(v): k for k, v in self.indexers['mf_terms'].items()}

            shared_bp = [bp_inv.get(x, '?') for x in sorted(s_bp & t_bp)]
            shared_mf = [mf_inv.get(x, '?') for x in sorted(s_mf & t_mf)]

            func_matches.append({
                'rank': i + 1,
                'gene_id': mid,
                'symbol': mname,
                'score': float(mscore),
                'shared_bp': ';'.join(shared_bp),
                'shared_mf': ';'.join(shared_mf),
            })

        # --- Assemble result ---
        return {
            'query': {
                'gene_id': gene_id,
                'symbol': source_symbol,
                'species': species,
            },
            'ortholog': {
                'gene_id': ortholog_id,
                'symbol': ortholog_symbol,
                'species': target_species,
            },
            'functional_matches': func_matches,
            'diseases': rows,
            'summary': {
                'total_diseases': len(rows),
                'direct_count': sum(1 for r in rows if r['Direct_TFIDF'] > 0),
                'ortholog_count': sum(1 for r in rows if r['Ortholog_TFIDF'] > 0),
                'enrichment_count': sum(
                    1 for r in rows if r['Enrichment_WeightedScore'] > 0
                ),
                'multi_source_count': sum(
                    1 for r in rows if r['Num_Sources'] >= 2
                ),
            },
            'config': {
                'topk_func_genes': self.topk_func_genes,
                'weight_direct': self.weight_direct,
                'weight_ortholog': self.weight_ortholog,
                'weight_enrichment': self.weight_enrichment,
                'feature_weights': self.feature_weights,
            },
            'timestamp': datetime.now().isoformat(),
        }

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    def to_dataframe(self, result: Dict) -> pd.DataFrame:
        """Convert disease rows to a DataFrame."""
        return pd.DataFrame(result['diseases'])

    def to_excel(self, result: Dict, path: str):
        """
        Export a full disease report to a multi-sheet Excel workbook.

        Sheets:
            1. All_Diseases_Ranked – every disease sorted by Combined_Relevance
            2. Direct              – query gene's own disease associations
            3. Ortholog            – ortholog's disease associations
            4. FuncCluster         – functional-cluster enrichment
            5. MultiSource         – diseases supported by ≥ 2 sources
            6. Functional_Matches  – top-N functionally similar genes
            7. Legend              – column descriptions
            8. Summary             – analysis metadata
        """
        df = self.to_dataframe(result)
        if df.empty:
            logger.warning("No diseases found – skipping Excel export")
            return

        with pd.ExcelWriter(path, engine='openpyxl') as writer:
            # Sheet 1
            df.to_excel(writer, sheet_name='All_Diseases_Ranked', index_label='Rank')

            # Sheet 2
            df_direct = df[df['Direct_TFIDF'] > 0].sort_values(
                'Direct_TFIDF', ascending=False,
            )
            df_direct.to_excel(writer, sheet_name='Direct', index_label='Rank')

            # Sheet 3
            df_ortho = df[df['Ortholog_TFIDF'] > 0].sort_values(
                'Ortholog_TFIDF', ascending=False,
            )
            df_ortho.to_excel(writer, sheet_name='Ortholog', index_label='Rank')

            # Sheet 4
            df_enrich = df[df['Enrichment_WeightedScore'] > 0].sort_values(
                'Enrichment_WeightedScore', ascending=False,
            )
            df_enrich.to_excel(writer, sheet_name='FuncCluster', index_label='Rank')

            # Sheet 5
            df_multi = df[df['Num_Sources'] >= 2].sort_values(
                'Combined_Relevance', ascending=False,
            )
            df_multi.to_excel(writer, sheet_name='MultiSource', index_label='Rank')

            # Sheet 6 – functional matches
            pd.DataFrame(result['functional_matches']).to_excel(
                writer, sheet_name='Functional_Matches', index=False,
            )

            # Sheet 7 – legend
            q = result['query']
            species_label = q['species']
            orth = result['ortholog']
            target_label = orth['species']
            legend = pd.DataFrame({
                'Column': [
                    'Disease_ID', 'Disease_Name', 'Ontology',
                    'Combined_Relevance', 'Num_Sources', 'Evidence_Sources',
                    'Direct_TFIDF', 'Direct_RawScore', 'Direct_EvidenceCount',
                    'Ortholog_TFIDF', 'Ortholog_RawScore',
                    'Ortholog_EvidenceCount', 'Ortholog_DataSources',
                    'Enrichment_WeightedScore', 'Enrichment_GeneCount',
                    'Enrichment_Genes',
                ],
                'Description': [
                    'Disease ontology ID (EFO/MONDO/Orphanet/HP …)',
                    'Disease name (Open Targets Platform)',
                    'Ontology source prefix',
                    f'{self.weight_direct}*Direct + '
                    f'{self.weight_ortholog}*Ortholog + '
                    f'{self.weight_enrichment}*Enrichment',
                    'Number of independent evidence sources (1-3)',
                    'Which sources support this disease',
                    f'{species_label} {q["symbol"]} TF-IDF weight (direct)',
                    f'{species_label} {q["symbol"]} raw evidence score',
                    f'{species_label} {q["symbol"]} evidence count',
                    f'{target_label} {orth["symbol"]} TF-IDF weight (ortholog)',
                    f'{target_label} {orth["symbol"]} raw evidence score',
                    f'{target_label} {orth["symbol"]} evidence count',
                    f'{target_label} {orth["symbol"]} data sources',
                    'Functional cluster weighted enrichment score',
                    'Number of cluster genes supporting this disease',
                    'Gene symbols in functional cluster',
                ],
            })
            legend.to_excel(writer, sheet_name='Legend', index=False)

            # Sheet 8 – summary
            s = result['summary']
            summary = pd.DataFrame({
                'Item': [
                    'Query gene', 'Query Ensembl ID', 'Query species',
                    'Ortholog gene', 'Ortholog Ensembl ID', 'Ortholog species',
                    'Top-N functional matches',
                    'Functional match genes',
                    'Total diseases', 'Direct diseases',
                    'Ortholog diseases', 'Enrichment diseases',
                    'Multi-source diseases (>=2)',
                    'Relevance formula',
                    'Timestamp',
                ],
                'Value': [
                    q['symbol'], q['gene_id'], q['species'],
                    orth['symbol'], orth['gene_id'], orth['species'],
                    str(self.topk_func_genes),
                    ', '.join(m['symbol'] for m in result['functional_matches']),
                    str(s['total_diseases']), str(s['direct_count']),
                    str(s['ortholog_count']), str(s['enrichment_count']),
                    str(s['multi_source_count']),
                    f'{self.weight_direct}*Direct_TFIDF + '
                    f'{self.weight_ortholog}*Ortholog_TFIDF + '
                    f'{self.weight_enrichment}*Enrichment',
                    result['timestamp'],
                ],
            })
            summary.to_excel(writer, sheet_name='Summary', index=False)

        # Auto-adjust column widths
        try:
            from openpyxl import load_workbook
            wb = load_workbook(path)
            for ws in wb.worksheets:
                for col in ws.columns:
                    letter = col[0].column_letter
                    max_len = max(
                        (len(str(cell.value or '')) for cell in col), default=0,
                    )
                    ws.column_dimensions[letter].width = min(max_len + 2, 55)
            wb.save(path)
        except Exception:
            pass  # non-critical

        logger.info(f"Excel report saved to {path}")

    def to_json(self, result: Dict, path: str):
        """Export disease report to JSON."""
        with open(path, 'w') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"JSON report saved to {path}")

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_species(gene_id: str) -> Optional[str]:
        if gene_id.startswith('ENSMUSG'):
            return 'mouse'
        if gene_id.startswith('ENSG'):
            return 'human'
        return None


# ---------------------------------------------------------------------------
# Convenience entry-point
# ---------------------------------------------------------------------------

def run_disease_report(
    query: str,
    config: Dict,
    output: Optional[str] = None,
    resolve_names: bool = True,
) -> Dict:
    """
    Run a disease report for one or more genes.

    Args:
        query: Semicolon-separated Ensembl gene IDs.
        config: Project configuration dict.
        output: If provided and ends with ``.xlsx`` or ``.json``, export there.
        resolve_names: Whether to resolve disease IDs via Open Targets API.

    Returns:
        A dict with ``per_gene`` (list of per-gene reports) and ``meta``.
    """
    from .io_schema import load_indexers
    from .build_sparse import load_sparse_matrices

    proc_dir = Path(config['paths']['proc_dir'])
    raw_dir = Path(config['paths']['raw_dir'])
    io_cfg = config['io']

    logger.info("Loading preprocessed data …")
    indexers = load_indexers(proc_dir / 'indexers.json')
    sparse_matrices = load_sparse_matrices(proc_dir)

    # Load raw disease CSVs for detail columns
    logger.info("Loading raw disease CSVs …")
    raw_h = pd.read_csv(raw_dir / io_cfg['human_dis'])
    raw_m = pd.read_csv(raw_dir / io_cfg['mouse_dis'])

    # Override resolve_names if config says otherwise
    dr_cfg = config.get('disease_report', {})
    if not resolve_names:
        dr_cfg['resolve_names'] = False
    config_copy = {**config, 'disease_report': {**dr_cfg, 'resolve_names': resolve_names}}

    reporter = DiseaseReporter(
        sparse_matrices=sparse_matrices,
        indexers=indexers,
        config=config_copy,
        raw_human_disease=raw_h,
        raw_mouse_disease=raw_m,
    )

    gene_ids = [g.strip() for g in query.split(';') if g.strip()]
    per_gene = []
    for gid in gene_ids:
        logger.info(f"Generating disease report for {gid} …")
        result = reporter.report(gid)
        per_gene.append(result)

    combined = {
        'per_gene': per_gene,
        'meta': {
            'query': query,
            'gene_count': len(gene_ids),
            'timestamp': datetime.now().isoformat(),
        },
    }

    # Export
    if output:
        out_path = Path(output)
        if out_path.suffix == '.xlsx':
            if len(per_gene) == 1:
                reporter.to_excel(per_gene[0], str(out_path))
            else:
                # Multiple genes → one file per gene
                for i, res in enumerate(per_gene):
                    gid = res['query']['gene_id']
                    p = out_path.parent / f"{out_path.stem}_{gid}{out_path.suffix}"
                    reporter.to_excel(res, str(p))
        elif out_path.suffix == '.json':
            reporter.to_json(combined, str(out_path))
        else:
            # Default to JSON
            reporter.to_json(combined, str(out_path))

    return combined
