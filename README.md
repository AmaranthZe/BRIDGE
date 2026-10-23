# BRIDGE

**Disease-informed functional alignment of human and mouse genes.**

Given a human disease gene, BRIDGE ranks the mouse genes most likely to reproduce
its function — explicitly including non-orthologous matches (paralog compensation,
pathway substitution, multi-gene models). A one-to-one ortholog table is the
**baseline to beat**, not the target.

The method combines three components:

1. A **unified heterogeneous graph** over five node types (Human_Gene, Mouse_Gene,
   biological-process GO terms, molecular-function GO terms, Disease) with twelve
   bipartite edge types. Human and mouse genes connect to the *same* pool of GO /
   Disease nodes, which is what aligns the two embedding spaces.
2. A **Heterogeneous Graph Transformer (HGT)** with per-modality fusion, followed by
   **Fused Gromov-Wasserstein (FGW) optimal transport** to produce a cross-species
   transport plan.
3. A **weighted Reciprocal Rank Fusion (RRF)** layer that fuses 22 ranking strategies
   (per-modality SVD cosine, 2-hop annotation overlap, Reactome pathways, STRING PPI,
   FGW transport quality, and a min-rank aggregator) into the final ranking.

## Repository layout

```
v2_redesign/      Production model: preprocessing, HGT, fusion, candidate
                  selection, FGW transport, training, inference, CLI.
  data/           CSV -> sparse matrices -> SVD -> heterogeneous graph.
  model/          HGT, GeneModuleFusion, SoftCandidateSelector.
  transport/      Sinkhorn / FGW optimal transport.
  training/       Trainer, losses, memory bank.
  inference/      Inferencer and ranked readout.
  evaluation/     Metrics, baselines, evaluator.
v3_crosshop/      Experimental variant: v2 plus direct cross-species gene-gene
                  edges (18 edge types).
benchmarks/       Benchmark suite (literature analogues, phenotype transfer,
                  essentiality conservation, genome-wide HMD ranking, external
                  baselines, functional Jaccard, Ortholog-Override Criterion)
                  and the four evaluated model variants M1-M4.
scripts/          Statistical analyses (Kendall tau, Nemenyi, negative controls)
                  and FGW grid searches.
configs/          Training / preprocessing YAML configurations.
run_*.py          Top-level driver scripts (training, evaluation, RRF validation,
                  ablations).
src/              Legacy v1 prototype (self-contained HGT + FGW). Superseded by
                  v2_redesign/; retained for reference and covered by tests/.
tests/            Unit tests for the v1 prototype.
```

## Model variants

| Variant | Backbone | Ensembler | Notes |
|---------|----------|-----------|-------|
| M1 | v2 HGT (12 edges) | FGW only | |
| M2 | v3 cross-hop (18 edges) | FGW only | |
| M3 | v2 HGT | + 22-strategy RRF | |
| M4 (BRIDGE) | v3 cross-hop | + 22-strategy RRF | final system |

## Installation

```bash
conda env create -f environment.yml
conda activate csgm
pip install -e .
```

Requires Python >= 3.10, PyTorch >= 2.3 with CUDA, and PyTorch Geometric >= 2.5.
A GPU with >= 16 GB is recommended for training.

## Data

Raw inputs are expected under `dataset/` (not distributed with this repository):

| File | Contents |
|------|----------|
| `human_GOBP_ensembl.csv`, `human_GOMF_ensembl.csv` | human gene -> GO BP / MF annotations (`ENSGID, GeneName, GOTerm`) |
| `mouse_GOBP_ensembl.csv`, `mouse_GOMF_ensembl.csv` | mouse gene -> GO BP / MF annotations (`ENSMUSG*`) |
| `HumanDiseaseGene.csv`, `MouseDiseaseGene.csv` | Open Targets gene-disease associations |
| `reactome_ensembl_pathway.tsv`, `string_mouse.txt` | RRF strategy inputs |
| `MGI_GenePheno.rpt`, `HMD_HumanPhenotype.rpt` | phenotype-transfer ground truth |
| `external/` | DepMap, IMPC viability, Ensembl orthologs, PhenoDigm, drug targets |

Preprocessing rules: drop GO root nodes (`GO:0008150`, `GO:0003674`, `GO:0005575`);
TF-IDF column weighting with row L2 normalization; winsorize disease scores
(q = 0.995) then scale to [0, 1]; species is always decided by ID prefix
(`ENSG` vs `ENSMUSG`), never by column name. Derived artifacts are written to
`data/processed/`.

## Usage

```bash
# Preprocess: CSV -> sparse matrices -> SVD -> heterogeneous graph
python -m v2_redesign.cli preprocess --config configs/default.yaml

# Train the production model
python -m v2_redesign.cli train --device cuda

# Inference (semicolon-separated IDs; species may be mixed)
python -m v2_redesign.cli infer "ENSG00000141510;ENSMUSG00000059552" \
    --checkpoint data/processed/runs/retrain_v6/best_model.pt

# Train the v3 cross-hop variant (reuses v2 preprocessing)
python v3_crosshop/run_train_v3.py --resume-v2 data/processed/runs/retrain_v6/best_model.pt

# 22-strategy weighted RRF (main ranking pipeline)
python run_validation_final_rrf.py

# Benchmarks (variant x benchmark matrix; --skip-ppi avoids the 650 MB STRING load)
python benchmarks/run_benchmarks.py --variant all --benchmark all

# Unit tests (cover the legacy v1 prototype in src/)
python -m pytest tests -q
```

## License

MIT.
