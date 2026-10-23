"""Quick test: SVD-init embedding quality."""
import sys, pickle, numpy as np, torch
sys.path.insert(0, '.')
import torch.nn.functional as F
from v2_redesign.config import Config
from v2_redesign.data.sparse import load_sparse_matrices
from v2_redesign.data.svd import load_svd_artifacts
from v2_redesign.model.model import create_model

cfg = Config()
proc = 'data/processed'
with open(f'{proc}/indexers.pkl', 'rb') as f:
    idx = pickle.load(f)
Z_h, Z_m, _, _ = load_svd_artifacts(proc)
sm = load_sparse_matrices(proc)
nc = {
    'Human_Gene': len(idx['human_genes']),
    'Mouse_Gene': len(idx['mouse_genes']),
    'BP': len(idx['bp_terms']),
    'MF': len(idx['mf_terms']),
    'Disease': len(idx['diseases']),
}
model = create_model(cfg, nc)
model.hgt.init_from_svd(Z_h, Z_m, sm)
E_h = F.normalize(model.hgt.embeddings['Human_Gene'].weight.data, dim=1)
E_m = F.normalize(model.hgt.embeddings['Mouse_Gene'].weight.data, dim=1)

test_pairs = [
    ('ENSMUSG00000059552', 'Trp53', 'ENSG00000141510', 'TP53'),
    ('ENSMUSG00000017146', 'Brca1', 'ENSG00000012048', 'BRCA1'),
    ('ENSMUSG00000020122', 'Egfr', 'ENSG00000146648', 'EGFR'),
]

for m_id, m_sym, h_id, h_sym in test_pairs:
    m_i = idx['mouse_genes'][m_id]
    h_i = idx['human_genes'][h_id]
    sims = (E_m[m_i:m_i+1] @ E_h.t()).squeeze()
    vals, inds = torch.topk(sims, 5)
    h_rank = (sims > sims[h_i]).sum().item() + 1
    print(f'\n{m_sym} -> {h_sym}: rank={h_rank}, cos={sims[h_i]:.4f}')
    print(f'  Top-5:')
    for r in range(5):
        gid = idx['human_genes_inv'][inds[r].item()]
        sym = idx['gene_symbols'].get(gid, 'N/A')
        print(f'    #{r+1}: {sym} ({gid}) cos={vals[r]:.4f}')
