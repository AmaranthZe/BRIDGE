"""Training loop for Cross-Species Gene Mapping.

Architecture: CPU HGT + GPU FGW (epoch-cached)
- HGT forward pass runs ONCE per epoch on CPU (8.3M edges, too large for GPU).
  Embeddings are cached and reused for all batches within the epoch.
- Fusion modules run on GPU each batch (with gradients for fusion params).
- Candidate selector and FGW run on GPU each batch.
- HGT parameters are updated via a separate gradient step every N epochs.

Performance rationale:
  Running HGT on CPU per-batch (~94 batches/epoch) would take ~15-45 min/epoch.
  Caching HGT per-epoch reduces this to one ~10-30s forward pass + fast GPU batches.
  Fusion/candidates/FGW still learn from every batch. HGT learns periodically.

FIXES from v1:
- Z_h/Z_m SVD features are pre-computed and passed to CostComputer (no .toarray())
- Global refresh properly merges old/new indices
- Bidirectional training: alternates mouse→human and human→mouse
"""

import gc
import logging
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from ..model.model import CrossSpeciesModel
from ..transport.costs import CostComputer, structural_cost
from ..transport.fgw import FGWSolver
from .memory_bank import MemoryBank
from .losses import compute_total_loss

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    def __init__(
        self,
        model: CrossSpeciesModel,
        graph_data,
        Z_h: np.ndarray,
        Z_m: np.ndarray,
        indexers: Dict,
        cfg,
        device: str = "cuda",
    ):
        self.graph_data = graph_data
        self.indexers = indexers
        self.cfg = cfg
        self.device = device

        tc = cfg.train
        fc = cfg.fgw

        self.epochs = tc.epochs
        self.batch_size = tc.batch_size
        self.clip_grad = tc.clip_grad

        # Split model: HGT on CPU, fusion/candidates on GPU
        self.model = model
        self.model.hgt = self.model.hgt.to("cpu")
        self.model.fusion_human = self.model.fusion_human.to(device)
        self.model.fusion_mouse = self.model.fusion_mouse.to(device)
        self.model.candidate_selector = self.model.candidate_selector.to(device)

        # FGW solver — use stronger tau_col during training to prevent hub formation
        train_tau_col = getattr(tc, 'train_tau_col', fc.tau_col)
        self.fgw_solver = FGWSolver(
            alpha=fc.alpha, epsilon=fc.epsilon,
            lambda_prior=fc.lambda_prior, lambda_memory=fc.lambda_memory,
            use_unbalanced=fc.use_unbalanced,
            tau_row=fc.tau_row, tau_col=train_tau_col,
            max_iter=fc.max_iter,
        )
        logger.info(f"Training tau_col={train_tau_col} (inference default={fc.tau_col})")

        # Anti-hub column entropy weight
        self.col_entropy_weight = fc.col_entropy_weight
        # Anti-isotropy covariance regularization weight
        self.cov_reg_weight = fc.cov_reg_weight
        # FGW-driven embedding alignment weight
        self.align_loss_weight = getattr(fc, 'align_loss_weight', 0.0)

        # Cost computer with pre-computed SVD features (FIX from v1)
        self.cost_computer = CostComputer(Z_h=Z_h, Z_m=Z_m, device=device)

        # SVD features on GPU for candidate retrieval (annotation-based recall)
        self.Z_h_gpu = self.cost_computer.Z_h  # already on device
        self.Z_m_gpu = self.cost_computer.Z_m

        # Per-modality gene features for fusion (from HGT buffers, move to GPU)
        self.mod_features = {}
        for key in ['Z_h_BP', 'Z_h_MF', 'Z_h_Disease', 'Z_m_BP', 'Z_m_MF', 'Z_m_Disease']:
            buf = getattr(self.model.hgt, key, None)
            if buf is not None:
                self.mod_features[key] = buf.to(device)
        if self.mod_features:
            logger.info(f"Loaded {len(self.mod_features)} per-modality gene feature tensors on GPU")
        else:
            logger.warning("No per-modality gene features found — fusion will use unified gene_emb")

        # Memory banks (bidirectional) — on CPU to save GPU memory
        N_h = len(indexers["human_genes"])
        N_m = len(indexers["mouse_genes"])
        self.memory_m2h = MemoryBank(N_m, N_h, tc.memory_momentum, "cpu")
        self.memory_h2m = MemoryBank(N_h, N_m, tc.memory_momentum, "cpu")

        # Optimizer
        self._init_optimizer(tc)

        # Graph data stays on CPU (for HGT)
        self._prepare_graph()

        # Pre-cache fusion edge indices on GPU (they never change)
        self._cache_fusion_edges()

        # Free graph_data reference — we only need edge_index_dict/edge_attr_dict now
        del self.graph_data
        gc.collect()

        # Logging
        self.log_dir = Path(cfg.paths.proc_dir) / "runs"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.best_loss = float("inf")

    def _init_optimizer(self, tc):
        qk_mult = tc.qk_lr_multiplier
        hgt_emb_mult = getattr(tc, 'hgt_emb_lr_multiplier', 0.1)

        hgt_emb_params, qk_params, other_params = [], [], []
        for name, p in self.model.named_parameters():
            if "hgt.embeddings" in name:
                hgt_emb_params.append(p)
            elif "W_Q" in name or "W_K" in name:
                qk_params.append(p)
            else:
                other_params.append(p)

        logger.info(
            f"Optimizer param groups: HGT embeddings ({len(hgt_emb_params)} tensors, "
            f"lr={tc.lr * hgt_emb_mult:.6f}), QK ({len(qk_params)}, "
            f"lr={tc.lr * qk_mult:.6f}), other ({len(other_params)}, lr={tc.lr:.6f})"
        )

        self.optimizer = torch.optim.AdamW([
            {"params": other_params, "lr": tc.lr},
            {"params": qk_params, "lr": tc.lr * qk_mult},
            {"params": hgt_emb_params, "lr": tc.lr * hgt_emb_mult},
        ], weight_decay=tc.weight_decay)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=tc.epochs, eta_min=tc.lr * 0.01,
        )

    def _prepare_graph(self):
        """Keep graph on CPU — HGT runs on CPU."""
        self.edge_index_dict = {}
        self.edge_attr_dict = {}
        for et in self.graph_data.edge_types:
            ed = self.graph_data[et]
            self.edge_index_dict[et] = ed.edge_index  # CPU
            if hasattr(ed, "edge_attr") and ed.edge_attr is not None:
                self.edge_attr_dict[et] = ed.edge_attr  # CPU

    def _cache_fusion_edges(self):
        """Pre-transfer fusion edge indices/attrs to GPU (they never change).

        Avoids redundant CPU→GPU transfers every batch (~133MB total, well within
        GPU memory budget).
        """
        dev = self.device

        def _cache(key):
            ei = self.edge_index_dict.get(key)
            ea = self.edge_attr_dict.get(key)
            return (ei.to(dev) if ei is not None else None,
                    ea.to(dev) if ea is not None else None)

        # Human gene → term edges
        self._ei_hb, self._ea_hb = _cache(("Human_Gene", "HG-BP", "BP"))
        self._ei_hf, self._ea_hf = _cache(("Human_Gene", "HG-MF", "MF"))
        self._ei_hd, self._ea_hd = _cache(("Human_Gene", "HG-D", "Disease"))

        # Mouse gene → term edges
        self._ei_mb, self._ea_mb = _cache(("Mouse_Gene", "MG-BP", "BP"))
        self._ei_mf_e, self._ea_mf_e = _cache(("Mouse_Gene", "MG-MF", "MF"))
        self._ei_md, self._ea_md = _cache(("Mouse_Gene", "MG-D", "Disease"))

        logger.info("Cached fusion edge indices on GPU")

    def _get_batches(self, n: int, shuffle: bool = True):
        indices = list(range(n))
        if shuffle:
            random.shuffle(indices)
        for i in range(0, n, self.batch_size):
            yield torch.tensor(indices[i:i + self.batch_size], dtype=torch.long, device=self.device)

    def _hgt_forward_cpu(self):
        """Run HGT on CPU, get all node embeddings."""
        return self.model.hgt(self.edge_index_dict, self.edge_attr_dict)

    def _fused_embeddings(self, emb_dict):
        """Apply fusion on GPU using pre-cached edge indices.

        HGT outputs may be on CPU (dict values). They're moved to GPU once.
        Fusion modules run on GPU with gradients for their own parameters.
        """
        dev = self.device

        E_h_raw = emb_dict["Human_Gene"].to(dev)
        E_m_raw = emb_dict["Mouse_Gene"].to(dev)
        E_bp = emb_dict["BP"].to(dev)
        E_mf = emb_dict["MF"].to(dev)
        E_dis = emb_dict["Disease"].to(dev)

        # Human fusion (using cached GPU edge indices + per-modality gene features)
        E_h, _, ent_h = self.model.fusion_human(
            E_h_raw, E_bp, E_mf, E_dis,
            ei_bp=self._ei_hb, ei_mf=self._ei_hf, ei_disease=self._ei_hd,
            ea_bp=self._ea_hb, ea_mf=self._ea_hf, ea_disease=self._ea_hd,
            bp_gene_feat=self.mod_features.get('Z_h_BP'),
            mf_gene_feat=self.mod_features.get('Z_h_MF'),
            dis_gene_feat=self.mod_features.get('Z_h_Disease'),
        )

        # Mouse fusion (using cached GPU edge indices + per-modality gene features)
        E_m, _, ent_m = self.model.fusion_mouse(
            E_m_raw, E_bp, E_mf, E_dis,
            ei_bp=self._ei_mb, ei_mf=self._ei_mf_e, ei_disease=self._ei_md,
            ea_bp=self._ea_mb, ea_mf=self._ea_mf_e, ea_disease=self._ea_md,
            bp_gene_feat=self.mod_features.get('Z_m_BP'),
            mf_gene_feat=self.mod_features.get('Z_m_MF'),
            dis_gene_feat=self.mod_features.get('Z_m_Disease'),
        )

        # Center embeddings (remove dominant direction to break isotropy)
        E_h = E_h - E_h.mean(dim=0, keepdim=True)
        E_m = E_m - E_m.mean(dim=0, keepdim=True)

        return E_h, E_m, ent_h + ent_m

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.hgt.train()
        self.model.fusion_human.train()
        self.model.fusion_mouse.train()
        self.model.candidate_selector.train()

        total_loss = 0.0
        n_batches = 0
        accum = {}

        N_m = len(self.indexers["mouse_genes"])
        N_h = len(self.indexers["human_genes"])

        # Alternate direction each epoch
        if epoch % 2 == 1:
            batches = self._get_batches(N_m)
            direction = "mouse_to_human"
        else:
            batches = self._get_batches(N_h)
            direction = "human_to_mouse"

        # ── Memory: disabled for first training run ──
        # The memory bank's KL regularization is unstable when embeddings shift
        # between epochs — stored coupling patterns diverge from current T,
        # causing KL explosion. Disable until the model stabilizes.
        self._use_memory = False
        self.fgw_solver.lambda_memory = 0.0

        # ── Phase 1: Cache HGT embeddings (one CPU forward pass, no grad) ──
        with torch.no_grad():
            emb_cache = self._hgt_forward_cpu()
        logger.info(f"Epoch {epoch}: HGT forward done, starting batches ({direction})")

        # ── Phase 2: Batch training (fusion/candidates/FGW learn per batch) ──
        for I_batch in tqdm(batches, desc=f"Epoch {epoch} ({direction})"):
            self.optimizer.zero_grad()
            loss, losses = self._train_batch(I_batch, direction, emb_cache)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1
            for k, v in losses.items():
                accum[k] = accum.get(k, 0) + (v.item() if torch.is_tensor(v) else v)

        self.scheduler.step()

        # ── Phase 3: HGT gradient step ──
        # Disabled when system RAM is constrained — the HGT gradient step runs
        # full-graph forward WITH gradient on CPU, which requires ~28GB RAM.
        # With skip_gate α≈0.8, HGT contributes only ~20% of final embeddings;
        # fusion/candidates/FGW still learn from every batch in Phase 2.
        hgt_grad_start = getattr(self.cfg.train, 'hgt_grad_start_epoch', 0)
        if epoch > hgt_grad_start:
            n_hgt_batches = min(4, max(1, epoch // 5 + 1))  # ramp up: 1→2→3→4
            self._hgt_gradient_step(direction, n_batches=n_hgt_batches)

            # Memory refresh (less frequently, just for memory bank data)
            if epoch % self.cfg.train.global_refresh_every == 0:
                with torch.no_grad():
                    emb = self._hgt_forward_cpu()
                    E_h, E_m, _ = self._fused_embeddings(emb)
                self.memory_m2h.global_refresh(self.model, E_m.cpu(), E_h.cpu())
                self.memory_h2m.global_refresh(self.model, E_h.cpu(), E_m.cpu())

        avg = {k: v / n_batches for k, v in accum.items()}
        avg["total"] = total_loss / n_batches
        return avg

    def _train_batch(self, I_batch, direction, emb_cache) -> Tuple[torch.Tensor, Dict]:
        """Train one batch using cached HGT embeddings.

        Fusion modules run on GPU with gradients (their parameters learn).
        HGT embeddings are detached (cached from Phase 1) — HGT learns via
        periodic gradient steps in Phase 3.
        """
        # Fusion on GPU (fusion params get gradients, HGT embeddings are detached)
        E_h, E_m, entropy_loss = self._fused_embeddings(emb_cache)

        if direction == "mouse_to_human":
            E_src, E_tgt = E_m, E_h
            Z_src, Z_tgt = self.Z_m_gpu, self.Z_h_gpu
            memory = self.memory_m2h
            Z_src_mods = {
                'bp': self.mod_features.get('Z_m_BP'),
                'mf': self.mod_features.get('Z_m_MF'),
                'disease': self.mod_features.get('Z_m_Disease'),
            }
            Z_tgt_mods = {
                'bp': self.mod_features.get('Z_h_BP'),
                'mf': self.mod_features.get('Z_h_MF'),
                'disease': self.mod_features.get('Z_h_Disease'),
            }
        else:
            E_src, E_tgt = E_h, E_m
            Z_src, Z_tgt = self.Z_h_gpu, self.Z_m_gpu
            memory = self.memory_h2m
            Z_src_mods = {
                'bp': self.mod_features.get('Z_h_BP'),
                'mf': self.mod_features.get('Z_h_MF'),
                'disease': self.mod_features.get('Z_h_Disease'),
            }
            Z_tgt_mods = {
                'bp': self.mod_features.get('Z_m_BP'),
                'mf': self.mod_features.get('Z_m_MF'),
                'disease': self.mod_features.get('Z_m_Disease'),
            }

        E_batch = E_src[I_batch]

        # Candidate selection (GPU) — per-modality SVD ensemble retrieval
        J_cand, soft_mask, Pi0, row_mask = self.model.candidate_selector(
            E_source=E_src, E_target=E_tgt, source_indices=I_batch,
            Z_source=Z_src, Z_target=Z_tgt,
            Z_source_mods=Z_src_mods, Z_target_mods=Z_tgt_mods)

        # Costs (GPU)
        costs = self.cost_computer.compute(
            E_source=E_batch,
            E_target=E_tgt[J_cand],
            source_indices=I_batch,
            target_indices=J_cand,
            direction=direction,
        )

        M_masked = costs["M"] * soft_mask

        # Memory prior: skip entirely during warmup to avoid KL explosion
        if self._use_memory:
            mem_prior = memory.get_prior(I_batch.cpu(), J_cand.cpu()).to(self.device)
        else:
            mem_prior = None

        # FGW solve (GPU)
        T, _ = self.fgw_solver.solve(
            C_s=costs["C_s"], C_t=costs["C_t"], M=M_masked,
            Pi0=Pi0, memory_prior=mem_prior, mask=row_mask)

        # Update memory (CPU) — always update so it accumulates data
        memory.update(I_batch.cpu(), J_cand.cpu(), T.cpu())

        # Loss (GPU) — include column entropy for anti-hub regularization
        fgw_losses = self.fgw_solver.compute_loss(
            C_s=costs["C_s"], C_t=costs["C_t"], M=M_masked,
            T=T, Pi0=Pi0, memory_prior=mem_prior)
        losses = compute_total_loss(fgw_losses, entropy_loss,
                                    T=T, col_entropy_weight=self.col_entropy_weight,
                                    E_h=E_h, E_m=E_m,
                                    cov_reg_weight=self.cov_reg_weight,
                                    align_loss_weight=self.align_loss_weight,
                                    E_source_batch=E_batch,
                                    E_target_cand=E_tgt[J_cand])

        return losses["total"], losses

    def _hgt_gradient_step(self, direction, n_batches=1):
        """Full-gradient step to update HGT parameters.

        Runs HGT forward WITH gradient on CPU, then fusion+candidates+FGW on GPU
        for random batches. Backward flows: GPU loss → GPU fusion → CPU HGT.
        Multiple batches per step give HGT broader coverage of genes.
        """
        logger.info(f"Running HGT gradient step ({n_batches} batches)...")

        N_m = len(self.indexers["mouse_genes"])
        N_h = len(self.indexers["human_genes"])

        total_loss = 0.0
        for b in range(n_batches):
            self.optimizer.zero_grad()

            # HGT forward WITH gradient (CPU, mixed-device autograd)
            emb = self._hgt_forward_cpu()
            E_h, E_m, entropy_loss = self._fused_embeddings(emb)

            if direction == "mouse_to_human":
                E_src, E_tgt = E_m, E_h
                Z_src, Z_tgt = self.Z_m_gpu, self.Z_h_gpu
                memory = self.memory_m2h
                N_src = N_m
                Z_src_mods = {
                    'bp': self.mod_features.get('Z_m_BP'),
                    'mf': self.mod_features.get('Z_m_MF'),
                    'disease': self.mod_features.get('Z_m_Disease'),
                }
                Z_tgt_mods = {
                    'bp': self.mod_features.get('Z_h_BP'),
                    'mf': self.mod_features.get('Z_h_MF'),
                    'disease': self.mod_features.get('Z_h_Disease'),
                }
            else:
                E_src, E_tgt = E_h, E_m
                Z_src, Z_tgt = self.Z_h_gpu, self.Z_m_gpu
                memory = self.memory_h2m
                N_src = N_h
                Z_src_mods = {
                    'bp': self.mod_features.get('Z_h_BP'),
                    'mf': self.mod_features.get('Z_h_MF'),
                    'disease': self.mod_features.get('Z_h_Disease'),
                }
                Z_tgt_mods = {
                    'bp': self.mod_features.get('Z_m_BP'),
                    'mf': self.mod_features.get('Z_m_MF'),
                    'disease': self.mod_features.get('Z_m_Disease'),
                }

            # Random batch for HGT update
            I_batch = torch.randperm(N_src, device=self.device)[:self.batch_size]
            E_batch = E_src[I_batch]

            J_cand, soft_mask, Pi0, row_mask = self.model.candidate_selector(
                E_source=E_src, E_target=E_tgt, source_indices=I_batch,
                Z_source=Z_src, Z_target=Z_tgt,
                Z_source_mods=Z_src_mods, Z_target_mods=Z_tgt_mods)

            costs = self.cost_computer.compute(
                E_source=E_batch, E_target=E_tgt[J_cand],
                source_indices=I_batch, target_indices=J_cand,
                direction=direction)

            M_masked = costs["M"] * soft_mask
            if self._use_memory:
                mem_prior = memory.get_prior(I_batch.cpu(), J_cand.cpu()).to(self.device)
            else:
                mem_prior = None

            T, _ = self.fgw_solver.solve(
                C_s=costs["C_s"], C_t=costs["C_t"], M=M_masked,
                Pi0=Pi0, memory_prior=mem_prior, mask=row_mask)

            fgw_losses = self.fgw_solver.compute_loss(
                C_s=costs["C_s"], C_t=costs["C_t"], M=M_masked,
                T=T, Pi0=Pi0, memory_prior=mem_prior)
            losses = compute_total_loss(fgw_losses, entropy_loss,
                                        T=T, col_entropy_weight=self.col_entropy_weight,
                                        E_h=E_h, E_m=E_m,
                                        cov_reg_weight=self.cov_reg_weight)

            losses["total"].backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
            total_loss += losses["total"].item()

        logger.info(f"HGT gradient step loss: {total_loss / n_batches:.4f}")

    def train(self):
        logger.info(f"Starting training for {self.epochs} epochs")
        logger.info(f"Architecture: HGT on CPU (cached per epoch), "
                     f"fusion/candidates/FGW on {self.device}")
        for epoch in range(1, self.epochs + 1):
            losses = self.train_epoch(epoch)
            log_str = f"Epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" for k, v in losses.items())
            logger.info(log_str)

            if epoch % self.cfg.train.save_every == 0:
                self.save_checkpoint(epoch, losses["total"])
            if losses["total"] < self.best_loss:
                self.best_loss = losses["total"]
                self.save_checkpoint(epoch, losses["total"], is_best=True)

        logger.info(f"Training complete. Best loss: {self.best_loss:.4f}")

    def save_checkpoint(self, epoch, loss, is_best=False):
        # Collect full model state (CPU + GPU parts)
        state = {}
        for k, v in self.model.state_dict().items():
            state[k] = v.cpu()

        ckpt = {
            "epoch": epoch,
            "model_state_dict": state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "loss": loss,
        }
        path = self.log_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(ckpt, path)
        if is_best:
            torch.save(ckpt, self.log_dir / "best_model.pt")
            logger.info(f"Saved best model (epoch {epoch}, loss {loss:.4f})")

    def load_checkpoint(self, path: str):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
        # Re-apply device split
        self.model.hgt = self.model.hgt.to("cpu")
        self.model.fusion_human = self.model.fusion_human.to(self.device)
        self.model.fusion_mouse = self.model.fusion_mouse.to(self.device)
        self.model.candidate_selector = self.model.candidate_selector.to(self.device)
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        return ckpt["epoch"], ckpt["loss"]
