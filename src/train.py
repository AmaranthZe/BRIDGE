"""
Training Loop for Cross-Species Gene Mapping.

This module implements:
- Mini-batch training with memory bank
- Periodic global refresh
- Logging and checkpointing
"""

import logging
import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from .model import CrossSpeciesModel, create_model
from .memory_bank import MemoryBank
from .costs import CostComputer, structural_cost
from .fgw import compute_fgw_loss

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Trainer:
    """
    Trainer for Cross-Species Gene Mapping Model.
    """

    def __init__(
        self,
        model: CrossSpeciesModel,
        graph_data,  # HeteroData
        sparse_matrices: Dict,
        indexers: Dict,
        config: Dict,
        device: str = 'cuda',
    ):
        self.model = model.to(device)
        self.graph_data = graph_data
        self.sparse_matrices = sparse_matrices
        self.indexers = indexers
        self.config = config
        self.device = device

        # Training config
        train_cfg = config['train']
        self.epochs = train_cfg['epochs']
        self.batch_size = train_cfg['batch_size']
        self.lr = train_cfg['lr']
        self.weight_decay = train_cfg['weight_decay']
        self.use_amp = train_cfg['amp']
        self.clip_grad = train_cfg['clip_grad']
        self.val_every = train_cfg.get('val_every', 1)
        self.save_every = train_cfg.get('save_every', 5)

        # Memory bank config
        cand_cfg = config['candidates']
        self.global_refresh_every = cand_cfg['global_refresh_every']
        self.memory_momentum = cand_cfg['memory_momentum']

        # FGW config
        fgw_cfg = config['fgw']
        self.feature_weights = fgw_cfg['feature_weights']

        # Initialize components
        self._init_optimizer()
        self._init_memory_bank()
        self._init_cost_computer()
        self._prepare_graph_data()

        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Logging
        log_cfg = config.get('logging', {})
        self.log_dir = Path(log_cfg.get('log_dir', 'runs/default'))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Best model tracking
        self.best_loss = float('inf')
        self.best_epoch = 0

    def _init_optimizer(self):
        """Initialize optimizer with separate learning rate for Q-K projections."""
        # Get Q-K projection learning rate multiplier from config
        qk_lr_mult = self.config.get('candidates', {}).get('qk_lr_multiplier', 1.0)

        # Separate Q-K projection parameters from other parameters
        qk_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if 'candidate_selector.W_Q' in name or 'candidate_selector.W_K' in name:
                qk_params.append(param)
                logger.info(f"Q-K param with {qk_lr_mult}x LR: {name}")
            else:
                other_params.append(param)

        # Create parameter groups with different learning rates
        param_groups = [
            {'params': other_params, 'lr': self.lr},
            {'params': qk_params, 'lr': self.lr * qk_lr_mult},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=self.lr * 0.01,
        )

    def _init_memory_bank(self):
        """Initialize memory bank."""
        N_m = len(self.indexers['mouse_genes'])
        N_h = len(self.indexers['human_genes'])

        self.memory_bank = MemoryBank(
            N_source=N_m,
            N_target=N_h,
            momentum=self.memory_momentum,
            device=self.device,
            sparse=True,
        )

    def _init_cost_computer(self):
        """Initialize cost computer."""
        # Prepare sparse matrices for cost computation
        W_mouse = {
            'bp': self.sparse_matrices['W_mB'],
            'mf': self.sparse_matrices['W_mF'],
            'disease': self.sparse_matrices['W_mD'],
        }
        W_human = {
            'bp': self.sparse_matrices['W_hB'],
            'mf': self.sparse_matrices['W_hF'],
            'disease': self.sparse_matrices['W_hD'],
        }

        self.cost_computer = CostComputer(
            W_source=W_mouse,
            W_target=W_human,
            feature_weights=self.feature_weights,
            device=self.device,
        )

    def _prepare_graph_data(self):
        """Prepare graph data for training."""
        self.edge_index_dict = {}
        self.edge_attr_dict = {}

        for edge_type in self.graph_data.edge_types:
            edge_data = self.graph_data[edge_type]
            self.edge_index_dict[edge_type] = edge_data.edge_index.to(self.device)
            if hasattr(edge_data, 'edge_attr') and edge_data.edge_attr is not None:
                self.edge_attr_dict[edge_type] = edge_data.edge_attr.to(self.device)

    def _get_batches(self, shuffle: bool = True):
        """Generate batch indices for mouse genes."""
        N_m = len(self.indexers['mouse_genes'])
        indices = list(range(N_m))

        if shuffle:
            random.shuffle(indices)

        for i in range(0, N_m, self.batch_size):
            batch = indices[i:i + self.batch_size]
            yield torch.tensor(batch, dtype=torch.long, device=self.device)

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        loss_components = {}

        pbar = tqdm(self._get_batches(), desc=f"Epoch {epoch}")

        for I_batch in pbar:
            self.optimizer.zero_grad()

            # Forward pass with optional AMP
            if self.use_amp:
                with autocast():
                    loss, batch_losses = self._train_batch(I_batch)
            else:
                loss, batch_losses = self._train_batch(I_batch)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Accumulate loss components
            for k, v in batch_losses.items():
                if k not in loss_components:
                    loss_components[k] = 0
                loss_components[k] += v.item() if torch.is_tensor(v) else v

            pbar.set_postfix({'loss': loss.item()})

        # Average losses
        avg_loss = total_loss / num_batches
        for k in loss_components:
            loss_components[k] /= num_batches

        # Step scheduler
        self.scheduler.step()

        # Global refresh
        if epoch % self.global_refresh_every == 0 and epoch > 0:
            self._global_refresh()

        return {'total': avg_loss, **loss_components}

    def _train_batch(
        self,
        I_batch: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """Train on a single batch."""
        B = len(I_batch)

        # Get all embeddings (forward pass through model)
        E_h, E_m, entropy_loss = self.model.get_gene_embeddings(
            self.edge_index_dict,
            self.edge_attr_dict,
        )

        # Get batch embeddings
        E_m_batch = E_m[I_batch]

        # Soft candidate selection (now returns per-query row_mask)
        J_cand, soft_mask, Pi0, row_mask, _ = self.model.candidate_selector(
            E_source=E_m,
            E_target=E_h,
            source_indices=I_batch,
        )

        # Compute structural costs only for batch and candidates (memory efficient)
        C_s, _ = structural_cost(E_m_batch, E_m_batch)  # [B, B]
        E_h_cand = E_h[J_cand]
        _, C_t = structural_cost(E_h_cand, E_h_cand)  # [J, J]

        # Compute feature cost
        M = self.cost_computer.compute_all(
            E_source=E_m_batch,
            E_target=E_h_cand,
            source_indices=I_batch,
            target_indices=J_cand,
        )['M']

        # Apply soft mask
        M_masked = M * soft_mask

        # Get memory prior
        memory_prior = self.memory_bank.get_prior(I_batch, J_cand)

        # Solve FGW with per-query row mask (A-fix #1)
        T, fgw_info = self.model.fgw_solver.solve(
            C_s=C_s,
            C_t=C_t,
            M=M_masked,
            Pi0=Pi0,
            memory_prior=memory_prior,
            mask=row_mask,
        )

        # Update memory bank
        self.memory_bank.update(I_batch, J_cand, T)

        # Compute losses
        losses = self.model.fgw_solver.compute_loss(
            C_s=C_s,
            C_t=C_t,
            M=M_masked,
            T=T,
            Pi0=Pi0,
            memory_prior=memory_prior,
        )

        # Add entropy loss
        losses['fusion_entropy'] = entropy_loss
        losses['total'] = losses['total'] + entropy_loss

        return losses['total'], losses

    def _global_refresh(self):
        """Perform global memory refresh."""
        logger.info("Performing global memory refresh...")

        with torch.no_grad():
            # Get embeddings
            E_h, E_m, _ = self.model.get_gene_embeddings(
                self.edge_index_dict,
                self.edge_attr_dict,
            )

            # Compute global coupling (simplified: use co-attention scores)
            # Process in chunks to avoid OOM
            N_m = E_m.shape[0]
            N_h = E_h.shape[0]
            chunk_size = 1000

            K = self.model.candidate_selector.W_K(E_h)  # [N_h, D]

            for start in range(0, N_m, chunk_size):
                end = min(start + chunk_size, N_m)
                E_m_chunk = E_m[start:end]

                Q = self.model.candidate_selector.W_Q(E_m_chunk)  # [chunk, D]
                A = torch.matmul(Q, K.t()) / (self.model.out_dim ** 0.5)  # [chunk, N_h]

                # Softmax to get coupling
                T_chunk = torch.softmax(A, dim=1)

                # Get top-k per row to update memory bank
                topk = min(256, N_h)
                topk_vals, topk_idx = torch.topk(T_chunk, topk, dim=1)

                # Update memory bank for this chunk
                for i in range(end - start):
                    src_idx = start + i
                    if self.memory_bank.bank_indices[src_idx] is None:
                        self.memory_bank.bank_indices[src_idx] = topk_idx[i]
                        self.memory_bank.bank_values[src_idx] = topk_vals[i]
                    else:
                        # Blend old and new
                        old_vals = self.memory_bank.bank_values[src_idx]
                        old_idx = self.memory_bank.bank_indices[src_idx]
                        self.memory_bank.bank_values[src_idx] = 0.5 * old_vals + 0.5 * topk_vals[i]
                        self.memory_bank.bank_indices[src_idx] = topk_idx[i]

        logger.info("Global refresh complete")

    def train(self):
        """Full training loop."""
        logger.info(f"Starting training for {self.epochs} epochs")

        for epoch in range(1, self.epochs + 1):
            losses = self.train_epoch(epoch)

            # Log
            log_str = f"Epoch {epoch}: "
            log_str += ", ".join([f"{k}={v:.4f}" for k, v in losses.items()])
            logger.info(log_str)

            # Save checkpoint
            if epoch % self.save_every == 0:
                self.save_checkpoint(epoch, losses['total'])

            # Track best model
            if losses['total'] < self.best_loss:
                self.best_loss = losses['total']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, losses['total'], is_best=True)

        logger.info(f"Training complete. Best loss: {self.best_loss:.4f} at epoch {self.best_epoch}")

    def save_checkpoint(self, epoch: int, loss: float, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'loss': loss,
            'config': self.config,
        }

        # Save regular checkpoint
        path = self.log_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

        # Save best model
        if is_best:
            best_path = self.log_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model to {best_path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        logger.info(f"Loaded checkpoint from {path}")
        return checkpoint['epoch'], checkpoint['loss']


def train_model(config: Dict, device: str = 'cuda'):
    """
    Main training function.

    Args:
        config: Configuration dictionary
        device: Device to use
    """
    from .io_schema import load_all_data, load_indexers, save_indexers
    from .build_sparse import (
        build_all_sparse_matrices, load_sparse_matrices, save_sparse_matrices,
        build_unified_svd, save_svd_artifacts, load_svd_artifacts,
    )
    from .unified_graph import build_unified_graph, load_unified_graph, save_unified_graph

    set_seed(config['train']['seed'])

    proc_dir = Path(config['paths']['proc_dir'])
    proc_dir.mkdir(parents=True, exist_ok=True)

    # Load or build preprocessed data
    indexers_path = proc_dir / 'indexers.json'
    graph_path = proc_dir / 'unified_graph.pt'

    if indexers_path.exists() and graph_path.exists():
        logger.info("Loading preprocessed data...")
        indexers = load_indexers(indexers_path)
        sparse_matrices = load_sparse_matrices(proc_dir)
        graph_data = load_unified_graph(graph_path)
    else:
        logger.info("Preprocessing data...")
        dataframes, indexers = load_all_data(config)
        sparse_matrices = build_all_sparse_matrices(dataframes, indexers, config)
        graph_data = build_unified_graph(sparse_matrices, indexers)

        # Save
        save_indexers(indexers, indexers_path)
        save_sparse_matrices(sparse_matrices, proc_dir)
        save_unified_graph(graph_data, graph_path)

    # Build or load unified SVD (A-fix #4)
    z_h_path = proc_dir / 'Z_h.npy'
    if z_h_path.exists():
        logger.info("Loading unified SVD artifacts...")
        Z_h, Z_m, svd_unified, svd_split_info = load_svd_artifacts(proc_dir)
    else:
        logger.info("Building unified SVD...")
        svd_cfg = config.get('preproc', {})
        Z_h, Z_m, svd_unified, svd_split_info = build_unified_svd(
            matrices=sparse_matrices,
            svd_dim=config.get('model', {}).get('svd_dim', 128),
            heldout_ratio=svd_cfg.get('heldout_ratio', 0.1),
            seed=config['train']['seed'],
        )
        save_svd_artifacts(Z_h, Z_m, svd_unified, svd_split_info, proc_dir)

    # Create model
    node_counts = {
        'Human_Gene': len(indexers['human_genes']),
        'Mouse_Gene': len(indexers['mouse_genes']),
        'BP': len(indexers['bp_terms']),
        'MF': len(indexers['mf_terms']),
        'Disease': len(indexers['diseases']),
    }

    model = create_model(config, node_counts)
    logger.info(f"Created model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create trainer
    trainer = Trainer(
        model=model,
        graph_data=graph_data,
        sparse_matrices=sparse_matrices,
        indexers=indexers,
        config=config,
        device=device,
    )

    # Train
    trainer.train()

    return trainer
