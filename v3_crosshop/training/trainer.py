"""v3 Trainer: extends v2 Trainer for cross-hop graph.

The only behavioral difference from v2:
  - Uses CrossSpeciesModelV3 (18-edge-type HGT)
  - Can load a v2 checkpoint for warm-starting (strict=False)
  - All cross-hop edges stay on CPU for HGT (automatic — v2's _prepare_graph()
    already iterates over all edge_types from graph_data)
  - Fusion caching is unchanged (only gene-term edges)
"""

import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from v2_redesign.training.trainer import Trainer as V2Trainer

logger = logging.getLogger(__name__)


class TrainerV3(V2Trainer):
    """v3 Trainer: HGT now has 18 edge types but training loop is unchanged.

    The v2 Trainer's _prepare_graph() iterates over graph_data.edge_types,
    so it automatically picks up all 18 edge types for HGT.
    The v2 _cache_fusion_edges() only caches the 6 gene-term types by
    their specific keys, so cross-species edges stay on CPU (correct).
    """

    def load_checkpoint_v2(self, path: str):
        """Load a v2 checkpoint as warm-start.

        PyG's HGTConv stores per-edge-type parameters as concatenated tensors
        (e.g., k_rel.weight [num_types*heads, head_dim, head_dim]). Since v3
        has 18 edge types (vs v2's 12), these tensors have different shapes.
        We handle this by:
        1. Copying the first 12*heads rows from v2 into v3's larger tensor
        2. Loading all other parameters with strict=False

        Returns:
            (epoch, loss) from the checkpoint.
        """
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt["model_state_dict"]

        # Register per-modality buffers if present in checkpoint
        for key in [
            "Z_h_BP", "Z_m_BP", "Z_h_MF", "Z_m_MF",
            "Z_h_Disease", "Z_m_Disease",
        ]:
            fk = f"hgt.{key}"
            if fk in state and not hasattr(self.model.hgt, key):
                self.model.hgt.register_buffer(key, state[fk])

        # Handle HGTConv size mismatches: v2 has 12 edge types, v3 has 18
        # Parameters like k_rel.weight are [num_types*heads, dim, dim]
        model_state = self.model.state_dict()
        size_mismatch_keys = []
        for k, v in list(state.items()):
            if k in model_state and v.shape != model_state[k].shape:
                # Copy v2 data into the first N rows of v3's larger tensor
                v3_param = model_state[k].clone()
                n_v2 = v.shape[0]
                v3_param[:n_v2] = v
                state[k] = v3_param
                size_mismatch_keys.append(
                    f"{k}: {v.shape} -> {model_state[k].shape} (copied first {n_v2} rows)"
                )

        if size_mismatch_keys:
            logger.info(
                f"Handled {len(size_mismatch_keys)} HGTConv size mismatches:"
            )
            for msg in size_mismatch_keys:
                logger.info(f"  {msg}")

        missing, unexpected = self.model.load_state_dict(state, strict=False)
        logger.info(
            f"Loaded v2 checkpoint: {len(missing)} missing keys, "
            f"{len(unexpected)} unexpected keys"
        )
        if missing:
            logger.info(f"  First 5 missing: {missing[:5]}")
        if unexpected:
            logger.info(f"  First 5 unexpected: {unexpected[:5]}")

        # Re-apply device split (v2 Trainer.__init__ already did this,
        # but load_state_dict may have reset devices)
        self.model.hgt = self.model.hgt.to("cpu")
        self.model.fusion_human = self.model.fusion_human.to(self.device)
        self.model.fusion_mouse = self.model.fusion_mouse.to(self.device)
        self.model.candidate_selector = self.model.candidate_selector.to(self.device)

        # Re-cache per-modality features on GPU
        self.mod_features = {}
        for key in [
            'Z_h_BP', 'Z_h_MF', 'Z_h_Disease',
            'Z_m_BP', 'Z_m_MF', 'Z_m_Disease',
        ]:
            buf = getattr(self.model.hgt, key, None)
            if buf is not None:
                self.mod_features[key] = buf.to(self.device)

        epoch = ckpt.get("epoch", 0)
        loss = ckpt.get("loss", float("inf"))
        logger.info(f"  Warm-start from epoch {epoch}, loss {loss:.4f}")
        return epoch, loss
