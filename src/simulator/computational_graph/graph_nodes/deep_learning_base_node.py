"""
Base class for all torch-backed deep-learning graph nodes.

Replaces the qxcl keras version. Loads either a TorchScript module or a
plain `torch.save`-d module from `{model_base_dir}/final_model.pt`.
"""

from __future__ import annotations

from os.path import join
from typing import List, Optional

import numpy as np
import torch

from simulator.computational_graph.graph_nodes.abstract_graph_node import AbstractGraphNode
from simulator.computational_graph.graph_nodes.node_configs import DeepLearningBaseNodeConfig


class DeepLearningBaseNode(AbstractGraphNode):
    """
    Base class for graph nodes that wrap a trained torch model.

    Subclasses implement `_preprocess(np.ndarray) -> np.ndarray` and
    `_postprocess(np.ndarray) -> np.ndarray`; `compute()` does
    `model(preprocess(x))` and unwraps the result.
    """

    def __init__(self, deeplearning_config: DeepLearningBaseNodeConfig) -> None:
        super().__init__(deeplearning_config)

        if deeplearning_config.model is not None:
            self.model = deeplearning_config.model
        else:
            self.model = self._load_model(deeplearning_config.model_base_dir)

        self.device = torch.device(deeplearning_config.device)
        self.model = self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def _load_model(model_base_dir: str) -> torch.nn.Module:
        path = join(model_base_dir, "final_model.pt")
        obj = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(obj, torch.nn.Module):
            return obj
        raise TypeError(
            f"Loaded object from {path} is not a torch.nn.Module "
            f"(got {type(obj).__name__}). Save your trained model with "
            "`torch.save(model, path)` (the whole module, not just state_dict)."
        )

    def compute(self, x: Optional[np.ndarray] = None) -> List[np.ndarray]:
        if x is None:
            raise ValueError(f"Input must be provided for node {self.name}")

        x_pre = self._preprocess(x)
        with torch.no_grad():
            tensor = torch.as_tensor(x_pre, dtype=torch.float32, device=self.device)
            y = self.model(tensor)
        if isinstance(y, torch.Tensor):
            y_np = y.detach().cpu().numpy()
        else:
            y_np = np.asarray(y)
        return [self._postprocess(y_np)]

    def _preprocess(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Deep learning node preprocessing function not implemented")

    def _postprocess(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Deep learning node postprocessing function not implemented")
