import logging
import os
import pathlib
import typing

import requests
import torch
import torch.nn as nn
import torch.nn.functional as F

from clm.embedding_provider import EmbeddingProvider

logger = logging.getLogger(__name__)

CHECKPOINT_ENV_VAR = "CLM_CHECKPOINT"
CHECKPOINT_NAME = "CLM_v0.1-8B.pt"
CHECKPOINT_URL = f"https://huggingface.co/sekkit/CLM-v0.1-8B/resolve/main/{CHECKPOINT_NAME}"
# Same location used by the official `clm-serve`, so the file is shared.
CACHE_DIR = pathlib.Path.home() / ".cache" / "clm"

_ACTIVATIONS: dict[str, typing.Callable[[], nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class _Head(nn.Module):
    def __init__(self, cfg: dict[str, typing.Any]) -> None:
        super().__init__()
        width = cfg["width"]
        depth = cfg["depth"]

        if cfg["activation"] not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation: {cfg['activation']}")

        self.inp = nn.Linear(cfg["hidden_size"], width)
        self.hidden = nn.ModuleList(nn.Linear(width, width) for _ in range(depth - 2))
        self.norms = nn.ModuleList(
            nn.LayerNorm(width) if cfg["layernorm"] else nn.Identity() for _ in range(depth - 2)
        )
        self.out = nn.Linear(width, cfg["projection_dim"])
        self.act = _ACTIVATIONS[cfg["activation"]]()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.inp(x))
        for linear, norm in zip(self.hidden, self.norms):
            x = self.act(norm(linear(x)))
        return typing.cast(torch.Tensor, self.out(x))


def _download_checkpoint(destination: pathlib.Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    try:
        with requests.get(CHECKPOINT_URL, stream=True, timeout=60) as response:
            response.raise_for_status()
            with partial.open("wb") as f:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
        partial.replace(destination)  # atomic: never leaves a truncated checkpoint behind
    finally:
        partial.unlink(missing_ok=True)


def resolve_checkpoint(path: str | os.PathLike[str] | None = None) -> pathlib.Path:
    """Locate the CLM checkpoint.

    An explicit path (argument or $CLM_CHECKPOINT) must exist. Otherwise the checkpoint is
    taken from the cache directory, downloading it from Hugging Face on first use.
    """
    explicit = path or os.environ.get(CHECKPOINT_ENV_VAR)
    if explicit:
        candidate = pathlib.Path(explicit)
        if not candidate.is_file():
            raise FileNotFoundError(f"CLM checkpoint not found at '{candidate}'.")
        return candidate

    cached = CACHE_DIR / CHECKPOINT_NAME
    if not cached.is_file():
        logger.info("Downloading CLM checkpoint to %s", cached)
        _download_checkpoint(cached)
    return cached


class CLM:
    """Scores candidate actions against a state using the CLM projection heads."""

    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        checkpoint_path: str | os.PathLike[str] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        """`device` is where the projection heads run (e.g. "cpu", "cuda", "cuda:1", "mps")."""
        self._device = torch.device(device)
        checkpoint = torch.load(
            resolve_checkpoint(checkpoint_path), map_location=self._device, weights_only=True
        )
        cfg = checkpoint["cfg"]

        self._state_head = self._load_head(cfg, checkpoint["state_head"], self._device)
        self._action_head = self._load_head(cfg, checkpoint["action_head"], self._device)
        self._logit_scale = float(
            torch.as_tensor(checkpoint["logit_scale"]).float().exp().clamp(max=100.0).cpu()
        )
        self._embeddings = embedding_provider
        self._hidden_size = int(cfg["hidden_size"])
        self._action_cache: dict[tuple[str, ...], torch.Tensor] = {}

    @staticmethod
    def _load_head(
        cfg: dict[str, typing.Any], state: dict[str, torch.Tensor], device: torch.device
    ) -> _Head:
        head = _Head(cfg)
        head.load_state_dict(state)
        return head.to(device).eval()

    def _embed(self, texts: list[str]) -> torch.Tensor:
        x = torch.tensor(self._embeddings.embed(texts), dtype=torch.float32, device=self._device)
        if x.shape[-1] != self._hidden_size:
            raise ValueError(
                f"Embedding provider returned {x.shape[-1]}-dim vectors, but the CLM "
                f"checkpoint expects {self._hidden_size} (Qwen3-8B embeddings)."
            )
        # Same L2 normalization used by the official CLM Embedder.
        return x / (x.norm(dim=-1, keepdim=True) + 1e-12)

    def _project_actions(self, actions: typing.Sequence[str]) -> torch.Tensor:
        key = tuple(actions)
        if key not in self._action_cache:
            with torch.no_grad():
                self._action_cache[key] = F.normalize(
                    self._action_head(self._embed(list(actions))), dim=-1
                )
        return self._action_cache[key]

    def probabilities(self, state: str, actions: typing.Sequence[str]) -> list[float]:
        action_proj = self._project_actions(actions)
        with torch.no_grad():
            state_proj = F.normalize(self._state_head(self._embed([state])), dim=-1)
            scores = self._logit_scale * (action_proj @ state_proj.T).squeeze(1)
            return torch.softmax(scores, dim=0).tolist()

    def select(self, state: str, actions: typing.Sequence[str]) -> int:
        """Index of the most likely action for the given state."""
        probs = self.probabilities(state, actions)
        return max(range(len(probs)), key=probs.__getitem__)
