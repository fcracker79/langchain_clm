import dataclasses
import enum
import os
import typing

import torch

from clm.embedding_provider import EmbeddingProvider, LlamaEmbeddingProvider
from clm.model import CLM

S = typing.TypeVar("S")
E = typing.TypeVar("E", bound=enum.Enum)


@dataclasses.dataclass(frozen=True)
class Route(typing.Generic[E]):
    """A possible outcome: enum value, how CLM should read it, and where it leads."""

    value: E
    description: str
    node: str


def create_router(
    *,
    state_selector: typing.Callable[[S], str],
    routes: typing.Sequence[Route[E]],
    embedding_provider: EmbeddingProvider | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    device: str | torch.device = "cpu",
    on_decision: typing.Callable[[str, dict[E, float]], None] | None = None,
) -> typing.Callable[[S], str]:
    """Build a function usable with `StateGraph.add_conditional_edges`.

    The returned callable extracts the text from the graph state, asks CLM which route
    description fits best, and returns the name of the corresponding graph node.
    `device` is where the CLM projection heads run ("cpu", "cuda", "cuda:1", "mps", ...).
    `on_decision`, if given, is called with the state text and the probability CLM assigned to each
    route value, right before the router returns (handy for logging or visualising decisions).
    """
    if not routes:
        raise ValueError("routes must not be empty")
    if len({r.value for r in routes}) != len(routes):
        raise ValueError("route values must be unique")

    descriptions = [r.description for r in routes]
    nodes_by_value = {r.value: r.node for r in routes}
    values = [r.value for r in routes]

    # Fail fast on a missing checkpoint; the embedding server is only contacted on first use.
    clm = CLM(embedding_provider or LlamaEmbeddingProvider(), checkpoint_path, device)

    def route(state: S) -> str:
        text = state_selector(state)
        probabilities = clm.probabilities(text, descriptions)
        selected = values[max(range(len(values)), key=probabilities.__getitem__)]
        if on_decision is not None:
            on_decision(text, dict(zip(values, probabilities)))
        return nodes_by_value[selected]

    return route
