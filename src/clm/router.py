import dataclasses
import enum
import os
import typing

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
) -> typing.Callable[[S], str]:
    """Build a function usable with `StateGraph.add_conditional_edges`.

    The returned callable extracts the text from the graph state, asks CLM which route
    description fits best, and returns the name of the corresponding graph node.
    """
    if not routes:
        raise ValueError("routes must not be empty")
    if len({r.value for r in routes}) != len(routes):
        raise ValueError("route values must be unique")

    descriptions = [r.description for r in routes]
    nodes_by_value = {r.value: r.node for r in routes}
    values = [r.value for r in routes]

    # Fail fast on a missing checkpoint; the embedding server is only contacted on first use.
    clm = CLM(embedding_provider or LlamaEmbeddingProvider(), checkpoint_path)

    def route(state: S) -> str:
        selected = values[clm.select(state_selector(state), descriptions)]
        return nodes_by_value[selected]

    return route
