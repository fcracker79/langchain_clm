import enum
import pathlib
import typing

import pytest
import torch
from langgraph.graph import END, START, StateGraph

import clm.model
from clm import Route, create_router
from clm.model import _Head

DIM = 8


class Kind(enum.Enum):
    REFUND = "refund"
    EMAIL = "email"
    DELETE = "delete"


class OneHotProvider:
    """Embeds each known text as its own axis, so similarity is exact-match."""

    def __init__(self, dim: int = DIM) -> None:
        self.dim = dim
        self.calls: list[list[str]] = []
        self._axes: dict[str, int] = {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(list(texts))
        vectors = []
        for text in texts:
            axis = self._axes.setdefault(text, len(self._axes))
            vectors.append([1.0 if i == axis else 0.0 for i in range(self.dim)])
        return vectors


@pytest.fixture
def checkpoint(tmp_path: pathlib.Path) -> pathlib.Path:
    """A tiny CLM checkpoint whose heads are the identity, so cosine == embedding similarity."""
    cfg = {
        "width": DIM,
        "projection_dim": DIM,
        "depth": 2,
        "activation": "relu",
        "layernorm": False,
        "hidden_size": DIM,
    }
    head = _Head(cfg)
    with torch.no_grad():
        head.inp.weight.copy_(torch.eye(DIM))
        head.inp.bias.zero_()
        head.out.weight.copy_(torch.eye(DIM))
        head.out.bias.zero_()
    path = tmp_path / "clm.pt"
    torch.save(
        {
            "cfg": cfg,
            "state_head": head.state_dict(),
            "action_head": head.state_dict(),
            "logit_scale": torch.tensor(3.0),
        },
        path,
    )
    return path


@pytest.fixture
def routes() -> list[Route[Kind]]:
    return [
        Route(Kind.REFUND, "refund the charge", "refund_node"),
        Route(Kind.EMAIL, "send an email", "email_node"),
        Route(Kind.DELETE, "delete the account", "delete_node"),
    ]


def make_router(
    routes: list[Route[Kind]], checkpoint: pathlib.Path, provider: OneHotProvider | None = None
) -> typing.Callable[[dict[str, str]], str]:
    return create_router(
        state_selector=lambda s: s["text"],
        routes=routes,
        embedding_provider=provider or OneHotProvider(),
        checkpoint_path=checkpoint,
    )


@pytest.mark.parametrize("route_index", [0, 1, 2])
def test_returns_node_of_best_matching_route(
    routes: list[Route[Kind]], checkpoint: pathlib.Path, route_index: int
) -> None:
    router = make_router(routes, checkpoint)
    assert router({"text": routes[route_index].description}) == routes[route_index].node


def test_state_selector_picks_the_text(routes: list[Route[Kind]], checkpoint: pathlib.Path) -> None:
    router = make_router(routes, checkpoint)
    state = {"text": "send an email", "other": "delete the account"}
    assert router(state) == "email_node"


def test_node_name_is_independent_from_enum_value(checkpoint: pathlib.Path) -> None:
    routes = [Route(Kind.REFUND, "a", "x"), Route(Kind.EMAIL, "b", "x")]
    assert make_router(routes, checkpoint)({"text": "b"}) == "x"


def test_action_embeddings_are_computed_once(
    routes: list[Route[Kind]], checkpoint: pathlib.Path
) -> None:
    provider = OneHotProvider()
    router = make_router(routes, checkpoint, provider)

    router({"text": "refund the charge"})
    router({"text": "send an email"})

    descriptions = [r.description for r in routes]
    assert provider.calls == [descriptions, ["refund the charge"], ["send an email"]]


def test_on_decision_receives_text_and_probabilities(
    routes: list[Route[Kind]], checkpoint: pathlib.Path
) -> None:
    seen: list[tuple[str, dict[Kind, float]]] = []
    router = create_router(
        state_selector=lambda s: s["text"],
        routes=routes,
        embedding_provider=OneHotProvider(),
        checkpoint_path=checkpoint,
        on_decision=lambda text, probs: seen.append((text, probs)),
    )

    assert router({"text": "send an email"}) == "email_node"

    [(text, probs)] = seen
    assert text == "send an email"
    assert set(probs) == set(Kind)
    assert max(probs, key=probs.__getitem__) == Kind.EMAIL
    assert sum(probs.values()) == pytest.approx(1.0)


def test_works_as_conditional_edge(routes: list[Route[Kind]], checkpoint: pathlib.Path) -> None:
    class State(typing.TypedDict):
        text: str
        visited: str

    def make_node(name: str) -> typing.Callable[[State], dict[str, str]]:
        return lambda state: {"visited": name}

    graph = StateGraph(State)
    for r in routes:
        graph.add_node(r.node, make_node(r.node))
        graph.add_edge(r.node, END)
    graph.add_node("start", lambda state: {})
    graph.add_edge(START, "start")
    graph.add_conditional_edges("start", make_router(routes, checkpoint))  # type: ignore[arg-type]

    result = graph.compile().invoke({"text": "delete the account", "visited": ""})
    assert result["visited"] == "delete_node"


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        torch.device("cpu"),
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device"),
        ),
    ],
)
def test_device_is_configurable(
    routes: list[Route[Kind]], checkpoint: pathlib.Path, device: str | torch.device
) -> None:
    router = create_router(
        state_selector=lambda s: s["text"],
        routes=routes,
        embedding_provider=OneHotProvider(),
        checkpoint_path=checkpoint,
        device=device,
    )
    assert router({"text": "send an email"}) == "email_node"


def test_invalid_device_rejected(routes: list[Route[Kind]], checkpoint: pathlib.Path) -> None:
    with pytest.raises(RuntimeError):
        create_router(
            state_selector=lambda s: s["text"],
            routes=routes,
            embedding_provider=OneHotProvider(),
            checkpoint_path=checkpoint,
            device="not-a-device",
        )


def test_empty_routes_rejected(checkpoint: pathlib.Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        make_router([], checkpoint)


def test_duplicate_enum_values_rejected(checkpoint: pathlib.Path) -> None:
    routes = [Route(Kind.REFUND, "a", "n1"), Route(Kind.REFUND, "b", "n2")]
    with pytest.raises(ValueError, match="unique"):
        make_router(routes, checkpoint)


def test_missing_explicit_checkpoint_fails_at_creation(
    routes: list[Route[Kind]], tmp_path: pathlib.Path
) -> None:
    with pytest.raises(FileNotFoundError):
        make_router(routes, tmp_path / "missing.pt")


def test_embedding_dimension_mismatch_is_reported(
    routes: list[Route[Kind]], checkpoint: pathlib.Path
) -> None:
    router = make_router(routes, checkpoint, OneHotProvider(dim=DIM + 1))
    with pytest.raises(ValueError, match="expects 8"):
        router({"text": "refund the charge"})


def test_checkpoint_taken_from_env_var(
    routes: list[Route[Kind]], checkpoint: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(clm.model.CHECKPOINT_ENV_VAR, str(checkpoint))
    router = create_router(
        state_selector=lambda s: s["text"], routes=routes, embedding_provider=OneHotProvider()
    )
    assert router({"text": "send an email"}) == "email_node"


def test_checkpoint_downloaded_once_when_not_provided(
    routes: list[Route[Kind]],
    checkpoint: pathlib.Path,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = tmp_path / "cache"
    downloads: list[pathlib.Path] = []

    def fake_download(destination: pathlib.Path) -> None:
        downloads.append(destination)
        destination.parent.mkdir(parents=True)
        destination.write_bytes(checkpoint.read_bytes())

    monkeypatch.delenv(clm.model.CHECKPOINT_ENV_VAR, raising=False)
    monkeypatch.setattr(clm.model, "CACHE_DIR", cache)
    monkeypatch.setattr(clm.model, "_download_checkpoint", fake_download)

    for _ in range(2):
        router = create_router(
            state_selector=lambda s: s["text"], routes=routes, embedding_provider=OneHotProvider()
        )
        assert router({"text": "refund the charge"}) == "refund_node"

    assert downloads == [cache / clm.model.CHECKPOINT_NAME]
