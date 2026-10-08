# langchain-clm

Route a LangGraph graph with a CLM (Contrastive Language Model).

```python
router = create_router(
    state_selector=lambda s: s["last_message"],
    routes=[Route(Intent.REFUND, "Refund the duplicate charge", "refund_node")],
)
graph.add_conditional_edges("classify", router)
```

## Checkpoint

`checkpoint_path` (or `$CLM_CHECKPOINT`) if given; otherwise `~/.cache/clm/CLM_v0.1-8B.pt`,
downloaded from Hugging Face on first use.

## Device

The CLM projection heads (the small networks on top of the encoder) run on the CPU by default.
Pass `device="cuda"` (or `"cuda:1"`, `"mps"`, a `torch.device`, ...) to `create_router` to use a
GPU:

```python
create_router(..., device="cuda")
```

This only affects the projection heads. The heavy part, the Qwen3-8B encoder that produces the
embeddings, runs in the embedding server, so GPU use there is configured on the server itself
(e.g. `llama serve ... -ngl 99` to offload layers to the GPU), not through this library.

## Embeddings

By default embeddings come from a llama server running Qwen3-8B
(`llama serve -m Qwen3-8B-Q4_K_M.gguf --embedding --pooling last --port 8090`).
Configure it, or replace it, with `embedding_provider`:

```python
# Different server / address
create_router(..., embedding_provider=LlamaEmbeddingProvider(base_url="http://host:8090"))

# Any LangChain Embeddings object
create_router(..., embedding_provider=LangChainEmbeddingProvider(my_embeddings))

# Fully custom: any object with embed(list[str]) -> list[list[float]]
class MyProvider:
    def embed(self, texts: list[str]) -> list[list[float]]: ...
```

The CLM heads were trained on Qwen3-8B embeddings (4096 dims, last-token pooling), so a custom
provider must serve the same encoder; a dimension mismatch raises a `ValueError`.

## Example: Wumpus World

`examples/wumpus.py` solves the Wumpus World with a LangGraph graph in which CLM takes the
decisions. It needs the llama embedding server running (see above).

```bash
uv run python examples/wumpus.py              # the classic 4x4 world
uv run python examples/wumpus.py --random     # a random world
uv run python examples/wumpus.py --seed 7     # a reproducible random world (implies --random)
uv run python examples/wumpus.py --gui        # a window instead of the terminal (needs tkinter)
```

`--gui` shows the board (pits, wumpus, gold, agent), a Start button, and a side panel with what CLM
answered at each step (the probability of every intent), how long each decision took (embedding
server vs. CLM heads) and a log. `--delay SECONDS` sets the pause between steps (default 1).

The map is printed first (`S` start, `P` pit, `W` wumpus, `G` gold), then the agent's
observations and actions, and finally the outcome and score.

How it works:

- An `observe` node updates what the agent knows from the percepts (breeze, stench, glitter) and
  describes the situation in plain English.
- `create_router` gives that text to CLM, which chooses between three intents: *explore*, *grab
  the gold* or *leave*. The router returns the node that carries the intent out.
- Those nodes do the path-finding over the cells the agent has proven safe, so it never steps on
  a pit or on the wumpus.

Two things to know:

- CLM is a semantic matcher and does not tell "turn left" from "turn right" (nor north from
  east). That is why it picks high-level intents and the nodes handle navigation.
- The agent never takes risks and never shoots. Random worlds are filtered so that the gold can
  be reached through cells proven safe.

The wording of the situation text and of the route descriptions affects which route CLM picks;
the ones in the example were tuned against the model, so re-check them if you change them.
