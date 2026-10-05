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
