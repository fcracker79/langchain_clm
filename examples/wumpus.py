"""Wumpus World solved with a LangGraph graph whose decisions are taken by CLM.

Each loop iteration the `observe` node updates the agent's knowledge from the percepts and writes
a plain-English description of the situation. `create_router` hands that text to CLM, which picks
the intent that fits best (explore / grab the gold / leave); the router returns the node that
carries it out.

CLM is a semantic matcher: it tells "the gold is here" from "keep searching" reliably, but it does
not tell "turn left" from "turn right". So it chooses high-level intents, and the nodes do the
path-finding over the cells the agent has deduced to be safe.

Requires the llama embedding server (see README). Run with:
    uv run python examples/wumpus.py [--random [--seed N]] [--gui [--delay SECONDS]]
"""

import argparse
import dataclasses
import enum
import queue
import random
import threading
import time
import typing

from langgraph.graph import END, START, StateGraph

from clm import Route, create_router
from clm.embedding_provider import EmbeddingProvider, LlamaEmbeddingProvider

SIZE = 4
MAX_STEPS = 100
HEADINGS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # east, north, west, south (turn left = +1)

FACING = ["east", "north", "west", "south"]

Cell = tuple[int, int]
Knowledge = dict[Cell, frozenset[str]]  # visited cell -> percepts felt there


class Intent(enum.Enum):
    EXPLORE = "explore"
    GRAB = "grab"
    LEAVE = "leave"


@dataclasses.dataclass
class World:
    pits: set[Cell]
    wumpus: Cell
    gold: Cell
    pos: Cell = (0, 0)
    heading: int = 0
    has_gold: bool = False
    alive: bool = True
    escaped: bool = False
    steps: int = 0
    score: int = 0

    def percepts(self) -> frozenset[str]:
        found = set()
        for n in neighbors(self.pos):
            if n in self.pits:
                found.add("breeze")
            if n == self.wumpus:
                found.add("stench")
        if self.pos == self.gold and not self.has_gold:
            found.add("glitter")
        return frozenset(found)

    # Primitive actions; each one costs a point, as in the classic scoring.
    def turn_left(self) -> None:
        self._spend()
        self.heading = (self.heading + 1) % 4

    def forward(self) -> None:
        self._spend()
        dx, dy = HEADINGS[self.heading]
        self.pos = (self.pos[0] + dx, self.pos[1] + dy)
        if self.pos in self.pits or self.pos == self.wumpus:
            self.alive = False
            self.score -= 1000

    def grab(self) -> None:
        self._spend()
        self.has_gold = self.has_gold or self.pos == self.gold

    def climb(self) -> None:
        self._spend()
        if self.pos == (0, 0):
            self.escaped = True
            self.score += 1000 if self.has_gold else 0

    def _spend(self) -> None:
        self.steps += 1
        self.score -= 1

    def walk(self, path: list[Cell]) -> None:
        for cell in path:
            wanted = HEADINGS.index((cell[0] - self.pos[0], cell[1] - self.pos[1]))
            while self.heading != wanted:
                self.turn_left()
            self.forward()


def neighbors(c: Cell) -> list[Cell]:
    cells = [(c[0] + dx, c[1] + dy) for dx, dy in HEADINGS]
    return [n for n in cells if 0 <= n[0] < SIZE and 0 <= n[1] < SIZE]


# --- Knowledge: what the agent can deduce from the percepts seen so far -----------------------


def is_safe(cell: Cell, known: Knowledge) -> bool:
    if cell in known:
        return True
    seen = [known[n] for n in neighbors(cell) if n in known]
    # A visited neighbour without breeze/stench rules out a pit/wumpus in this cell.
    return any("breeze" not in p for p in seen) and any("stench" not in p for p in seen)


def find_path(start: Cell, goal: typing.Callable[[Cell], bool], known: Knowledge) -> list[Cell]:
    """Shortest path over safe cells to the nearest cell satisfying `goal` ([] if none)."""
    queue: list[list[Cell]] = [[]]
    seen = {start}
    while queue:
        path = queue.pop(0)
        here = path[-1] if path else start
        if path and goal(here):
            return path
        for n in neighbors(here):
            if n not in seen and is_safe(n, known):
                seen.add(n)
                queue.append([*path, n])
    return []


def unexplored_path(world: World, known: Knowledge) -> list[Cell]:
    return find_path(world.pos, lambda c: c not in known, known)


def describe(world: World, known: Knowledge) -> str:
    """The text CLM sees. Wording matters: these phrasings were tuned against the model."""
    percepts = ", ".join(sorted(world.percepts())) or "nothing"
    text = f"I am in a cave at {world.pos}. I perceive: {percepts}."
    if "glitter" in world.percepts():
        return text + " The gold is here, in front of me."
    if world.has_gold:
        return text + " I already have the gold in my bag."
    if not unexplored_path(world, known):
        return text + " I searched everywhere safe and found nothing else."
    return text + " I still have not found the gold, and there are cells I have not searched yet."


# --- World generation ---------------------------------------------------------------------------

PIT_PROBABILITY = 0.2


def classic_world() -> World:
    return World(pits={(2, 0), (2, 2), (3, 3)}, wumpus=(0, 2), gold=(1, 2))


def is_solvable(world: World) -> bool:
    """Whether exploring only cells proven safe is enough to reach the gold.

    The agent never takes risks, so on a world where the gold is behind an undecidable cell it
    would just leave empty-handed. Random worlds are filtered with this check.
    """
    probe = dataclasses.replace(world, pits=set(world.pits))
    known: Knowledge = {}
    while True:
        known[probe.pos] = probe.percepts()
        path = unexplored_path(probe, known)
        if not path:
            return world.gold in known
        probe.walk(path)


def random_world(rng: random.Random) -> World:
    """A random world where the start is hazard-free and the gold is safely reachable."""
    cells = [(x, y) for x in range(SIZE) for y in range(SIZE) if (x, y) != (0, 0)]
    while True:
        wumpus, gold = rng.choice(cells), rng.choice(cells)
        pits = {c for c in cells if c != wumpus and rng.random() < PIT_PROBABILITY}
        world = World(pits=pits - {gold}, wumpus=wumpus, gold=gold)
        if is_solvable(world):
            return world


def render(world: World) -> str:
    symbols = {c: "P" for c in world.pits} | {world.wumpus: "W", world.gold: "G", (0, 0): "S"}
    rows = [
        " ".join(symbols.get((x, y), ".") for x in range(SIZE)) for y in reversed(range(SIZE))
    ]
    return "\n".join(rows) + "\n(S start, P pit, W wumpus, G gold; y grows upwards)"


# --- Graph --------------------------------------------------------------------------------------


class State(typing.TypedDict):
    world: World
    known: Knowledge
    situation: str
    log: list[str]


def observe(state: State) -> dict[str, typing.Any]:
    world = state["world"]
    known = {**state["known"], world.pos: world.percepts()}
    situation = describe(world, known)
    return {"known": known, "situation": situation, "log": [*state["log"], situation]}


def explore(state: State) -> dict[str, typing.Any]:
    world = state["world"]
    path = unexplored_path(world, state["known"])
    world.walk(path)
    return {"log": [*state["log"], f"  -> explore: walked to {world.pos}"]}


def grab(state: State) -> dict[str, typing.Any]:
    state["world"].grab()
    return {"log": [*state["log"], "  -> grab the gold"]}


def leave(state: State) -> dict[str, typing.Any]:
    world = state["world"]
    world.walk(find_path(world.pos, lambda c: c == (0, 0), state["known"]))
    world.climb()
    return {"log": [*state["log"], "  -> walked back to the exit and climbed out"]}


NODES = {"explore": explore, "grab": grab, "leave": leave}

# Descriptions are what CLM matches against the situation text.
ROUTES = [
    Route(Intent.EXPLORE, "Keep searching: move to a new cell.", "explore"),
    Route(Intent.GRAB, "Pick up the gold.", "grab"),
    Route(Intent.LEAVE, "The job is done, so leave the cave.", "leave"),
]


def build_graph(
    on_decision: typing.Callable[[str, dict[Intent, float]], None] | None = None,
    embedding_provider: EmbeddingProvider | None = None,
) -> typing.Any:
    # Default embedding provider: llama server with Qwen3-8B on :8090.
    router = create_router(
        state_selector=lambda s: s["situation"],
        routes=ROUTES,
        embedding_provider=embedding_provider,
        on_decision=on_decision,
    )

    def route_or_stop(state: State) -> str:
        world = state["world"]
        if not world.alive or world.escaped or world.steps >= MAX_STEPS:
            return END
        return router(state)

    graph = StateGraph(State)
    graph.add_node("observe", observe)
    for name, fn in NODES.items():
        graph.add_node(name, fn)
        graph.add_edge(name, "observe")
    graph.add_edge(START, "observe")
    graph.add_conditional_edges("observe", route_or_stop)
    return graph.compile()


# --- GUI ----------------------------------------------------------------------------------------
#
# The graph runs in a worker thread and posts events to a queue; the Tk main loop drains the queue
# and redraws. Besides the board it shows what CLM answered at each step (probability of every
# intent) and how long each decision took, split between the embedding server and the CLM heads.

tk: typing.Any  # the tkinter module, imported by run_gui

CELL = 128
MARGIN = 34
PANEL_WIDTH = 520

BG = "#12151c"
PANEL_BG = "#1a1f2b"
CELL_BG = "#232a38"
VISITED_BG = "#2f3a50"
TEXT = "#e6e9ef"
MUTED = "#8a93a6"
ACCENT = "#4cc38a"
BAR = "#3b4a68"
GOLD = "#f5c542"
DANGER = "#e5484d"
AGENT = "#4c9aff"

FONT = ("DejaVu Sans", 11)
MONO = ("DejaVu Sans Mono", 10)


class TimedProvider:
    """Wraps an embedding provider to measure how much of a decision is spent in the server."""

    def __init__(self, inner: EmbeddingProvider) -> None:
        self._inner = inner
        self.reset()

    def reset(self) -> None:
        self.first_call_at: float | None = None
        self.embed_seconds = 0.0

    def embed(self, texts: list[str]) -> list[list[float]]:
        start = time.perf_counter()
        if self.first_call_at is None:
            self.first_call_at = start
        try:
            return self._inner.embed(texts)
        finally:
            self.embed_seconds += time.perf_counter() - start


class App:
    def __init__(self, world: World, delay: float) -> None:
        self.initial = world
        self.delay = delay
        self.graph: typing.Any = None
        self.events: queue.Queue[tuple[str, dict[str, typing.Any]]] = queue.Queue()
        self.timer = TimedProvider(LlamaEmbeddingProvider())
        self.running = False

        self.world = self._copy(world)
        self.known: dict[tuple[int, int], frozenset[str]] = {}
        self.latencies: list[float] = []

        self.root = tk.Tk()
        self.root.title("Wumpus World · CLM")
        self.root.configure(bg=BG)
        self._build_ui()
        self._reset_panel()
        self._draw_board()
        self.root.after(40, self._poll)

    # --- UI ---------------------------------------------------------------------------------

    def _build_ui(self) -> None:
        board_side = SIZE * CELL + 2 * MARGIN
        left = tk.Frame(self.root, bg=BG)
        left.pack(side=tk.LEFT, padx=18, pady=18)
        self.canvas = tk.Canvas(
            left, width=board_side, height=board_side, bg=BG, highlightthickness=0
        )
        self.canvas.pack()
        self.button = tk.Button(
            left,
            text="▶  Start",
            command=self.start,
            font=("DejaVu Sans", 15, "bold"),
            bg=ACCENT,
            fg="#0b1f15",
            activebackground="#6fd9a6",
            relief=tk.FLAT,
            padx=30,
            pady=8,
            cursor="hand2",
        )
        self.button.pack(pady=(14, 0))

        panel = tk.Frame(self.root, bg=PANEL_BG, width=PANEL_WIDTH)
        panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 18), pady=18)
        panel.pack_propagate(False)

        self._heading(panel, "CLM decision")
        self.bars = tk.Canvas(panel, width=PANEL_WIDTH - 40, height=108, bg=PANEL_BG, bd=0)
        self.bars.configure(highlightthickness=0)
        self.bars.pack(padx=20)

        self._heading(panel, "Timings")
        self.timing_var = tk.StringVar()
        tk.Label(
            panel, textvariable=self.timing_var, font=MONO, fg=TEXT, bg=PANEL_BG, justify=tk.LEFT
        ).pack(anchor=tk.W, padx=20)

        self._heading(panel, "Agent")
        self.agent_var = tk.StringVar()
        tk.Label(
            panel, textvariable=self.agent_var, font=MONO, fg=TEXT, bg=PANEL_BG, justify=tk.LEFT
        ).pack(anchor=tk.W, padx=20)

        self._heading(panel, "Log")
        log_frame = tk.Frame(panel, bg=PANEL_BG)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        self.log = tk.Text(
            log_frame,
            font=MONO,
            bg=CELL_BG,
            fg=TEXT,
            wrap=tk.WORD,
            relief=tk.FLAT,
            state=tk.DISABLED,
            padx=8,
            pady=6,
        )
        self.log.pack(fill=tk.BOTH, expand=True)
        self.log.tag_configure("situation", foreground=MUTED)
        self.log.tag_configure("decision", foreground=ACCENT)
        self.log.tag_configure("action", foreground=TEXT)
        self.log.tag_configure("error", foreground=DANGER)

    def _heading(self, parent: typing.Any, text: str) -> None:
        tk.Label(
            parent, text=text.upper(), font=("DejaVu Sans", 10, "bold"), fg=MUTED, bg=PANEL_BG
        ).pack(anchor=tk.W, padx=20, pady=(16, 6))

    def _reset_panel(self) -> None:
        self.latencies = []
        self._draw_bars({}, None)
        self.timing_var.set("last decision   –\nembedding      –\nCLM heads      –\naverage        –")
        self._update_agent("ready")
        self.log.configure(state=tk.NORMAL)
        self.log.delete("1.0", tk.END)
        self.log.configure(state=tk.DISABLED)

    def _write_log(self, text: str, tag: str) -> None:
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, text + "\n", tag)
        self.log.configure(state=tk.DISABLED)
        self.log.see(tk.END)

    def _update_agent(self, status: str) -> None:
        w = self.world
        self.agent_var.set(
            f"status    {status}\n"
            f"position  {w.pos}, facing {FACING[w.heading]}\n"
            f"gold      {'in the bag' if w.has_gold else 'not yet'}\n"
            f"steps     {w.steps}\n"
            f"score     {w.score}"
        )

    # --- Drawing ----------------------------------------------------------------------------

    def _cell_box(self, cell: tuple[int, int]) -> tuple[int, int, int, int]:
        x0 = MARGIN + cell[0] * CELL
        y0 = MARGIN + (SIZE - 1 - cell[1]) * CELL  # y grows upwards
        return x0, y0, x0 + CELL, y0 + CELL

    def _draw_board(self) -> None:
        c, w = self.canvas, self.world
        c.delete("all")
        for i in range(SIZE):
            mid = MARGIN + i * CELL + CELL // 2
            c.create_text(mid, MARGIN // 2, text=str(i), fill=MUTED, font=FONT)
            c.create_text(MARGIN // 2, mid, text=str(SIZE - 1 - i), fill=MUTED, font=FONT)

        for x in range(SIZE):
            for y in range(SIZE):
                x0, y0, x1, y1 = self._cell_box((x, y))
                fill = VISITED_BG if (x, y) in self.known else CELL_BG
                c.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2, fill=fill, outline="")
                if (x, y) == (0, 0):
                    c.create_text(x0 + 14, y0 + 14, text="START", fill=MUTED, font=("", 8, "bold"))
                hints = sorted(self.known.get((x, y), frozenset()) - {"glitter"})
                if hints:
                    c.create_text(
                        (x0 + x1) // 2, y1 - 12, text=" · ".join(hints), fill=MUTED, font=("", 9)
                    )

        for pit in w.pits:
            self._draw_pit(*self._center(pit))
        self._draw_wumpus(*self._center(w.wumpus))
        if not w.has_gold:
            self._draw_gold(*self._center(w.gold))
        self._draw_agent(*self._center(w.pos))

    def _center(self, cell: tuple[int, int]) -> tuple[int, int]:
        x0, y0, x1, y1 = self._cell_box(cell)
        return (x0 + x1) // 2, (y0 + y1) // 2

    def _draw_pit(self, cx: int, cy: int) -> None:
        c = self.canvas
        c.create_oval(cx - 50, cy - 12, cx + 50, cy + 38, fill="#4a4036", outline="#2b251f", width=2)
        c.create_oval(cx - 42, cy - 6, cx + 42, cy + 32, fill="#0a0806", outline="")
        base = cy + 22
        for dx, height in ((-30, 24), (30, 26), (-15, 36), (15, 38), (0, 30)):
            x = cx + dx
            c.create_polygon(x - 7, base, x, base - height, x, base, fill="#e8ecf3", outline="")
            c.create_polygon(x, base, x, base - height, x + 7, base, fill="#8b94a8", outline="")

    def _draw_wumpus(self, cx: int, cy: int) -> None:
        c = self.canvas
        for sign in (-1, 1):
            c.create_polygon(
                cx + sign * 26, cy - 18, cx + sign * 34, cy - 44, cx + sign * 12, cy - 28,
                fill="#f2e6c9", outline="#a89870", width=2,
            )  # fmt: skip
            c.create_oval(
                cx + sign * 28 - 10, cy + 30, cx + sign * 28 + 10, cy + 44,
                fill="#6a38b5", outline="#4b2a85", width=2,
            )  # fmt: skip
        c.create_oval(cx - 38, cy - 32, cx + 38, cy + 36, fill="#7a46c9", outline="#4b2a85", width=3)
        for sign in (-1, 1):
            c.create_oval(cx + sign * 14 - 9, cy - 19, cx + sign * 14 + 9, cy - 1, fill="white")
            c.create_oval(
                cx + sign * 14 - sign * 2 - 4, cy - 14, cx + sign * 14 - sign * 2 + 4, cy - 6,
                fill=DANGER, outline="",
            )  # fmt: skip
            c.create_line(
                cx + sign * 24, cy - 24, cx + sign * 6, cy - 16, fill="#1c0f33", width=3
            )  # angry brow
        c.create_oval(cx - 24, cy + 6, cx + 24, cy + 28, fill="#2a0f1f", outline="#1c0f33", width=2)
        for x in (-12, -4, 4):
            c.create_polygon(cx + x, cy + 8, cx + x + 8, cy + 8, cx + x + 4, cy + 17, fill="white")
        for x in (-9, 1):
            c.create_polygon(cx + x, cy + 27, cx + x + 8, cy + 27, cx + x + 4, cy + 20, fill="white")

    def _draw_gold(self, cx: int, cy: int) -> None:
        c = self.canvas
        rows = ((30, (-30, -10, 10, 30)), (21, (-20, 0, 20)), (12, (-10, 10)), (3, (0,)))
        for y, xs in rows:
            for x in xs:
                c.create_oval(
                    cx + x - 15, cy + y - 8, cx + x + 15, cy + y + 8,
                    fill=GOLD, outline="#a87a00", width=2,
                )  # fmt: skip
                c.create_oval(cx + x - 9, cy + y - 4, cx + x + 9, cy + y + 4, outline="#d9a400")
        c.create_text(cx, cy + 3, text="$", fill="#a87a00", font=("DejaVu Sans", 8, "bold"))
        for sx, sy in ((-34, 0), (34, 8), (14, -14)):  # sparkles
            c.create_polygon(
                cx + sx, cy + sy - 8, cx + sx + 2, cy + sy - 2, cx + sx + 8, cy + sy,
                cx + sx + 2, cy + sy + 2, cx + sx, cy + sy + 8, cx + sx - 2, cy + sy + 2,
                cx + sx - 8, cy + sy, cx + sx - 2, cy + sy - 2, fill="#fff6c9", outline="",
            )  # fmt: skip

    def _draw_agent(self, cx: int, cy: int) -> None:
        c, w = self.canvas, self.world
        dead = not w.alive
        jacket = DANGER if dead else "#b5713a"
        for sign in (-1, 1):
            c.create_line(
                cx + sign * 13, cy + 4, cx + sign * 22, cy + 18,
                fill=jacket, width=5, capstyle=tk.ROUND,
            )  # fmt: skip
            c.create_rectangle(cx + sign * 6 - 3, cy + 18, cx + sign * 6 + 3, cy + 36, fill="#3b4a68")
            c.create_oval(
                cx + sign * 7 - 7, cy + 34, cx + sign * 7 + 7, cy + 41, fill="#2f1d0e", outline=""
            )  # fmt: skip
        c.create_oval(cx - 14, cy - 2, cx + 14, cy + 24, fill=jacket, outline="#6b4423", width=2)
        c.create_oval(cx - 14, cy - 26, cx + 14, cy + 2, fill="#f1c27d", outline="#a9784a", width=2)
        if dead:
            for sign in (-1, 1):
                x = cx + sign * 5
                c.create_line(x - 3, cy - 15, x + 3, cy - 9, fill="black", width=2)
                c.create_line(x - 3, cy - 9, x + 3, cy - 15, fill="black", width=2)
        else:
            for sign in (-1, 1):
                c.create_oval(cx + sign * 5 - 2, cy - 14, cx + sign * 5 + 2, cy - 10, fill="black")
            c.create_arc(
                cx - 7, cy - 14, cx + 7, cy - 2, start=210, extent=120, style=tk.ARC, width=2
            )
        c.create_oval(cx - 15, cy - 46, cx + 15, cy - 26, fill="#7a4f2a", outline="#2f1d0e", width=2)
        c.create_line(cx - 6, cy - 45, cx, cy - 41, cx + 6, cy - 45, fill="#2f1d0e", width=2)
        c.create_rectangle(cx - 14, cy - 34, cx + 14, cy - 28, fill="#2f1d0e", outline="")
        c.create_oval(cx - 28, cy - 33, cx + 28, cy - 21, fill="#5c3a1e", outline="#2f1d0e", width=2)
        if w.has_gold:
            c.create_oval(cx + 17, cy + 13, cx + 31, cy + 25, fill=GOLD, outline="#a87a00", width=2)

    def _draw_bars(self, probs: dict[str, float], chosen: str | None) -> None:
        b = self.bars
        b.delete("all")
        width = PANEL_WIDTH - 40
        names = list(probs) or ["explore", "grab", "leave"]
        for i, name in enumerate(names):
            y = 6 + i * 34
            p = probs.get(name, 0.0)
            color = ACCENT if name == chosen else BAR
            b.create_text(0, y + 12, text=name, anchor=tk.W, fill=TEXT, font=FONT)
            x0, x1 = 80, width - 56
            b.create_rectangle(x0, y + 2, x1, y + 22, fill=CELL_BG, outline="")
            b.create_rectangle(x0, y + 2, x0 + (x1 - x0) * p, y + 22, fill=color, outline="")
            b.create_text(width, y + 12, text=f"{p:.1%}" if probs else "–", anchor=tk.E,
                          fill=TEXT, font=FONT)  # fmt: skip

    # --- Run control ------------------------------------------------------------------------

    @staticmethod
    def _copy(world: World) -> World:
        return dataclasses.replace(world, pits=set(world.pits))

    def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.button.configure(state=tk.DISABLED, text="Running…")
        self.world = self._copy(self.initial)
        self.known = {}
        self.timer.reset()
        self._reset_panel()
        self._draw_board()
        threading.Thread(target=self._run, daemon=True).start()

    def _post(self, kind: str, **payload: typing.Any) -> None:
        self.events.put((kind, payload))

    def _run(self) -> None:
        try:
            if self.graph is None:
                self._post("status", text="loading CLM…")
                self.graph = build_graph(self._decide, self.timer)
            world = self._copy(self.initial)
            state = {"world": world, "known": {}, "situation": "", "log": []}
            for update in self.graph.stream(state, {"recursion_limit": 200}, stream_mode="updates"):
                for node, delta in update.items():
                    self._post("node", node=node, delta=delta, world=self._copy(world))
                    if node != "observe":  # the decision that follows already pauses
                        time.sleep(self.delay)
            self._post("done", world=self._copy(world))
        except Exception as e:  # shown in the log; the window stays usable
            self._post("error", message=f"{type(e).__name__}: {e}")

    def _decide(self, text: str, probs: dict[Intent, float]) -> None:
        """Runs in the worker thread, inside the router."""
        now = time.perf_counter()
        started = self.timer.first_call_at or now
        total, embed = now - started, self.timer.embed_seconds
        self.timer.reset()
        self._post(
            "decision",
            probs={intent.value: p for intent, p in probs.items()},
            total=total,
            embed=embed,
        )
        time.sleep(self.delay)

    # --- Event handling (Tk thread) ---------------------------------------------------------

    def _poll(self) -> None:
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                break
            getattr(self, f"_on_{kind}")(**payload)
        self.root.after(40, self._poll)

    def _on_status(self, text: str) -> None:
        self._update_agent(text)

    def _on_node(self, node: str, delta: dict[str, typing.Any], world: World) -> None:
        self.world = world
        if node == "observe":
            self.known = delta["known"]
            self._write_log(f"step {world.steps}: {delta['situation']}", "situation")
            self._update_agent("observing")
        else:
            self._write_log(delta["log"][-1].strip(), "action")
            self._update_agent(f"running {node}")
        self._draw_board()

    def _on_decision(self, probs: dict[str, float], total: float, embed: float) -> None:
        chosen = max(probs, key=probs.__getitem__)
        self._draw_bars(probs, chosen)
        self.latencies.append(total)
        average = sum(self.latencies) / len(self.latencies)
        self.timing_var.set(
            f"last decision  {total * 1000:7.0f} ms\n"
            f"embedding      {embed * 1000:7.0f} ms\n"
            f"CLM heads      {(total - embed) * 1000:7.0f} ms\n"
            f"average        {average * 1000:7.0f} ms  ({len(self.latencies)} calls)"
        )
        self._write_log(f"CLM → {chosen} ({probs[chosen]:.1%}) in {total * 1000:.0f} ms", "decision")

    def _on_done(self, world: World) -> None:
        self.world = world
        outcome = "died" if not world.alive else "escaped" if world.escaped else "out of steps"
        self._update_agent(outcome)
        self._write_log(
            f"\n{outcome.upper()} · gold={world.has_gold} · score={world.score}", "decision"
        )
        self._draw_board()
        self.running = False
        self.button.configure(state=tk.NORMAL, text="↻  Restart")

    def _on_error(self, message: str) -> None:
        self._write_log(message, "error")
        self._update_agent("error")
        self.running = False
        self.button.configure(state=tk.NORMAL, text="↻  Retry")


def run_gui(world: World, delay: float) -> None:
    global tk
    import tkinter as tk  # only needed (and possibly missing) when the GUI is requested

    App(world, delay).root.mainloop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--random", action="store_true", help="use a random world")
    parser.add_argument("--seed", type=int, help="seed for the random world (implies --random)")
    parser.add_argument("--gui", action="store_true", help="show a window instead of the terminal")
    parser.add_argument(
        "--delay", type=float, default=1.0, help="seconds between steps in the GUI (default 1)"
    )
    args = parser.parse_args()

    if args.random or args.seed is not None:
        world = random_world(random.Random(args.seed))
    else:
        world = classic_world()
    if args.gui:
        run_gui(world, args.delay)
        return
    print(render(world), end="\n\n")
    result = build_graph().invoke(
        {"world": world, "known": {}, "situation": "", "log": []},
        {"recursion_limit": 200},
    )
    print("\n".join(result["log"]))
    outcome = "died" if not world.alive else "escaped" if world.escaped else "ran out of steps"
    print(f"\nAgent {outcome} after {world.steps} steps; gold={world.has_gold}; score={world.score}")


if __name__ == "__main__":
    main()
