"""Wumpus World solved with a LangGraph graph whose decisions are taken by CLM.

Each loop iteration the `observe` node updates the agent's knowledge from the percepts and writes
a plain-English description of the situation. `create_router` hands that text to CLM, which picks
the intent that fits best (explore / grab the gold / leave); the router returns the node that
carries it out.

CLM is a semantic matcher: it tells "the gold is here" from "keep searching" reliably, but it does
not tell "turn left" from "turn right". So it chooses high-level intents, and the nodes do the
path-finding over the cells the agent has deduced to be safe.

Requires the llama embedding server (see README). Run with:  uv run python examples/wumpus.py
"""

import dataclasses
import enum
import typing

from langgraph.graph import END, START, StateGraph

from clm import Route, create_router

SIZE = 4
MAX_STEPS = 100
HEADINGS = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # east, north, west, south (turn left = +1)

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


def build_graph() -> typing.Any:
    # Default embedding provider: llama server with Qwen3-8B on :8090.
    router = create_router(state_selector=lambda s: s["situation"], routes=ROUTES)

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


def main() -> None:
    world = World(pits={(2, 0), (2, 2), (3, 3)}, wumpus=(0, 2), gold=(1, 2))
    result = build_graph().invoke(
        {"world": world, "known": {}, "situation": "", "log": []},
        {"recursion_limit": 200},
    )
    print("\n".join(result["log"]))
    outcome = "died" if not world.alive else "escaped" if world.escaped else "ran out of steps"
    print(f"\nAgent {outcome} after {world.steps} steps; gold={world.has_gold}; score={world.score}")


if __name__ == "__main__":
    main()
