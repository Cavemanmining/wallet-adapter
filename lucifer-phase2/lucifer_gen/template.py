"""Loading and validating graph templates.

Spec: docs/WORLD_BIBLE.md stage 1, "Load graph template".

A template is the authored part of a map: which rooms exist, what job each of
them does, which of them connect, and which of those connections the generator
may throw away.  Everything downstream assumes a template that has passed
:func:`validate_template`, so the checks here are the only place a malformed
template is allowed to be diagnosed.

The spec lists these rejections:

* an unknown role, shape or tile class,
* duplicate node ids,
* an edge naming a node that does not exist,
* not exactly one entrance,
* more than one boss,
* a disconnected graph,
* an optional edge whose removal would disconnect the graph.

Two readings had to be settled to make the last one usable:

*Optional nodes.*  A template may hang an optional node off a single optional
edge -- that is the whole point of, say, a mechanic room that only sometimes
appears.  Removing that edge does strand the node, so "disconnect the graph"
is read as *disconnect the required graph*: after dropping one optional edge,
every non-optional node must still be reachable from the entrance.  Optional
nodes are allowed to fall away, and stage 2 then omits them.

*Beyond the list.*  A few further checks are applied because they can only be
authoring mistakes and are far cheaper to catch here than downstream: an empty
node list, an anchor the shape does not define, a self-loop edge, a duplicate
edge, and more than one exit.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple

from .contracts import (
    GraphTemplate,
    Role,
    Shape,
    SpawnRules,
    TemplateEdge,
    TemplateNode,
    TileClass,
)
from .shapes import has_anchor

__all__ = [
    "TemplateError",
    "DATA_DIR",
    "template_from_dict",
    "template_to_dict",
    "load_template",
    "load_builtin_template",
    "builtin_template_names",
    "validate_template",
]

#: Where the shipped templates live.
DATA_DIR = Path(__file__).resolve().parent / "data"


class TemplateError(ValueError):
    """A template is malformed, or breaks one of the stage 1 rules."""


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------


def _enum_from_value(enum_cls, value: Any, what: str, where: str):
    """Look an enum up by its JSON spelling, with a message that lists the set."""
    if isinstance(value, enum_cls):
        return value
    if not isinstance(value, str):
        raise TemplateError(f"{where}: {what} must be a string, got {value!r}")
    for member in enum_cls:
        if member.value == value:
            return member
    known = ", ".join(sorted(m.value for m in enum_cls))
    raise TemplateError(f"{where}: unknown {what} {value!r}; known {what}s: {known}")


def _require(data: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in data:
        raise TemplateError(f"{where}: missing required field {key!r}")
    return data[key]


def _as_int(value: Any, what: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TemplateError(f"{where}: {what} must be an integer, got {value!r}")
    return value


def _as_float(value: Any, what: str, where: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TemplateError(f"{where}: {what} must be a number, got {value!r}")
    return float(value)


def _as_bool(value: Any, what: str, where: str) -> bool:
    if not isinstance(value, bool):
        raise TemplateError(f"{where}: {what} must be true or false, got {value!r}")
    return value


def _as_str(value: Any, what: str, where: str) -> str:
    if not isinstance(value, str):
        raise TemplateError(f"{where}: {what} must be a string, got {value!r}")
    return value


def _node_from_dict(data: Any, where: str) -> TemplateNode:
    if not isinstance(data, Mapping):
        raise TemplateError(f"{where}: each node must be an object, got {data!r}")
    node_id = _as_str(_require(data, "id", where), "node id", where)
    where = f"{where} node {node_id!r}"
    role = _enum_from_value(Role, _require(data, "role", where), "role", where)
    anchor = _as_str(_require(data, "anchor", where), "anchor", where)
    set_piece = data.get("set_piece")
    if set_piece is not None:
        set_piece = _as_str(set_piece, "set_piece", where)
    return TemplateNode(
        id=node_id,
        role=role,
        anchor=anchor,
        optional=_as_bool(data.get("optional", False), "optional", where),
        set_piece=set_piece,
    )


def _edge_from_dict(data: Any, where: str) -> TemplateEdge:
    if not isinstance(data, Mapping):
        raise TemplateError(f"{where}: each edge must be an object, got {data!r}")
    a = _as_str(_require(data, "a", where), "edge end 'a'", where)
    b = _as_str(_require(data, "b", where), "edge end 'b'", where)
    return TemplateEdge(
        a=a,
        b=b,
        optional=_as_bool(data.get("optional", False), "optional", f"{where} {a}-{b}"),
    )


def template_from_dict(
    data: Mapping[str, Any], *, source: str = "<dict>", validate: bool = True
) -> GraphTemplate:
    """Build a :class:`contracts.GraphTemplate` from decoded JSON.

    ``source`` only shapes error messages.  Unless ``validate`` is false the
    result is passed through :func:`validate_template` before it is returned,
    so a template that parses but breaks a stage 1 rule never escapes.
    """
    if not isinstance(data, Mapping):
        raise TemplateError(f"{source}: template must be a JSON object")

    tile_class_value = data.get("class", data.get("tile_class"))
    if tile_class_value is None:
        raise TemplateError(f"{source}: missing required field 'class'")

    nodes_raw = _require(data, "nodes", source)
    edges_raw = _require(data, "edges", source)
    if not isinstance(nodes_raw, Sequence) or isinstance(nodes_raw, (str, bytes)):
        raise TemplateError(f"{source}: 'nodes' must be a list")
    if not isinstance(edges_raw, Sequence) or isinstance(edges_raw, (str, bytes)):
        raise TemplateError(f"{source}: 'edges' must be a list")

    spawn_raw = data.get("spawn", {})
    if not isinstance(spawn_raw, Mapping):
        raise TemplateError(f"{source}: 'spawn' must be an object")
    spawn = SpawnRules(
        base_density=_as_float(
            spawn_raw.get("base_density", 0.018), "spawn.base_density", source
        ),
        elite_rate=_as_float(
            spawn_raw.get("elite_rate", 0.12), "spawn.elite_rate", source
        ),
    )

    landmarks_raw = data.get("landmarks", ())
    if isinstance(landmarks_raw, (str, bytes)) or not isinstance(
        landmarks_raw, Iterable
    ):
        raise TemplateError(f"{source}: 'landmarks' must be a list of strings")
    landmarks = tuple(
        _as_str(item, "landmark", source) for item in landmarks_raw
    )

    template = GraphTemplate(
        id=_as_str(_require(data, "id", source), "template id", source),
        version=_as_int(_require(data, "version", source), "version", source),
        tile_class=_enum_from_value(TileClass, tile_class_value, "class", source),
        shape=_enum_from_value(Shape, _require(data, "shape", source), "shape", source),
        grid=_as_int(_require(data, "grid", source), "grid", source),
        cell_m=_as_float(_require(data, "cell_m", source), "cell_m", source),
        nodes=tuple(
            _node_from_dict(n, f"{source} nodes[{i}]") for i, n in enumerate(nodes_raw)
        ),
        edges=tuple(
            _edge_from_dict(e, f"{source} edges[{i}]") for i, e in enumerate(edges_raw)
        ),
        tileset=_as_str(_require(data, "tileset", source), "tileset", source),
        landmarks=landmarks,
        spawn=spawn,
    )
    if validate:
        validate_template(template, source=source)
    return template


def template_to_dict(template: GraphTemplate) -> Dict[str, Any]:
    """The inverse of :func:`template_from_dict`, for tooling and round trips."""
    return {
        "id": template.id,
        "version": template.version,
        "class": template.tile_class.value,
        "shape": template.shape.value,
        "grid": template.grid,
        "cell_m": template.cell_m,
        "tileset": template.tileset,
        "landmarks": list(template.landmarks),
        "spawn": {
            "base_density": template.spawn.base_density,
            "elite_rate": template.spawn.elite_rate,
        },
        "nodes": [
            {
                "id": n.id,
                "role": n.role.value,
                "anchor": n.anchor,
                **({"optional": True} if n.optional else {}),
                **({"set_piece": n.set_piece} if n.set_piece else {}),
            }
            for n in template.nodes
        ],
        "edges": [
            {"a": e.a, "b": e.b, **({"optional": True} if e.optional else {})}
            for e in template.edges
        ],
    }


def load_template(path: "os.PathLike[str] | str", *, validate: bool = True) -> GraphTemplate:
    """Read one template from a JSON file."""
    p = Path(path)
    try:
        raw = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise TemplateError(f"cannot read template {p}: {exc}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TemplateError(f"{p}: invalid JSON: {exc}") from exc
    return template_from_dict(data, source=str(p), validate=validate)


#: The keys that mark a JSON file in ``data/`` as a graph template.  The same
#: directory also holds the room and tile libraries later stages load, so
#: listing templates has to look at content, not just at the extension.
_TEMPLATE_MARKERS = ("shape", "nodes", "edges")


def _looks_like_template(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return isinstance(data, Mapping) and all(k in data for k in _TEMPLATE_MARKERS)


def builtin_template_names() -> Tuple[str, ...]:
    """The templates shipped in ``lucifer_gen/data``, sorted for determinism."""
    if not DATA_DIR.is_dir():
        return ()
    return tuple(
        sorted(p.stem for p in DATA_DIR.glob("*.json") if _looks_like_template(p))
    )


def load_builtin_template(name: str, *, validate: bool = True) -> GraphTemplate:
    """Load a shipped template by file stem, e.g. ``"crypt"``."""
    path = DATA_DIR / f"{name}.json"
    if not path.is_file():
        known = ", ".join(builtin_template_names()) or "none"
        raise TemplateError(f"no built-in template {name!r}; available: {known}")
    return load_template(path, validate=validate)


# --------------------------------------------------------------------------
# Validation (stage 1)
# --------------------------------------------------------------------------


def _adjacency(
    node_ids: Iterable[str], edges: Iterable[TemplateEdge]
) -> Dict[str, List[str]]:
    """Undirected adjacency, with neighbour lists sorted for determinism."""
    adj: Dict[str, List[str]] = {n: [] for n in node_ids}
    for e in edges:
        adj[e.a].append(e.b)
        adj[e.b].append(e.a)
    for key in adj:
        adj[key].sort()
    return adj


def _reachable(start: str, adj: Mapping[str, List[str]]) -> Set[str]:
    """Depth-first reachability from ``start``; deterministic order in, set out."""
    seen = {start}
    stack = [start]
    while stack:
        current = stack.pop()
        for nxt in adj[current]:
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def validate_template(template: GraphTemplate, *, source: str = "") -> None:
    """Run every stage 1 rule, raising :class:`TemplateError` on the first break.

    See the module docstring for the exact rules and for the two readings this
    implementation had to settle.
    """
    where = f"{source}: " if source else f"template {template.ref}: "

    # --- shell ---------------------------------------------------------
    if not isinstance(template.tile_class, TileClass):
        raise TemplateError(f"{where}unknown class {template.tile_class!r}")
    if not isinstance(template.shape, Shape):
        raise TemplateError(f"{where}unknown shape {template.shape!r}")
    if template.grid < 8:
        raise TemplateError(f"{where}grid must be at least 8 cells, got {template.grid}")
    if template.cell_m <= 0:
        raise TemplateError(f"{where}cell_m must be positive, got {template.cell_m}")
    if not template.nodes:
        raise TemplateError(f"{where}template has no nodes")

    # --- nodes ---------------------------------------------------------
    seen_ids: Set[str] = set()
    for node in template.nodes:
        if not isinstance(node.role, Role):
            raise TemplateError(f"{where}node {node.id!r} has unknown role {node.role!r}")
        if node.id in seen_ids:
            raise TemplateError(f"{where}duplicate node id {node.id!r}")
        seen_ids.add(node.id)
        if not has_anchor(template.shape, node.anchor):
            raise TemplateError(
                f"{where}node {node.id!r} uses anchor {node.anchor!r}, "
                f"which shape {template.shape.value} does not define"
            )

    # --- roles ---------------------------------------------------------
    def ids_with(role: Role) -> List[str]:
        return [n.id for n in template.nodes if n.role is role]

    entrances = ids_with(Role.ENTRANCE)
    if len(entrances) != 1:
        raise TemplateError(
            f"{where}expected exactly one entrance, found {len(entrances)}"
            + (f": {', '.join(entrances)}" if entrances else "")
        )
    entrance = entrances[0]
    if template.node(entrance).optional:
        raise TemplateError(f"{where}the entrance {entrance!r} may not be optional")

    bosses = ids_with(Role.BOSS)
    if len(bosses) > 1:
        raise TemplateError(
            f"{where}at most one boss allowed, found {len(bosses)}: {', '.join(bosses)}"
        )
    exits = ids_with(Role.EXIT)
    if len(exits) > 1:
        raise TemplateError(
            f"{where}at most one exit allowed, found {len(exits)}: {', '.join(exits)}"
        )

    # --- edges ---------------------------------------------------------
    seen_edges: Set[Tuple[str, str]] = set()
    for edge in template.edges:
        for end in (edge.a, edge.b):
            if end not in seen_ids:
                raise TemplateError(
                    f"{where}edge {edge.a!r}-{edge.b!r} names unknown node {end!r}"
                )
        if edge.a == edge.b:
            raise TemplateError(f"{where}edge {edge.a!r}-{edge.b!r} is a self-loop")
        key = (edge.a, edge.b) if edge.a < edge.b else (edge.b, edge.a)
        if key in seen_edges:
            raise TemplateError(f"{where}duplicate edge {edge.a!r}-{edge.b!r}")
        seen_edges.add(key)

    # --- connectivity --------------------------------------------------
    node_ids = sorted(seen_ids)  # sorted: never let set order reach the output
    adj = _adjacency(node_ids, template.edges)
    reached = _reachable(entrance, adj)
    if len(reached) != len(seen_ids):
        stranded = sorted(seen_ids - reached)
        raise TemplateError(
            f"{where}graph is disconnected; unreachable from {entrance!r}: "
            f"{', '.join(stranded)}"
        )

    # An optional edge may only be one that the map can do without: dropping
    # it must leave every non-optional node reachable from the entrance.
    required_ids = {n.id for n in template.nodes if not n.optional}
    for edge in template.edges:
        if not edge.optional:
            continue
        without = [e for e in template.edges if e is not edge]
        reached_without = _reachable(entrance, _adjacency(node_ids, without))
        stranded = sorted(required_ids - reached_without)
        if stranded:
            raise TemplateError(
                f"{where}optional edge {edge.a!r}-{edge.b!r} cannot be dropped; "
                f"its removal strands required node(s): {', '.join(stranded)}"
            )
