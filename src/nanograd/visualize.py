from src.nanograd.value_interface import ValueInterface


def trace(
    root: ValueInterface,
) -> tuple[set[ValueInterface], set[tuple[ValueInterface, ValueInterface]]]:
    nodes, edges = set(), set()

    def build(v: ValueInterface):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_dot(root: ValueInterface, format: str = "svg", rankdir: str = "LR"):
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    from graphviz import Digraph

    nodes, edges = trace(root)
    dot = Digraph(
        format=format, graph_attr={"rankdir": rankdir}
    )  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        dot.node(
            name=str(id(n)),
            label=f"""{{ {n.label + " = " if n.label != "" else ""}{round(n.data, 5)} | grad {round(n.grad, 5)}}}""",
            shape="record",
        )
        if n.operator:
            dot.node(name=str(id(n)) + n.operator, label=n.operator)
            dot.edge(str(id(n)) + n.operator, str(id(n)))

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.operator)

    return dot
