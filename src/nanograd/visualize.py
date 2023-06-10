import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import figure

from src.nanograd.value_interface import ValueInterface


def trace(
    root: ValueInterface,
) -> tuple[set[ValueInterface], set[tuple[ValueInterface, ValueInterface]]]:
    nodes, edges = set(), set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.children:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_networkx(
    root: ValueInterface, name: str = "", rankdir: str = "LR"
) -> figure.Figure:
    """
    format: png | svg | ...
    rankdir: TB (top to bottom graph) | LR (left to right)
    """
    assert rankdir in ["LR", "TB"]
    nodes, edges = trace(root)
    graph = nx.DiGraph(attr={"rankdir": rankdir})
    # dot = Digraph(
    #     format=format, graph_attr={"rankdir": rankdir}
    # )  # , node_attr={'rankdir': 'TB'})

    for n in nodes:
        graph.add_node(
            n,
            name=str(id(n)),
            label=f"{{ {n.label} | data {n.data} | grad {n.grad}}}",
            shape="record",
        )
        # dot.node(
        #     name=str(id(n)),
        #     label=f"{{ {n.label} | data {n.data} | grad {n.grad}}}",
        #     shape="record",
        # )
        if n.operation:
            graph.add_node(n, name=str(id(n)) + n.operation, label=n.operation)
            graph.add_edge(str(id(n)) + n.operation, str(id(n)))
        # if n._op:
        #     dot.node(name=str(id(n)) + n._op, label=n._op)
        #     dot.edge(str(id(n)) + n._op, str(id(n)))

    for n1, n2 in edges:
        graph.add_edge(str(id(n1)), str(id(n2)) + n2.operation)
        # dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    nx.draw_networkx(graph, with_labels=True)

    if name != "":
        plt.savefig(f"{name}.png")

    return plt.show()
