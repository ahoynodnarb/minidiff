from diff_graph import FuncNode
import networkx as nx

def print_graph(root: FuncNode):
    if root is None:
        return
    print(root.pretty_repr(), sep=" ")
    for node in root.input_nodes:
        print_graph(node)
    print()
    
# def draw_graph(root: FuncNode):
#     if root is None:
#         return
#     G = nx.graph()
    