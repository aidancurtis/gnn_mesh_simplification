from trimesh import Trimesh
import networkx as nx


def mesh_to_graph(mesh: Trimesh):
    G = nx.Graph()

    for i, vertex in enumerate(mesh.vertices):
        G.add_node(i, pos=vertex)

    for face in mesh.faces:
        G.add_edge(face[0], face[1])
        G.add_edge(face[1], face[2])
        G.add_edge(face[2], face[0])

    return G


def view_mesh(mesh: Trimesh):
    mesh.show()
