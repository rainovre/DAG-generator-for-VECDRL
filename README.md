# DAG-generator-for-VECDRL
hello


requirements(based Python 3.9):
```
matplotlib==3.10.8
networkx==3.2.1
numpy==2.4.1
```
code:
```python
import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import os
import ast
from matplotlib.patches import FancyArrowPatch


def generate_random_dag(n=100, density=0.2, seed=None):
    """
    Generate a random DAG from 0 to n-1.
    Force 0 to be Start, and n-1 to be Exit.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # 1. Generate basic random graph (upper triangular) 
    # # node range 0 to n-1
    mask = np.triu(np.ones((n, n)), k=1) 
    rand_matrix = np.random.rand(n, n)
    adj_matrix = (rand_matrix < density) * mask
    
    edges = []
    rows, cols = np.nonzero(adj_matrix)
    for r, c in zip(rows, cols):
        edges.append((r, c)) # 0-based

    # 2.Connectivity repair 
    # # We want the graph to flow from 0 to n-1
    # 2a. Ensure that all nodes, except for node 0, have inbound edges. 
    # # If node i does not have an inbound edge, randomly connect it to a j (j < i)
    nodes_with_indegree = set(v for u, v in edges)
    for i in range(1, n): 
        if i not in nodes_with_indegree:
            parent = random.randint(0, i - 1)
            edges.append((parent, i))

    # 2b. Ensure that all nodes except for node n-1 have outgoing edges. 
    # If node i does not have an outgoing edge, connect it to a node j (j > i)
    nodes_with_outdegree = set(u for u, v in edges)
    for i in range(0, n - 1): 
        if i not in nodes_with_outdegree:
            # Prefer to connect to n-1, or any node after i
            child = random.randint(i + 1, n - 1)
            edges.append((i, child))
            
    # 3.  Remove duplicate edges and sort them
    edges = sorted(list(set(edges)))
    
    # Simple position generation (for drawing)
    pos = {}
    pos[0] = (0, 0) 
    pos[n-1] = (n, 0)
    for i in range(1, n-1):
        pos[i] = (i, random.uniform(-5, 5))
        
    return edges, pos

def plot_DAG(edges,pos, filename="DAG_controlled_layers.pdf", max_nodes_per_layer=8):
    
    Y_SPACING_UNIT = 0.8   
    X_SPACING_UNIT = 0.8    
    
    NODE_SIZE = 1200      
    SHRINK_AMOUNT = 17      
    
    ARROW_COLOR = "#335127" 
    LINE_WIDTH = 1.5        
    MUTATION_SCALE = 20    

    
    g1 = nx.DiGraph()
    g1.add_edges_from(edges)
    nodes = set(g1.nodes())
    pos = {} 
    
    entry_nodes = [n for n in nodes if g1.in_degree(n) == 0]
    exit_nodes = [n for n in nodes if g1.out_degree(n) == 0]
    intermediate_nodes = list(nodes - set(entry_nodes) - set(exit_nodes))

    try:
        ordered_nodes_list = list(nx.topological_sort(g1))
        sorted_intermediate_nodes = [n for n in ordered_nodes_list if n in intermediate_nodes]
    except nx.NetworkXUnfeasible:
        print("warning")
        sorted_intermediate_nodes = sorted(intermediate_nodes)

    current_layer = 0
    
    # A.
    pos_top_layer = {}
    num_entries = len(entry_nodes)
    if num_entries > 0:
        y_coord = current_layer * -Y_SPACING_UNIT
        for i, node in enumerate(entry_nodes):
            x_coord = (i - (num_entries - 1) / 2) * X_SPACING_UNIT
            pos_top_layer[node] = (x_coord, y_coord)
        pos.update(pos_top_layer)
        current_layer += 1
    
    # B. 
    current_layer_index = 0
    layer_y_coord = current_layer * -Y_SPACING_UNIT
    
    for i, node in enumerate(sorted_intermediate_nodes):
        if current_layer_index >= max_nodes_per_layer:
            current_layer += 1
            current_layer_index = 0
            layer_y_coord = current_layer * -Y_SPACING_UNIT

        pos[node] = (current_layer_index * X_SPACING_UNIT, layer_y_coord)
        current_layer_index += 1
        
    if current_layer_index > 0:
         current_layer += 1
         
    for node, (x, y) in pos.items():
        if node in intermediate_nodes:
            
             nodes_in_this_layer = [n for n, (px, py) in pos.items() if abs(py - y) < 1e-6 and n in intermediate_nodes]
             num_nodes_in_layer = len(nodes_in_this_layer)
             
             center_offset = (num_nodes_in_layer - 1) / 2 * X_SPACING_UNIT
             
             current_index = nodes_in_this_layer.index(node)
             new_x = (current_index * X_SPACING_UNIT) - center_offset
             pos[node] = (new_x, y)


    # C. 
    pos_bottom_layer = {}
    num_exits = len(exit_nodes)
    if num_exits > 0:
        y_coord = current_layer * -Y_SPACING_UNIT
        for i, node in enumerate(exit_nodes):
            x_coord = (i - (num_exits - 1) / 2) * X_SPACING_UNIT
            pos_bottom_layer[node] = (x_coord, y_coord)
        pos.update(pos_bottom_layer)
        current_layer += 1


    max_width = max(max_nodes_per_layer, num_entries, num_exits) * X_SPACING_UNIT * 1.5
    height = current_layer * Y_SPACING_UNIT * 1.5
    
    plt.figure(figsize=(min(12, max_width), min(10, height)))
    ax = plt.gca() 

    nx.draw_networkx_nodes(
        g1, pos,
        node_size=NODE_SIZE,
        node_color='#D5EBC7',
        edgecolors=ARROW_COLOR,
        linewidths=2.5
    )

    labels = {node: str(node) for node in g1.nodes()}
    nx.draw_networkx_labels(
        g1, pos,
        labels=labels,
        font_size=15,
        font_weight="bold"
    )
    
    for u, v in g1.edges():
        xyA = pos[u]
        xyB = pos[v]

        arrow = FancyArrowPatch(
            posA=xyA, posB=xyB, 
            shrinkA=0,              
            shrinkB=SHRINK_AMOUNT,  
            connectionstyle="arc3,rad=0.1", 
            arrowstyle='-|>', 
            mutation_scale=MUTATION_SCALE, 
            color=ARROW_COLOR,      
            linewidth=LINE_WIDTH, 
            alpha=1                 
        )
        ax.add_patch(arrow)
    
    plt.axis('off')
    plt.tight_layout(pad=1.0)
    
    # save to PDF
    # plt.savefig(r"\fig\task30_0.5.pdf")
    plt.show()
    print(f"The DAG image has been saved to the following location according to the custom layering rule: {filename}")

def write_list_to_py_append(filename, data_list, variable_name):
    if not filename.endswith(".py"):
        filename += ".py"
        
    try:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'a', encoding='utf-8') as f:
            list_as_string = str(data_list)
            f.write(f"{variable_name} = {list_as_string}\n")
            
        print(f"[{variable_name}] written {filename}")

    except Exception as e:
        print(f"error: {e}")

def read_lists_from_simple_file(filename):
    """
    Read a list from a .py file.
    """
    if not filename.endswith(".py"):
        filename += ".py"
    all_lists = []
    if not os.path.exists(filename):
        print(f"Error: File not found {filename}")
        return []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'): continue
                try:
                    split_index = line.index('=')
                    list_string = line[split_index + 1:].strip()
                    if list_string.startswith('[') and list_string.endswith(']'):
                        actual_list = ast.literal_eval(list_string)
                        all_lists.append(actual_list)
                except: continue
    except Exception as e:
        print(f"error: {e}")
        return []
    return all_lists

def g_dag_b(i, n=100, density=0.2):
    """
    The main function for generating and saving DAG.
    """
    # 1. generate random DAG
    edges, pos = generate_random_dag(n=n, density=density, seed=i) 
    
    # 2. plot
    if i == 0:
        plot_DAG(edges, pos, filename=f"DAG_sample_{i}.png")

    # 3. 
    output_filename = f"Task_List/task_data_{n}" 
    variable_in_file = f"task_dependencies_{i}"

    # 4. 
    write_list_to_py_append(output_filename, edges, variable_in_file)

if __name__ == "__main__":

    NUM_TASKS = 30  # Number of task nodes
    NUM_DAGS = 1   # How many DAGs should be generated
    DENSITY = 0.5    # Sparsity (Above 0.3 is more suitable for generating complex structures)

    target_file = f"Task_List/task_data_{NUM_TASKS}.py"
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f"Old files have been deleted: {target_file}")

    print(f"Starting to generate {NUM_DAGS} random DAGs (N={NUM_TASKS}, Density={DENSITY})...")
    
    for i in range(NUM_DAGS):
        g_dag_b(i, n=NUM_TASKS, density=DENSITY)
        
    print("Generation complete!")
```
