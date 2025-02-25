#librerie importate
import numpy as np
import random
import networkx as nx
import matplotlib.pyplot as plt

#funzione che genera un albero (al più ternario) fortemente connesso
def generate_ternary_tree(num_nodes):#prendendo in input il numero di nodi da avere nell'albero ne generiamo uno casualmente
    if num_nodes <= 1:
        print("invalid input")
        return None
    
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    
    for i in range(1, num_nodes-1):  # Iniziamo dal nodo 1 (0 è la radice)
        parent = random.choice(range(i))  # Scegli un nodo già creato
        adjacency_matrix[parent][i] = 1
        adjacency_matrix[parent][i+1] = 1
        i++
    
    return adjacency_matrix+adjacency_matrix.T #la ritorniamo in questo modo così abbiamo che il grafo orientato può essere percorso in entrambe le direzioni

#funzione che implementa il page rank con alcune modifiche
def pagerank(adjacency_matrix, alpha=0.85, tol=1.0e-6, max_iter=100):
    #adjacency_matrix è la matrice di adiacenza dell'albero
    #alpha il damping fanctor
    #tol valore di tolleranza
    #max_iter numero massimo di iterazioni
    n = len(adjacency_matrix)
    M = adjacency_matrix / np.sum(adjacency_matrix, axis=0, where=adjacency_matrix!=0)  # Normalizzazione colonne
    
    v = np.zeros(n)  # Inizializzazione: tutto a zero
    v[0] = 1.0  # Partiamo dal nodo 0
    
    for _ in range(max_iter):
        v_new = alpha * np.dot(M,v) + (1 - alpha) / n
        if np.linalg.norm(v_new - v, 1) < tol:
            break
        v = v_new
    
    return v    

#funzione che disegna il grafo (che rapresenta il labirinto che vogliamo generare)
def draw_graph(adjacency_matrix, exit):
    #adjacency_matrix: matrice di adiacenza del grafo
    #exit: nodo dove vogliamo che sia l'uscita del labirinto
    num_nodes = len(adjacency_matrix)
    G = nx.DiGraph()
    
    for i in range(num_nodes):
        G.add_node(i)
        for j in range(num_nodes):
            if adjacency_matrix[i][j] == 1:
                G.add_edge(i, j)
    
    pos = nx.spring_layout(G)
    #mettiamo in rosso il nodo 0 "ingresso" del labirinto e il nodo exit "uscita" del labirinto 
    node_colors = ['red' if (i == 0 or i==exit) else 'lightblue' for i in range(num_nodes)]
    
    nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color='gray', node_size=2000, font_size=10)
    plt.show()

def mef(vec):
    """
    Trova la posizione del valore minimo di un vettore unidimensionale escludendo il primo elemento.
    """
       if len(vec) <= 1:
        raise ValueError("Il vettore deve contenere almeno due elementi.")
    min_index = np.argmin(vec[1:]) + 1  # Aggiungiamo 1 per compensare l'indice
    return min_index

# Esempio di utilizzo
num_nodi = 10
adj_matrix = generate_ternary_tree(num_nodi)
print(adj_matrix)
print("")
PRk_Vector = pagerank(adj_matrix)
print(PRk_Vector)
print("")
#prendiamo il nodo associato al valore minimo del vettore che ci ritorna il PageRank come uscita del nostro labirinto
draw_graph (adj_matrix, mef(PRk_Vector))
