import heapq
import networkx as nx
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from collections import defaultdict
import numpy as np

# Djikstra
def dijkstra(graph, start, goal):
    pq = [(0, start)]  
    jarak = {node: float('inf') for node in graph}
    jarak[start] = 0
    came_from = {}
    
    while pq:
        current_distance, current_node = heapq.heappop(pq)
        
        if current_node == goal:
            return reconstruct_path(came_from, start, goal)

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < jarak[neighbor]:
                jarak[neighbor] = distance
                heapq.heappush(pq, (distance, neighbor))
                came_from[neighbor] = current_node
    return None

def reconstruct_path(came_from, start, goal):
    path = []
    current_node = goal
    while current_node in came_from:
        path.append(current_node)
        current_node = came_from[current_node]
    path.append(start)
    return path[::-1]

# Fungsi heuristik untuk A*
def heuristic(node, goal):
    return 1  

def a_star(graph, start, goal):
    pq = [(0, start)]
    g_costs = {node: float('inf') for node in graph}
    g_costs[start] = 0
    came_from = {}
    
    while pq:
        current_cost, current_node = heapq.heappop(pq)

        if current_node == goal:
            return reconstruct_path(came_from, start, goal)

        for neighbor, weight in graph[current_node].items():
            new_cost = g_costs[current_node] + weight
            if new_cost < g_costs[neighbor]:
                g_costs[neighbor] = new_cost
                priority = new_cost + heuristic(neighbor, goal)
                heapq.heappush(pq, (priority, neighbor))
                came_from[neighbor] = current_node
    return None

# Untuk graf
def draw_graph(graph, path):
    G = nx.DiGraph()
    for node in graph:
        for neighbor, weight in graph[node].items():
            G.add_edge(node, neighbor, weight=weight)
    pos = nx.spring_layout(G)
    labels = nx.get_edge_attributes(G, 'weight')

    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=12)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)

    if path:
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)

    plt.title("Graph Visualization")
    plt.show()

# Fungsi menyimpan hasil ke PDF
def save_to_pdf(filename, algorithm, start, goal, path, path_type):
    c = canvas.Canvas(filename, pagesize=letter)
    c.drawString(100, 750, f"Algoritma: {algorithm}")
    c.drawString(100, 730, f"Titik Awal: {start}")
    c.drawString(100, 710, f"Titik Tujuan: {goal}")
    c.drawString(100, 690, f"Jenis Jalur: {path_type}")

    c.drawString(100, 670, "Jalur yang Dipilih:")
    c.drawString(120, 650, " -> ".join(path) if path else "Tidak ditemukan jalur.")

    c.drawString(100, 620, "Deskripsi:")
    c.drawString(120, 600, "Jalur tercepat = waktu tempuh minim")
    c.drawString(120, 580, "Jalur aman = tidak melewati area berbahaya")
    c.drawString(120, 560, "Jalur jauh & aman = menghindari semua ancaman")

    c.save()
    print(f"Hasil telah disimpan ke {filename}")

# **PROGRAM UTAMA**
if __name__ == "__main__":
    while True:
        graph = defaultdict(dict)
        safety = {}

        # masukan jumlah simpul
        n = int(input("Masukkan jumlah simpul: "))
        
        # Input node dan bobotnya
        for _ in range(n):
            node = input("Masukkan nama simpul: ")
            graph[node] = {}
            safe = input(f"Apakah simpul {node} aman? (ya/tidak): ").lower()
            safety[node] = safe == "ya"

        for node in graph:
            edges = int(input(f"Masukkan jumlah tetangga untuk simpul {node}: "))
            for _ in range(edges):
                neighbor, weight = input(f"  Tetangga dan bobot (cth: B 4): ").split()
                graph[node][neighbor] = int(weight)

        # Input titik awal dan tujuan
        start = input("Masukkan titik awal: ")
        goal = input("Masukkan titik tujuan: ")

        # Validasi input
        if start not in graph or goal not in graph:
            print("Titik awal atau tujuan tidak valid!")
            continue

        # Pilih jenis jalur
        print("\n==========Pilih Jenis Jalur:========")
        print("1. Tercepat & Aman ✅")
        print("2. Tercepat Tidak Aman ⚠️")
        print("3. Jalur Jauh Tapi Aman 🛡️")
        choice = int(input("Masukkan pilihan (1/2/3): "))
        print("=======================================")

        selected_path = None
        path_type = ""

        if choice == 1:
            safe_graph = {node: {n: w for n, w in neighbors.items() if safety[n]} for node, neighbors in graph.items()}
            selected_path = dijkstra(safe_graph, start, goal)
            path_type = "Tercepat & Aman ✅"

        elif choice == 2:
            selected_path = dijkstra(graph, start, goal)
            path_type = "Tercepat Tidak Aman ⚠️"

        # Jalur Jauh Tapi Aman (menggunakan jalur terjauh dengan hanya node aman)
        elif choice == 3:
            safe_graph = {node: {n: w for n, w in neighbors.items() if safety[n]} for node, neighbors in graph.items()}
            selected_path = dijkstra(safe_graph, start, goal)
            path_type = "Jalur Jauh Tapi Aman 🛡️"
        
        # simpan ke PDF
        if selected_path:
            print(f"Hasil Jalur ({path_type}): {' -> '.join(selected_path)}")
            save_to_pdf("hasil_rute.pdf", "Dijkstra", start, goal, selected_path, path_type)
            draw_graph(graph, selected_path)
        else:
            print("Tidak ditemukan jalur yang sesuai!")

        repeat = input("Apakah Anda ingin melakukan pencarian lagi? (ya/tidak): ").lower()
        if repeat != 'ya':
            break