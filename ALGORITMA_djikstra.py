import heapq
import networkx as nx
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from collections import defaultdict
# pdf format
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer


# Djikstra Algorithm
def dijkstra(graph, start, goal):
    pq = [(0, start)]
    jarak = {node: float('inf') for node in graph}
    jarak[start] = 0
    came_from = {}

    while pq:
        jarakSaatIni, current_node = heapq.heappop(pq)

        if current_node == goal:
            return reconstruct_path(came_from, start, goal)

        for neighbor, weight in graph[current_node].items():
            distance = jarakSaatIni + weight
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

# Fungsi untuk menggambar graf
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

# Fungsi untuk menyimpan hasil ke PDF
def save_to_pdf(filename, algorithm, start, goal, path, path_type):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    elements = []

    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    normal_style = styles["Normal"]

    # Judul
    title = Paragraph(f"<b>Hasil Analisis Jalur Terbaik</b>", title_style)
    elements.append(title)
    elements.append(Spacer(1, 12))

    # Informasi Dasar
    info_text = f"""
    <b>Algoritma:</b> {algorithm} <br/>
    <b>Titik Awal:</b> {start} <br/>
    <b>Titik Tujuan:</b> {goal} <br/>
    <b>Jenis Jalur:</b> {path_type} <br/>
    """
    elements.append(Paragraph(info_text, normal_style))
    elements.append(Spacer(1, 12))

    # Garis pemisah
    elements.append(Paragraph("<hr/>", normal_style))
    elements.append(Spacer(1, 12))

    # Jalur yang Dipilih
    if path:
        path_text = "<b>Jalur yang Dipilih:</b><br/>"
        path_text += " → ".join([f"<b>{i+1}.</b> {node}" for i, node in enumerate(path)])
    else:
        path_text = "<b>Jalur yang Dipilih:</b> Tidak ditemukan jalur yang sesuai."
    
    elements.append(Paragraph(path_text, normal_style))
    elements.append(Spacer(1, 12))

    # Simpan ke PDF
    doc.build(elements)
    print(f"Hasil telah disimpan ke {filename}")

while True:
    graph = defaultdict(dict)
    safety = {}
    while True:
        try:
            print("============= Masukkan Data Simpul ==============")
            n = int(input("Masukkan jumlah simpul: "))
            if n <= 0:
                print("Jumlah simpul harus lebih dari 0! Silakan coba lagi.")
            else:
                break
        except ValueError:
            print("Input tidak valid! Harap masukkan angka.")

    for _ in range(n):
        while True:
            node = input("Masukkan nama simpul: ").strip().upper()
            if not node:
                print("Nama simpul tidak boleh kosong! Silakan coba lagi.")
            elif node in graph:
                print(f"Simpul {node} sudah ada! Gunakan nama lain.")
            else:
                break

        graph[node] = {}

        while True:
            safe_input = input(f"Apakah simpul {node} aman? (ya/tidak): ").strip().lower()
            if safe_input in ["ya", "tidak"]:
                safety[node] = safe_input == "ya"
                break
            else:
                print("Input tidak valid! Harap masukkan 'ya' atau 'tidak'.")

    # Input tetangga dan bobotnya
    for node in graph:
        while True:
            try:
                print("============= Masukkan Data Tetangga ==============")
                edges = int(input(f"Masukkan jumlah tetangga untuk simpul {node}: "))
                if edges < 0:
                    print("Jumlah tetangga tidak boleh negatif! Coba lagi.")
                else:
                    break
            except ValueError:
                print("Input tidak valid! Harap masukkan angka.")

        for _ in range(edges):
            while True:
                try:
                    neighbor, weight = input(f"  Tetangga dan bobot (cth: B 4): ").split()
                    neighbor = neighbor.upper()

                    if neighbor not in graph:
                        print(f"Simpul {neighbor} belum terdaftar! Pastikan Anda memasukkan simpul yang sudah dibuat.")
                        continue
                    
                    weight = int(weight)
                    if weight <= 0:
                        print("Bobot harus lebih dari 0! Silakan coba lagi.")
                        continue
                    
                    if neighbor in graph[node]:
                        print(f"Simpul {neighbor} sudah menjadi tetangga dari {node}! Coba lagi.")
                        continue

                    graph[node][neighbor] = weight
                    break

                except ValueError:
                    print("Format salah! Gunakan format yang benar (contoh: B 4).")

    # Input titik awal dan tujuan
    while True:
        print("\n============= Masukkan Titik Awal & Tujuan ==============")
        start = input("Masukkan titik awal: ").strip().upper()
        goal = input("Masukkan titik tujuan: ").strip().upper()
        print("============================================================")
        if start in graph and goal in graph:
            break
        else:
            print("Titik awal atau tujuan tidak valid! Pastikan Anda memasukkan simpul yang benar.")

    # Memilih jenis jalur
    while True:
        print("\n==========Pilih Jenis Jalur:========")
        print("1. Tercepat & Aman ✅")
        print("2. Tercepat Tidak Aman ⚠️")
        print("3. Jalur Jauh Tapi Aman 🛡️")
        try:
            choice = int(input("Masukkan pilihan (1/2/3): "))
            if choice in [1, 2, 3]:
                break
            else:
                print("Pilihan tidak valid! Masukkan angka 1, 2, atau 3.")
        except ValueError:
            print("Input harus berupa angka!")

    selected_path = None
    path_type = ""

    if choice == 1:
        safe_graph = {node: {n: w for n, w in neighbors.items() if safety[n]} for node, neighbors in graph.items()}
        selected_path = dijkstra(safe_graph, start, goal)
        path_type = "Tercepat & Aman ✅"

    elif choice == 2:
        selected_path = dijkstra(graph, start, goal)
        path_type = "Tercepat Tidak Aman ⚠️"

    elif choice == 3:
        safe_graph = {node: {n: w for n, w in neighbors.items() if safety[n]} for node, neighbors in graph.items()}
        selected_path = dijkstra(safe_graph, start, goal)
        path_type = "Jalur Jauh Tapi Aman 🛡️"

    if selected_path:
        print(f"Hasil Jalur ({path_type}): {' -> '.join(selected_path)}")
        save_to_pdf("hasil_rute.pdf", "Dijkstra", start, goal, selected_path, path_type)
        draw_graph(graph, selected_path)
    else:
        print("Tidak ditemukan jalur yang sesuai!")
        
    while True:
        lanjutkan = input("\nTekan ENTER untuk lanjut atau ketik 'tidak' untuk keluar: ").strip().lower()
        if lanjutkan == "tidak":
            exit()  
        elif lanjutkan == "":
            print("Terima kasih telah menggunakan program ini! 🚀")
            break  
        else:
            print("Input tidak valid! Silakan tekan ENTER untuk lanjut atau ketik 'tidak' untuk keluar.")
