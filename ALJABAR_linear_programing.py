import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from typing import Tuple, List, Optional
import sys

class LinearAlgebraTools:
    """Kelas untuk operasi aljabar linear dengan fungsi yang lebih terstruktur."""
    
    def __init__(self):
        # Matriks default yang akan digunakan di beberapa fungsi
        self.default_matrix_a = np.array([[2, 3], [1, 4]])
        self.default_matrix_b = np.array([[5, 1], [2, 3]])
    
    def operasi_matriks(self, 
        A: Optional[np.ndarray] = None, 
        B: Optional[np.ndarray] = None) -> dict:
        """
        Melakukan operasi dasar pada matriks.
        
        Args:
            A: Matriks pertama, default ke matriks predefined jika None
            B: Matriks kedua, default ke matriks predefined jika None
            
        Returns:
            Dictionary hasil operasi matriks
        """
        A = self.default_matrix_a if A is None else A
        B = self.default_matrix_b if B is None else B
        
        results = {
            "Matrix A": A,
            "Matrix B": B,
            "A + B": A + B,
            "A * B (dot product)": A @ B,
        }
        
        try:
            results["Inverse of A"] = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            results["Inverse of A"] = "Matriks tidak memiliki invers"
        
        return results
    
    def eigenvalue_eigenvector(self, A: Optional[np.ndarray] = None) -> dict:
        """
        Menghitung nilai eigen dan vektor eigen dari matriks.
        
        Args:
            A: Matriks input, default ke matriks predefined jika None
            
        Returns:
            Dictionary hasil nilai eigen dan vektor eigen
        """
        A = self.default_matrix_a if A is None else A
        
        try:
            values, vectors = np.linalg.eig(A)
            return {
                "Matrix": A,
                "Eigenvalues": values,
                "Eigenvectors": vectors
            }
        except np.linalg.LinAlgError as e:
            return {
                "Matrix": A,
                "Error": f"Gagal menghitung nilai eigen: {str(e)}"
            }
    
    def singular_value_decomposition(self, A: Optional[np.ndarray] = None) -> dict:
        """
        Melakukan dekomposisi nilai singular pada matriks.
        
        Args:
            A: Matriks input, default ke matriks predefined jika None
            
        Returns:
            Dictionary hasil SVD
        """
        A = self.default_matrix_a if A is None else A
        
        try:
            U, S, Vt = svd(A)
            return {
                "Matrix": A,
                "U": U,
                "Singular Values": S,
                "V transpose": Vt,
                "Reconstructed": U @ np.diag(S) @ Vt
            }
        except np.linalg.LinAlgError as e:
            return {
                "Matrix": A,
                "Error": f"Gagal melakukan SVD: {str(e)}"
            }
    
    def gradient_descent(self,
                        A: np.ndarray,
                        b: np.ndarray,
                        lr: float = 0.01,
                        iterations: int = 1000,
                        tolerance: float = 1e-6) -> Tuple[np.ndarray, List[float]]:
        """
        Implementasi gradient descent dengan kriteria konvergensi.
        
        Args:
            A: Matriks fitur
            b: Vektor target
            lr: Learning rate
            iterations: Jumlah maksimum iterasi
            tolerance: Toleransi minimum untuk konvergensi
            
        Returns:
            Tuple berisi bobot optimal dan history loss
        """
        x = np.zeros((A.shape[1], 1))
        loss_history = []
        
        for i in range(iterations):
            # Hitung prediksi dan error
            pred = A @ x
            error = b - pred
            loss = np.mean(error ** 2)
            loss_history.append(loss)
            
            # Hitung gradien dan update bobot
            gradient = -2 * A.T @ error / len(b)
            x_new = x - lr * gradient
            
            # Cek konvergensi
            if np.linalg.norm(x_new - x) < tolerance:
                print(f"Konvergen setelah {i+1} iterasi")
                break
                
            x = x_new
            
            # Tampilkan progress setiap 200 iterasi
            if (i + 1) % 200 == 0:
                print(f"Iterasi {i+1}: Loss = {loss:.6f}")
        
        return x, loss_history
    
    def run_gradient_descent(self, visualize: bool = True) -> dict:
        """
        Menjalankan metode gradient descent pada data contoh linear.
        
        Args:
            visualize: Boolean untuk menampilkan visualisasi hasil
            
        Returns:
            Dictionary hasil gradient descent
        """
        # Buat data linear sederhana
        X = np.array([[1, 1], [1, 2], [1, 3], [1, 4]])
        y = np.array([[1], [2], [3], [4]])
        
        # Jalankan gradient descent
        weights, loss_history = self.gradient_descent(X, y)
        
        # Prediksi
        predictions = X @ weights
        
        # Hitung R-squared
        y_mean = np.mean(y)
        ss_total = np.sum((y - y_mean) ** 2)
        ss_residual = np.sum((y - predictions) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Visualisasi jika diminta
        if visualize:
            plt.figure(figsize=(12, 5))
            
            # Plot loss history
            plt.subplot(1, 2, 1)
            plt.plot(loss_history)
            plt.title('Loss per Iterasi')
            plt.xlabel('Iterasi')
            plt.ylabel('Mean Squared Error')
            
            # Plot hasil regresi
            plt.subplot(1, 2, 2)
            plt.scatter(X[:, 1], y, color='blue', label='Data Asli')
            plt.plot(X[:, 1], predictions, color='red', label='Prediksi')
            plt.title('Hasil Regresi Linear')
            plt.xlabel('X')
            plt.ylabel('y')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
        
        return {
            "Data X": X,
            "Data y": y,
            "Bobot Optimal": weights,
            "Prediksi": predictions,
            "R-squared": r_squared,
            "Final Loss": loss_history[-1] if loss_history else None
        }
    
    def transformasi_linear(self, 
                           vector: Optional[np.ndarray] = None,
                           angle: float = np.pi/4,
                           visualize: bool = True) -> dict:
        """
        Melakukan transformasi linear (rotasi) pada vektor.
        
        Args:
            vector: Vektor yang akan dirotasi, default ke [2, 1] jika None
            angle: Sudut rotasi dalam radian
            visualize: Boolean untuk menampilkan visualisasi hasil
            
        Returns:
            Dictionary hasil transformasi linear
        """
        if vector is None:
            vector = np.array([[2], [1]])
        elif vector.ndim == 1:
            vector = vector.reshape(-1, 1)
        
        # Buat matriks rotasi
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        
        # Lakukan rotasi
        rotated_vector = rotation_matrix @ vector
        
        # Visualisasi jika diminta
        if visualize:
            self._plot_vectors([vector.flatten(), rotated_vector.flatten()], 
                              ['blue', 'red'],
                              ['Vektor Asli', 'Vektor Hasil Rotasi'])
        
        return {
            "Sudut Rotasi": f"{angle:.4f} rad ({angle * 180/np.pi:.2f}°)",
            "Matriks Rotasi": rotation_matrix,
            "Vektor Asli": vector,
            "Vektor Hasil Rotasi": rotated_vector
        }
    
    def _plot_vectors(self, 
                     vectors: List[np.ndarray], 
                     colors: List[str] = None,
                     labels: List[str] = None) -> None:
        """
        Helper function untuk visualisasi vektor.
        
        Args:
            vectors: List vektor yang akan divisualisasikan
            colors: List warna untuk setiap vektor
            labels: List label untuk setiap vektor
        """
        if colors is None:
            colors = ['red', 'blue', 'green', 'orange'][:len(vectors)]
            
        if labels is None:
            labels = [f"Vector {i+1}" for i in range(len(vectors))]
        
        plt.figure(figsize=(8, 8))
        plt.axhline(0, color='gray', linestyle='--', alpha=0.3)
        plt.axvline(0, color='gray', linestyle='--', alpha=0.3)
        
        for i, vector in enumerate(vectors):
            plt.quiver(0, 0, vector[0], vector[1], 
                      angles='xy', scale_units='xy', scale=1, 
                      color=colors[i], label=labels[i])
        
        # Tampilkan grid dengan unit circle
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'gray', alpha=0.3)
        
        plt.xlim(-4, 4)
        plt.ylim(-4, 4)
        plt.grid(alpha=0.3)
        plt.title("Visualisasi Transformasi Linear", fontsize=14)
        plt.xlabel("X", fontsize=12)
        plt.ylabel("Y", fontsize=12)
        plt.legend()
        plt.axis('equal')
        plt.show()
    
    def input_matrix(self, rows: int, cols: int, name: str = "Matriks") -> np.ndarray:
        """
        Meminta input matriks dari pengguna.
        
        Args:
            rows: Jumlah baris
            cols: Jumlah kolom
            name: Nama matriks untuk ditampilkan
            
        Returns:
            np.ndarray matriks yang diinput
        """
        print(f"\nInput {name} {rows}x{cols}:")
        matrix = np.zeros((rows, cols))
        
        for i in range(rows):
            while True:
                try:
                    row_input = input(f"Baris {i+1} (pisahkan dengan spasi): ")
                    values = list(map(float, row_input.split()))
                    
                    if len(values) != cols:
                        print(f"Error: Jumlah nilai harus {cols}")
                        continue
                        
                    matrix[i, :] = values
                    break
                except ValueError:
                    print("Error: Masukkan hanya angka")
        
        return matrix


def display_results(results: dict) -> None:
    """
    Menampilkan hasil operasi dengan format yang rapi.
    
    Args:
        results: Dictionary hasil operasi
    """
    print("\n" + "="*50)
    for key, value in results.items():
        print(f"\n{key}:")
        if isinstance(value, np.ndarray):
            print(value)
        elif isinstance(value, (int, float, str)):
            print(value)
        else:
            print(value)
    print("="*50 + "\n")


def clear_screen() -> None:
    """Membersihkan layar terminal."""
    print("\n" * 50)


def main() -> None:
    """Fungsi utama program."""
    # Inisialisasi tools
    la_tools = LinearAlgebraTools()
    
    while True:
        print("\n" + "="*50)
        print("    PROGRAM ALJABAR LINEAR INTERAKTIF    ")
        print("="*50)
        print("\nPilih operasi yang ingin dijalankan:")
        print("1: Operasi Matriks Dasar")
        print("2: Eigenvalue & Eigenvector")
        print("3: Singular Value Decomposition (SVD)")
        print("4: Gradient Descent & Regresi Linear")
        print("5: Transformasi Linear (Rotasi)")
        print("6: Input Matriks Kustom")
        print("0: Keluar")
        
        try:
            choice = input("\nMasukkan pilihan (0-6): ")
            
            if choice == '0':
                print("\nTerima kasih telah menggunakan program Aljabar Linear.")
                break
                
            elif choice == '1':
                clear_screen()
                print("\n--- OPERASI MATRIKS DASAR ---")
                results = la_tools.operasi_matriks()
                display_results(results)
                
            elif choice == '2':
                clear_screen()
                print("\n--- EIGENVALUE & EIGENVECTOR ---")
                results = la_tools.eigenvalue_eigenvector()
                display_results(results)
                
            elif choice == '3':
                clear_screen()
                print("\n--- SINGULAR VALUE DECOMPOSITION ---")
                results = la_tools.singular_value_decomposition()
                display_results(results)
                
            elif choice == '4':
                clear_screen()
                print("\n--- GRADIENT DESCENT & REGRESI LINEAR ---")
                results = la_tools.run_gradient_descent(visualize=True)
                display_results(results)
                
            elif choice == '5':
                clear_screen()
                print("\n--- TRANSFORMASI LINEAR (ROTASI) ---")
                angle_deg = float(input("Masukkan sudut rotasi (derajat, default=45): ") or "45")
                angle_rad = angle_deg * np.pi / 180
                results = la_tools.transformasi_linear(angle=angle_rad)
                display_results(results)
                
            elif choice == '6':
                clear_screen()
                print("\n--- INPUT MATRIKS KUSTOM ---")
                try:
                    rows = int(input("Jumlah baris: "))
                    cols = int(input("Jumlah kolom: "))
                    if rows <= 0 or cols <= 0:
                        print("Error: Dimensi matriks harus positif")
                        continue
                        
                    matrix = la_tools.input_matrix(rows, cols)
                    
                    print("\nPilih operasi untuk matriks kustom:")
                    print("1: Hitung determinan dan pangkat")
                    print("2: Eigenvalue & Eigenvector")
                    print("3: SVD")
                    
                    op_choice = input("Pilihan: ")
                    
                    if op_choice == '1':
                        results = {
                            "Matriks": matrix,
                            "Determinan": np.linalg.det(matrix) if rows == cols else "Bukan matriks persegi",
                            "Rank": np.linalg.matrix_rank(matrix),
                            "Trace": np.trace(matrix) if rows == cols else "Bukan matriks persegi"
                        }
                    elif op_choice == '2':
                        results = la_tools.eigenvalue_eigenvector(matrix)
                    elif op_choice == '3':
                        results = la_tools.singular_value_decomposition(matrix)
                    else:
                        print("Pilihan tidak valid.")
                        continue
                        
                    display_results(results)
                    
                except ValueError:
                    print("Error: Masukkan hanya angka untuk dimensi")
                
            else:
                print("Pilihan tidak valid. Silakan coba lagi.")
                
        except Exception as e:
            print(f"Terjadi kesalahan: {str(e)}")
            
        input("\nTekan Enter untuk melanjutkan...")
        clear_screen()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram dihentikan oleh pengguna.")
        sys.exit(0)