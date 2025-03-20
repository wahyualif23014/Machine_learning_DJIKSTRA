from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib

# Data contoh (resume, skripsi, jurnal)
documents = [
    "Saya seorang profesional dengan pengalaman 5 tahun di bidang software engineering.",  # Resume
    "Penelitian ini bertujuan untuk menganalisis pengaruh machine learning dalam data science.",  # Skripsi
    "Dalam jurnal ini, kami membahas perkembangan terbaru dalam deep learning dan AI."  # Jurnal Ilmiah
]
labels = ["Resume", "Skripsi", "Jurnal"]

# Konversi teks ke vektor numerik
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Latih model SVM
model = SVC(kernel="linear")
model.fit(X, labels)

# Simpan model
joblib.dump(model, "document_classifier.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
