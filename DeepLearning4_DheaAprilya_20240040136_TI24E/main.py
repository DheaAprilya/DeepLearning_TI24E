import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# Lokasi dataset
folder_dataset = "dataset_buah"
ukuran_gambar = (128, 128)
batch_size = 2

# Membaca dataset untuk pelatihan
data_latih = tf.keras.utils.image_dataset_from_directory(
    folder_dataset,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=ukuran_gambar,
    batch_size=batch_size
)

# Membaca dataset untuk validasi
data_validasi = tf.keras.utils.image_dataset_from_directory(
    folder_dataset,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=ukuran_gambar,
    batch_size=batch_size
)

# Menampilkan kategori buah
nama_kelas = data_latih.class_names
print("Kategori:", nama_kelas)

# Membuat model CNN
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(128, 128, 3)),
    tf.keras.layers.Rescaling(1.0 / 255),

    tf.keras.layers.Conv2D(
        16, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu"
    ),
    tf.keras.layers.MaxPooling2D(),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation="relu"),
    tf.keras.layers.Dense(
        len(nama_kelas),
        activation="softmax"
    )
])

# Mengatur proses pelatihan
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Menampilkan ringkasan model
model.summary()

# Melatih model CNN
history = model.fit(
    data_latih,
    validation_data=data_validasi,
    epochs=10
)

# Menampilkan grafik Accuracy
plt.figure(figsize=(8, 5))
plt.plot(
    history.history["accuracy"],
    label="Data Latih"
)
plt.plot(
    history.history["val_accuracy"],
    label="Data Validasi"
)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Grafik Accuracy")
plt.legend()
plt.tight_layout()
plt.show()

# Menampilkan grafik Loss
plt.figure(figsize=(8, 5))
plt.plot(
    history.history["loss"],
    label="Training Loss"
)
plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Grafik Loss")
plt.legend()
plt.tight_layout()
plt.show()

# Menyimpan model CNN
model.save("model_cnn_buah.keras")
print("Model berhasil disimpan.")

# Memuat model yang telah disimpan
model_tersimpan = tf.keras.models.load_model(
    "model_cnn_buah.keras"
)

# Lokasi gambar yang akan diuji
lokasi_gambar = "uji_gambar/apel_baru.jpg"

# Membaca gambar dan menyesuaikan ukurannya
gambar = tf.keras.utils.load_img(
    lokasi_gambar,
    target_size=ukuran_gambar
)

# Mengubah gambar menjadi array
gambar_array = tf.keras.utils.img_to_array(gambar)
gambar_array = np.expand_dims(gambar_array, axis=0)

# Melakukan prediksi
hasil_prediksi = model_tersimpan.predict(
    gambar_array,
    verbose=0
)

# Menentukan kategori dengan skor tertinggi
indeks_kelas = np.argmax(hasil_prediksi[0])
prediksi_buah = nama_kelas[indeks_kelas]
skor = hasil_prediksi[0][indeks_kelas] * 100

# Menampilkan hasil prediksi
print("Hasil prediksi:", prediksi_buah)
print("Skor prediksi: {:.2f}%".format(skor))

# Menampilkan gambar hasil prediksi
plt.figure(figsize=(5, 5))
plt.imshow(gambar)
plt.title("Hasil prediksi: " + prediksi_buah)
plt.axis("off")
plt.tight_layout()
plt.show()