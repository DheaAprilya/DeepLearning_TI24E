import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Memuat model yang sudah disimpan
model = tf.keras.models.load_model("model_cnn_buah.keras")

# Nama kelas sesuai urutan folder dataset
nama_kelas = ["apel", "pisang"]

# Membaca foto pisang baru
lokasi_gambar = "uji_gambar/pisang_baru.jpg"

gambar = tf.keras.utils.load_img(
    lokasi_gambar,
    target_size=(128, 128)
)

gambar_array = tf.keras.utils.img_to_array(gambar)
gambar_array = np.expand_dims(gambar_array, axis=0)

# Prediksi gambar
hasil = model.predict(gambar_array, verbose=0)[0]
indeks = np.argmax(hasil)

print("Hasil prediksi:", nama_kelas[indeks])
print("Skor prediksi: {:.2f}%".format(hasil[indeks] * 100))

plt.imshow(gambar)
plt.title("Hasil prediksi: " + nama_kelas[indeks])
plt.axis("off")
plt.show()
