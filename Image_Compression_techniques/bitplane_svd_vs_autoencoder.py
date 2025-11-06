
# =============================================================================
# COMPARATIVE IMAGE COMPRESSION: BitPlane + SVD vs BitPlane + Autoencoder
# Author: Vishal (powered by ChatGPT)
# =============================================================================

!pip install -q opencv-python-headless scikit-image matplotlib seaborn pillow tensorflow

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import mean_squared_error
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
from google.colab import files
import io
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings
warnings.filterwarnings('ignore')

# ========== UTILITIES ==========

def extract_bit_planes(image):
    return [(image >> i & 1).astype(np.uint8) for i in range(8)]

def reconstruct_from_planes(planes, indices, shape):
    img = np.zeros(shape, dtype=np.uint8)
    for plane, idx in zip(planes, indices):
        img += (plane * (2 ** idx)).astype(np.uint8)
    return img

def calculate_entropy(plane):
    hist = np.histogram(plane, bins=2, range=(0, 2))[0]
    prob = hist / hist.sum()
    prob = prob[prob > 0]
    return -np.sum(prob * np.log2(prob)) if len(prob) > 1 else 0

def select_important_planes(bit_planes, num=4):
    entropies = [calculate_entropy(plane) for plane in bit_planes]
    indices = np.argsort(entropies)[-num:]
    selected = [bit_planes[i] for i in indices]
    return selected, indices

# ========== SVD COMPRESSION ==========

def svd_compress(image, k=20):
    U, S, VT = np.linalg.svd(image, full_matrices=False)
    S_k = np.diag(S[:k])
    compressed = np.dot(U[:, :k], np.dot(S_k, VT[:k, :]))
    return np.clip(compressed, 0, 255).astype(np.uint8), k * (U.shape[0] + VT.shape[0] + 1)

# ========== AUTOENCODER COMPRESSION ==========

def build_autoencoder(input_shape, compression_ratio):
    tf.keras.backend.clear_session()
    input_img = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    shape_before_flatten = x.shape[1:]
    x = Flatten()(x)
    latent_dim = max(32, int(np.prod(input_shape) * compression_ratio))
    encoded = Dense(latent_dim, activation='relu')(x)

    encoded_input = Input(shape=(latent_dim,))
    x = Dense(np.prod(shape_before_flatten), activation='relu')(encoded_input)
    x = Reshape(shape_before_flatten)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    encoder = Model(input_img, encoded)
    decoder = Model(encoded_input, decoded)
    autoencoder = Model(input_img, decoder(encoder(input_img)))
    autoencoder.compile(optimizer=Adam(1e-3), loss='binary_crossentropy')

    return autoencoder, encoder, decoder

def autoencoder_compress(image, num_planes=4, size=(128, 128), compression_ratio=0.05):
    planes = extract_bit_planes(cv2.resize(image, size))
    selected_planes, indices = select_important_planes(planes, num_planes)
    data = np.array(selected_planes).reshape(-1, size[0], size[1], 1).astype('float32')

    autoencoder, encoder, decoder = build_autoencoder((size[0], size[1], 1), compression_ratio)
    autoencoder.fit(data, data, epochs=20, batch_size=4, verbose=0,
                    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)])

    encoded = [encoder.predict(plane[np.newaxis, ...], verbose=0) for plane in data]
    decoded = [decoder.predict(e, verbose=0).reshape(size) > 0.5 for e in encoded]
    recon = reconstruct_from_planes(decoded, indices, size)
    recon = cv2.resize(recon, (image.shape[1], image.shape[0]))
    return recon.astype(np.uint8), sum(e.nbytes for e in encoded)

# ========== METRICS & VISUALIZATION ==========

def evaluate(original, recon, compressed_size):
    return {
        "PSNR": psnr(original, recon, data_range=255),
        "SSIM": ssim(original, recon, data_range=255),
        "MSE": mean_squared_error(original.flatten(), recon.flatten()),
        "CR": original.nbytes / max(compressed_size, 1)
    }

def visualize_comparison(original, svd_img, ae_img, svd_metrics, ae_metrics):
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))
    axs[0, 0].imshow(original, cmap="gray"); axs[0, 0].set_title("Original"); axs[0, 0].axis("off")
    axs[0, 1].imshow(svd_img, cmap="gray"); axs[0, 1].set_title("BitPlane+SVD"); axs[0, 1].axis("off")
    axs[0, 2].imshow(ae_img, cmap="gray"); axs[0, 2].set_title("BitPlane+Autoencoder"); axs[0, 2].axis("off")
    axs[1, 0].bar(["PSNR", "SSIM"], [svd_metrics["PSNR"], svd_metrics["SSIM"]])
    axs[1, 0].set_title("SVD Metrics")
    axs[1, 1].bar(["PSNR", "SSIM"], [ae_metrics["PSNR"], ae_metrics["SSIM"]])
    axs[1, 1].set_title("Autoencoder Metrics")

    table_data = [
        ["Metric", "SVD", "Autoencoder"],
        ["PSNR", f"{svd_metrics['PSNR']:.2f}", f"{ae_metrics['PSNR']:.2f}"],
        ["SSIM", f"{svd_metrics['SSIM']:.4f}", f"{ae_metrics['SSIM']:.4f}"],
        ["MSE", f"{svd_metrics['MSE']:.2f}", f"{ae_metrics['MSE']:.2f}"],
        ["CR", f"{svd_metrics['CR']:.2f}:1", f"{ae_metrics['CR']:.2f}:1"]
    ]
    axs[1, 2].axis("off")
    axs[1, 2].table(cellText=table_data, loc='center', cellLoc='center')
    plt.tight_layout()
    plt.show()

# ========== MAIN RUN ==========

print("📁 Upload image to compare compression techniques")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
image = Image.open(io.BytesIO(uploaded[filename])).convert('L')
image = np.array(image)

print("🔄 Running BitPlane+SVD...")
svd_img, svd_size = svd_compress(image)
svd_metrics = evaluate(image, svd_img, svd_size)

print("🔄 Running BitPlane+Autoencoder...")
ae_img, ae_size = autoencoder_compress(image)
ae_metrics = evaluate(image, ae_img, ae_size)

visualize_comparison(image, svd_img, ae_img, svd_metrics, ae_metrics)
print("✅ Comparison complete!")
