import os
import cv2

# Directorio de imágenes de entrada y salida
img_dir = "static/images_medicas"
output_dir = "static/results_parte2"
os.makedirs(output_dir, exist_ok=True)

# Imágenes y kernels a procesar
img_names = [
    "Radiografia_simple_detorax.jpg",
    "Tomografia _de_abdomen_y_pelvis.jpg",
    "angiografia.jpg"
]

kernels = [7, 15, 37]

for img_name in img_names:
    path = os.path.join(img_dir, img_name)
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if gray is None:
        print(f"⚠️ No se pudo cargar: {path}")
        continue

    name_no_ext = os.path.splitext(img_name)[0].replace(" ", "_")

    for k in kernels:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))

        erosion = cv2.erode(gray, kernel)
        dilate = cv2.dilate(gray, kernel)
        top_hat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        black_hat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
        combinado = cv2.add(gray, cv2.subtract(top_hat, black_hat))

        cv2.imwrite(os.path.join(output_dir, f"{name_no_ext}_k{k}.png"), gray)
        cv2.imwrite(os.path.join(output_dir, f"{name_no_ext}_k{k}_erode.png"), erosion)
        cv2.imwrite(os.path.join(output_dir, f"{name_no_ext}_k{k}_dilate.png"), dilate)
        cv2.imwrite(os.path.join(output_dir, f"{name_no_ext}_k{k}_tophat.png"), top_hat)
        cv2.imwrite(os.path.join(output_dir, f"{name_no_ext}_k{k}_blackhat.png"), black_hat)
        cv2.imwrite(os.path.join(output_dir, f"{name_no_ext}_k{k}_comb.png"), combinado)

        print(f"✅ Generado para {img_name} con kernel {k}x{k}")
