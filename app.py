from io import BytesIO
import cv2
import numpy as np
import requests
import time
from flask import Flask, render_template, Response, request, send_file
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
app = Flask(__name__)

# Dirección IP del ESP32
_URL = 'http://192.168.1.177'
_PORT = '81'
_ST = '/stream'
SEP = ':'

flujo_flags = {
}
stream_url = f"{_URL}:{_PORT}{_ST}"

# Substractor de fondo global
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

noise_params = {
    "media": 100,
    "desviacion": 25,
    "varianza": 0.01
}

bitwise_mode = "AND"
selected_filter = "gray"

def video_capture():
    res = requests.get(stream_url, stream=True)
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=50, varThreshold=40)
    prev_frame_time = time.time()

    for chunk in res.iter_content(chunk_size=100000):
        if len(chunk) > 100:
            try:
                img_data = BytesIO(chunk)
                cv_img = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)

                blur = cv2.GaussianBlur(cv_img, (5, 5), 0)
                mask = bg_subtractor.apply(blur)
                mask = cv2.erode(mask, None, iterations=2)
                mask = cv2.dilate(mask, None, iterations=2)

                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for cnt in contours:
                    if cv2.contourArea(cnt) > 500:
                        x, y, w, h = cv2.boundingRect(cnt)
                        cv2.rectangle(cv_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

                curr_time = time.time()
                fps = 1.0 / (curr_time - prev_frame_time)
                prev_frame_time = curr_time
                cv2.putText(cv_img, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                _, jpeg = cv2.imencode('.jpg', cv_img)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            except Exception as e:
                print("Error:", e)
                continue
@app.route("/set_bitwise_mode", methods=["POST"])
def set_bitwise_mode():
    global bitwise_mode
    data = request.get_json()
    modo = data.get("mode", "AND").upper()
    if modo in ["AND", "OR", "XOR"]:
        bitwise_mode = modo
        print(f"🟢 Modo Bitwise actualizado a: {bitwise_mode}")
        return {"status": "ok", "mode": bitwise_mode}, 200
    return {"status": "invalid mode"}, 400


@app.route("/parte1a_completa")
def parte1a_completa():
    return render_template("index.html", stream_url="/video_stream_parte1a", current_mode="parte1a_completa")


def generate_parte1a_stream():
    try:
        res = requests.get(stream_url, stream=True)
        bg_subtractor = cv2.createBackgroundSubtractorMOG2()
        prev_time = time.time()
        global bitwise_mode

        for chunk in res.iter_content(chunk_size=100000):
            if len(chunk) <= 100:
                continue

            img_data = BytesIO(chunk)
            frame = cv2.imdecode(np.frombuffer(chunk, np.uint8), 1)
            if frame is None or frame.shape[0] == 0:
                print("⚠️ Frame inválido o vacío, omitido.")
                continue

            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time)
            prev_time = curr_time

            size = (480, 360)

            def preparar_img(img, etiqueta, fps_text=None):
                try:
                    if img.ndim == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img = cv2.resize(img, (480, 360))
                    cv2.putText(img, etiqueta, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    if fps_text:
                        cv2.putText(img, fps_text, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                    # Agregar borde vertical derecho (1px blanco)
                    img = cv2.copyMakeBorder(img, 0, 0, 0, 2, cv2.BORDER_CONSTANT, value=(255, 255, 255))
                    return img
                except Exception as e:
                    print(f"[ERROR preparar_img] {etiqueta}: {e}")
                    return None


            paneles = []
            fps_text = f"FPS: {fps:.2f}"
            paneles.append(preparar_img(frame.copy(), "ORIGINAL", fps_text))

            blur = cv2.GaussianBlur(frame, (5, 5), 0)
            mask = bg_subtractor.apply(blur)
            paneles.append(preparar_img(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), "SUBSTRACCIÓN (MOG2)"))

            if bitwise_mode == "AND":
                movimiento = cv2.bitwise_and(frame, frame, mask=mask)
            elif bitwise_mode == "OR":
                movimiento = cv2.bitwise_or(frame, np.zeros_like(frame), mask=mask)
            elif bitwise_mode == "XOR":
                movimiento = cv2.bitwise_xor(frame, np.full_like(frame, 255), mask=mask)
            else:
                movimiento = np.zeros_like(frame)

            paneles.append(preparar_img(movimiento, f"MOVIMIENTO ({bitwise_mode})"))

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            equalized = cv2.equalizeHist(gray)
            clahe_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
            gamma = 1.5
            lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)]).astype("uint8")
            gamma_img = cv2.LUT(gray, lut)

            if selected_filter == "gray":
                paneles.append(preparar_img(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), "GRIS"))
            elif selected_filter == "equalized":
                paneles.append(preparar_img(cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR), "ECUALIZADO"))
            elif selected_filter == "clahe":
                paneles.append(preparar_img(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR), "CLAHE"))
            elif selected_filter == "gamma":
                paneles.append(preparar_img(cv2.cvtColor(gamma_img, cv2.COLOR_GRAY2BGR), "GAMMA"))

            paneles = [p for p in paneles if p is not None]
            if len(paneles) < 4:
                print("⚠️ Frame omitido: no hay suficientes paneles válidos")
                continue

            try:
                fila1 = cv2.hconcat(paneles[:4])

                if len(paneles) > 4:
                    fila2 = cv2.hconcat(paneles[4:])

                    # Ajustar diferencias de ancho
                    if fila1.shape[1] != fila2.shape[1]:
                        diff = abs(fila1.shape[1] - fila2.shape[1])
                        if fila1.shape[1] > fila2.shape[1]:
                            pad = np.zeros((fila2.shape[0], diff, 3), dtype=np.uint8)
                            fila2 = np.hstack((fila2, pad))
                        else:
                            pad = np.zeros((fila1.shape[0], diff, 3), dtype=np.uint8)
                            fila1 = np.hstack((fila1, pad))

                    resultado = cv2.vconcat([fila1, fila2])
                else:
                    resultado = fila1

                _, jpeg = cv2.imencode('.jpg', resultado)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

            except Exception as e:
                print("❌ Error al concatenar filas finales:", e)
                continue

    except Exception as e:
        print("💥 Error en parte1a_completa:", e)



@app.route("/set_filter_mode", methods=["POST"])
def set_filter_mode():
    global selected_filter
    data = request.get_json()
    modo = data.get("filter", "gray")
    if modo in ["gray", "equalized", "clahe", "gamma"]:
        selected_filter = modo
        print(f"🟢 Filtro de iluminación actualizado a: {selected_filter}")
        return {"status": "ok", "filter": selected_filter}, 200
    return {"status": "invalid filter"}, 400

def generate_stream_flujo():
    try:
        res = requests.get(stream_url, stream=True)
        for chunk in res.iter_content(chunk_size=100000):
            if len(chunk) <= 100:
                continue

            img_data = BytesIO(chunk)
            frame = cv2.imdecode(np.frombuffer(img_data.read(), np.uint8), 1)
            if frame is None:
                continue

            frame = cv2.resize(frame, (480, 360))
            original = frame.copy()

            # Aplicar ruido
            media = noise_params["media"]
            std = noise_params["desviacion"]
            varianza = noise_params["varianza"]

            gauss_noise = np.random.normal(media, std, frame.shape).astype(np.float32)
            speckle_noise = np.random.normal(0, np.sqrt(varianza), frame.shape)
            noisy = frame.astype(np.float32) + gauss_noise + (frame.astype(np.float32) * speckle_noise)
            noisy = np.clip(noisy, 0, 255).astype(np.uint8)

            resultado = noisy.copy()

            # Mostrar qué flags están activos
            print("🛠️ FLAGS:", flujo_flags)

            # Aplicar suavizado
            suavizado = flujo_flags.get("suavizado")
            if suavizado == "mediana_3":
                resultado = cv2.medianBlur(resultado, 3)
            elif suavizado == "mediana_5":
                resultado = cv2.medianBlur(resultado, 5)
            elif suavizado == "mediana_7":
                resultado = cv2.medianBlur(resultado, 7)
            elif suavizado == "blur_3":
                resultado = cv2.blur(resultado, (3, 3))
            elif suavizado == "blur_5":
                resultado = cv2.blur(resultado, (5, 5))
            elif suavizado == "blur_7":
                resultado = cv2.blur(resultado, (7, 7))
            elif suavizado == "gauss_3":
                resultado = cv2.GaussianBlur(resultado, (3, 3), 0)
            elif suavizado == "gauss_5":
                resultado = cv2.GaussianBlur(resultado, (5, 5), 0)
            elif suavizado == "gauss_7":
                resultado = cv2.GaussianBlur(resultado, (7, 7), 0)

            # Aplicar detección de bordes
            bordes = flujo_flags.get("bordes")
            if bordes == "sobel":
                print("🧪 Aplicando filtro Sobel...")
                gray = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
                sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                abs_x = cv2.convertScaleAbs(sobel_x)
                abs_y = cv2.convertScaleAbs(sobel_y)
                sobel = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
                resultado = cv2.cvtColor(sobel, cv2.COLOR_GRAY2BGR)
            elif bordes == "canny":
                print("🧪 Aplicando filtro Canny...")
                gray = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 100, 200)
                resultado = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

            # Comparación usando máscara
            mask = cv2.cvtColor(resultado, cv2.COLOR_BGR2GRAY)
            _, bin_mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
            comparacion = cv2.bitwise_and(resultado, resultado, mask=bin_mask)

            # Preparar paneles visuales
            def prep(img, label):
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                img = cv2.resize(img, (480, 360))
                cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                return img

            panels = [
                prep(original, "Original"),
                prep(gauss_noise.astype(np.uint8), "Ruido Gaussiano"),
                prep((frame.astype(np.float32) * speckle_noise).astype(np.uint8), "Speckle"),
                prep(resultado, "Resultado Final"),
                prep(comparacion, "Resultado copyTo()")
            ]

            try:
                fila1 = cv2.hconcat(panels[:3])
                fila2 = cv2.hconcat(panels[3:])
                final = cv2.vconcat([fila1, fila2])
                _, jpeg = cv2.imencode('.jpg', final)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
            except Exception as e:
                print("❌ Concatenación fallida:", e)

    except Exception as e:
        print("💥 Error en flujo:", e)


@app.route("/set_flujo_flags", methods=["POST"])
def set_flujo_flags():
    data = request.get_json()
    flujo_flags["suavizado"] = data.get("suavizado", None)
    flujo_flags["bordes"] = data.get("bordes", None)
    return {"status": "ok"}

@app.route("/video_stream_flujo")
def video_stream_flujo():
    return Response(generate_stream_flujo(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/")
def index():
    kernels = [7, 15, 37]
    img_names = [
        "Radiografia_simple_detorax.jpg",
        "Tomografia _de_abdomen_y_pelvis.jpg",
        "angiografia.jpg"
    ]

    image_sets = []

    for kernel_size in kernels:
        for img_name in img_names:
            name_no_ext = os.path.splitext(img_name)[0].replace(" ", "_")
            set_images = []
            labels = ["Original", "Erosion", "Dilatación", "Top Hat", "Black Hat", "TopHat - BlackHat + Original"]
            suffixes = ["", "_erode", "_dilate", "_tophat", "_blackhat", "_comb"]
            for label, suffix in zip(labels, suffixes):
                filename = f"results_parte2/{name_no_ext}_k{kernel_size}{suffix}.png"
                set_images.append((filename, label))
            image_sets.append({
                "title": f"{name_no_ext.replace('_', ' ')} - Kernel {kernel_size}x{kernel_size}",
                "images": set_images,
                "kernel": kernel_size
            })

    return render_template("index.html", image_sets=image_sets)

@app.route("/parte2_morfologia")
def parte2_morfologia():
    return index()

@app.route("/todo_en_flujo")
def todo_en_flujo():
    return render_template("index.html", stream_url="/video_stream_flujo", current_mode="todo_en_flujo")


@app.route("/video_stream")
def video_stream():
    return Response(video_capture(), mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/set_noise_params", methods=["POST"])
def set_noise_params():
    data = request.get_json()
    try:
        noise_params["media"] = float(data.get("media", 100))
        noise_params["desviacion"] = float(data.get("desviacion", 25))
        noise_params["varianza"] = float(data.get("varianza", 0.01))
        return {"status": "ok"}, 200
    except Exception as e:
        print("Error en set_noise_params:", e)
        return {"status": "error"}, 400

@app.route("/video_stream_parte1a")
def video_stream_parte1a():
    return Response(generate_parte1a_stream(), mimetype="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    app.run(debug=False)  