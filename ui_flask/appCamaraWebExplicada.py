# Versión para webCam akaso, indice 1 en tiempo real
from __future__ import annotations   # permite usar anotaciones de tipo modernas en python
import sys, time, threading          # sys para argumentos, time para pausas, threading para hilos
from datetime import datetime        # manejar fecha y hora (guardar inicio, último guardado)
from pathlib import Path             # trabajar con rutas de archivos de forma más fácil

import cv2
import numpy as np
import torch                         # librería pytorch, usada por YOLO para la inferencia en GPU
from flask import Flask, render_template, Response, jsonify  # framework flask para la interfaz web
from ultralytics import YOLO         # carga el modelo YOLO
from sort import Sort                # algoritmo de seguimiento

# Rutas
ui_dir = Path(__file__).resolve().parent  # Guarda la carpeta donde esta appCamaraWebExplicada.py
root = ui_dir.parent  # Configura carpeta raiz del proyecto
data = root / "data"
models = root / "model"
scripts = root / "scripts"
model_path = models / "best.pt"
mask_path = data / "mask.jpg"

linea = (485, 450, 780, 450)  # coordenadas linea para el conteo
banday = 18  # tolerancia en pixeles para el cruce de linea

conf_th = 0.45  # Umbral de confianza
imgsz = 576  # Tamaño medio efectividad detección/ Velocidad
infer_every = 1  # Inferencia en todos los frames para no perder conteos. La webcam está dando 10 fps

# Stream web
stream_res = (960, 540)  # configuración tamaño de imagen jpeg enviada al navegador
jpeg_qlty = 70  # valor de la compresión de imagen enviada al navegador

# Camara web
cam_index = 1
cam_fps = 30  # asegura que la cámara trabaje máximo en 30 fps, aunque está entregando 10 fps
cam_width = 1280
cam_height = 720
use_directshow = True  # en Windows usar CAP_DSHOW

device = "cuda" if torch.cuda.is_available() else "cpu"  # Configuré el env para usar cuda.
torch.backends.cudnn.benchmark = True


# Se programó el constructor con un enfoque orientado a objetos, creando funciones para los hilos y procesos.
# --- Worker (lectura directa de webcam) ---
class CounterWorker:  # Constructor, inicializa atributos del objeto
    def __init__(self):
        self.thread: threading.Thread | None = None  # hilo de ejecución
        self.running = False  # Estado de ejecución
        self.lock = threading.Lock()  # control de acceso seguro
        self.latest_jpeg: bytes | None = None  # última imagen lista
        self.total = 0  # conteo acumulado
        self.start_time: datetime | None = None  # hora inicio
        self.last_saved: datetime | None = None  # hora guardado en base de datos
        self._stop_request = threading.Event()  # señal para detener el hilo

        # Cargar YOLO una sola vez
        self.model = YOLO(str(model_path))
        self.model.fuse()  # Combina capas Conv2d + BatchNorm2d en una sola, acelerando la inferencia
        self.model.to(device)
        if device == "cuda":
            self.model.model.half()  # FP16 en GPU para ganar velocidad, divide FP32 que es el formato original de los pesos

    def _punto_cruza_linea(self, cx, cy, linea, banda_y=10):
        x1, y1, x2, y2 = linea
        if x1 == x2:  # vertical
            return (min(y1, y2) <= cy <= max(y1, y2)) and (abs(cx - x1) <= banda_y)
        else:  # horizontal
            return (min(x1, x2) <= cx <= max(x1, x2)) and (abs(cy - y1) <= banda_y)

    def start(self):
        if self.running:
            return
        self._stop_request.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def stop(self):
        if not self.running:
            return
        self._stop_request.set()
        if self.thread:
            self.thread.join(timeout=5)
        # Autoguardar al detener
        try:
            import subprocess
            script_db = scripts / "guardar_en_db.py"  # se importa el script para manejar MySql
            if script_db.exists():
                subprocess.run([sys.executable, str(script_db), str(self.total)], check=False)
                self.last_saved = datetime.now()
                print(f"[SAVE] Guardado en DB total={self.total} @ {self.last_saved}")
            else:
                print(f"[WARN] No se encontró {script_db}; no se guardó en DB.")
        except Exception as e:
            print(f"[WARN] No se pudo autoguardar en DB: {e}")

    def run(self):  # metodo que ejecuta el conteo en un hilo
        self.running = True  # Marca que el worker está en ejecución
        self.start_time = datetime.now()  # Guarda hora de inicio
        self.total = 0  # Reinicia conteo total en cero
        counted_ids = set()  # Conjunto evita conteo del mismo objeto 2 veces

        print("[WORKER] Webcam en tiempo real")
        print(f"[WORKER] model_path: {model_path.resolve()}")
        print(f"[WORKER] mask_path : {mask_path.resolve()} (si no existe, se ignora)")

        # Abrir cámara, se dan las configuraciones de cv2
        backend = cv2.CAP_DSHOW if use_directshow else 0
        cap = cv2.VideoCapture(cam_index, backend)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,
                1)  # buffer es la cola donde se guardan los frames del video antes de ser leidos
        cap.set(cv2.CAP_PROP_FPS, cam_fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_height)

        if not cap.isOpened():
            print("[ERROR] No se pudo abrir la cámara.")
            self.running = False
            return

        # Cargar máscara
        mask = None
        if mask_path.exists():
            mask = cv2.imread(str(mask_path))
            if mask is None:
                print(f"[WARN] No se pudo leer la máscara: {mask_path}")
                mask = None

        # Leer un primer frame para dimensionar máscara
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[ERROR] No se pudo leer el primer frame de la cámara.")
            cap.release()
            self.running = False
            return

        if mask is not None and mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        tracker = Sort(max_age=25, min_hits=3,
                       iou_threshold=0.3)  # minhits, objeto debe ser detectado en 3 frames para asignar id. iou es las cajas entre frames
        x1, y1, x2, y2 = linea

        try:
            while not self._stop_request.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    time.sleep(0.005)
                    continue

                roi = cv2.bitwise_and(frame, mask) if mask is not None else frame

                # Se envían los frames al modelo
                results = self.model(  # se asignan las detecciones a results
                    roi, conf=conf_th, imgsz=imgsz, verbose=False, half=(device == "cuda")
                )[0]

                detecciones = []
                for box in results.boxes:  # extrae las coordenadas de las cajas con la confianza y se las pasa a sort
                    conf = float(box.conf[0])
                    if conf > conf_th:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        detecciones.append([bx1, by1, bx2, by2, conf])

                det_np = np.array(detecciones) if detecciones else np.empty((0, 5))
                outputs = tracker.update(det_np)

                # Dibujo y conteo. roi (region of interest) en este caso la parte de la mascara que no es negra
                cv2.line(roi, (x1, y1), (x2, y2), (0, 0, 255), 2)  # dibuja la linea para el conteo
                for ox1, oy1, ox2, oy2, obj_id in outputs:  # se calcula el centro de la caja
                    ox1, oy1, ox2, oy2 = map(int, [ox1, oy1, ox2, oy2])
                    cx, cy = (ox1 + ox2) // 2, (oy1 + oy2) // 2

                    if self._punto_cruza_linea(cx, cy, linea,
                                               banday):  # verifica que el centro del objeto cruza la linea
                        if obj_id not in counted_ids:  # revisa que el objeto no se haya contado
                            counted_ids.add(int(obj_id))  # se marca como contado
                            self.total += 1  # se aumenta el contador
                            cv2.line(roi, (x1, y1), (x2, y2), (0, 200, 0),
                                     4)  # linea verde momentanea para marcar cruce

                    cv2.rectangle(roi, (ox1, oy1), (ox2, oy2), (255, 0, 0),
                                  2)  # rectangulo azul alrededor de galleta detectada
                    cv2.putText(roi, f'ID {int(obj_id)}', (ox1, max(20, oy1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255),
                                2)  # número id en blanco encima de la caja

                cv2.putText(roi, f'Total: {self.total}', (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (50, 50, 255), 3)  # conteo en rojo en esquina

                # Stream liviano para el navegador
                disp = roi  # asigna la imagen procesada a disp con cajas, linea y contador
                if stream_res:
                    disp = cv2.resize(disp, stream_res)
                ok_jpg, jpg = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY,
                                                          jpeg_qlty])  # convierte el array de numpy del frame en jpg para enviarlo al navegador
                if ok_jpg:  # envia el jpeg al navegador
                    with self.lock:
                        self.latest_jpeg = jpg.tobytes()



        except Exception as e:
            print(f"[ERROR] Worker: {e}")

        finally:
            cap.release()  # cierra conexión de la cámara
            self.running = False


# --- Flask app ---
app = Flask(__name__, template_folder=str(ui_dir / "templates"),
            static_folder=str(ui_dir / "static"))  # se crea la app flask y ubicación de templates html y estáticos
worker = CounterWorker()  # se crea un objeto worker de la clase CounterWorker para abrir la cámara y hacer el conteo en segundo plano


def fmt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "-"  # Se establece el formato de fecha como string


@app.route("/")
def index():
    return render_template("index.html") if (ui_dir / "templates" / "index.html").exists() else (
        "<html><body style='background:#111;color:#eee;font-family:system-ui'>"
        "<h2>Conteo en tiempo real (Webcam Akaso)</h2>"
        "<div><img src='/stream' style='max-width:100%;border:1px solid #444'/></div>"
        "<div style='margin-top:12px'>"
        "<button onclick='fetch(\"/start\",{method:\"POST\"}).then(()=>location.reload())'>Iniciar</button>"
        "<button onclick='fetch(\"/stop\",{method:\"POST\"}).then(()=>location.reload())' style='margin-left:8px'>Detener</button>"
        "</div>"
        "<script>setInterval(()=>fetch('/metrics').then(r=>r.json()).then(m=>{document.title=`Total ${m.total}`;}),1000)</script>"
        "</body></html>"
    )


@app.route("/start", methods=[
    "POST"])  # Define la ruta principal de la app (URL raíz: http://127.0.0.1:5000/) post es para enviar información al navegador
def start():
    worker.start()  # inicia el hilo worker (cámara y conteo)
    return jsonify(ok=True, running=worker.running, start_time=fmt(worker.start_time),
                   total=worker.total)  # se envia la informacion al navegador en json


@app.route("/stop", methods=["POST"])
def stop():
    worker.stop()
    return jsonify(ok=True, running=worker.running, last_saved=fmt(worker.last_saved), total=worker.total)


@app.route("/metrics")
def metrics():
    return jsonify(
        running=worker.running,  # Si el conteo está activo o no
        total=worker.total,  # Cantidad de galletas contadas
        start_time=fmt(worker.start_time),  # Hora de inicio del conteo
        last_saved=fmt(worker.last_saved)  # Última vez que se guardó en la BD
    )


@app.route("/stream")
def stream():
    def gen():  # generador que envía frame uno tras otro
        boundary = b"--frame"  # separador
        while True:
            with worker.lock:
                frame = worker.latest_jpeg  # último frame generado
            if frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                ok_jpg, jpg = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
                frame = jpg.tobytes() if ok_jpg else b""  # envia frame como un hilo jpg
            yield (
                        boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")  # entrega frame por frame al navegador
            time.sleep(0.03)  # configura 33 fps en el stream

    return Response(gen(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")  # le dice a flask que devuelva el stream de video al navegador


if __name__ == "__main__":  # verifica que se está ejecutando este script como principal

    print(f"[CHECK] Usando cámara índice: {cam_index}")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
