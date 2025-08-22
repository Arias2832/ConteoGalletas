# appMejorInf2.py
from __future__ import annotations
import sys, time, threading, os
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

# --- Rutas del proyecto ---
UI_DIR   = Path(__file__).resolve().parent
ROOT     = UI_DIR.parent
DATA     = ROOT / "data"
MODELS   = ROOT / "model"
SCRIPTS  = ROOT / "scripts"

# para importar sort.py desde /scripts
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
from sort import Sort

# --- Configuración orientada a VIDEO (no cámara) ---
VIDEO_FILENAME = "100galletas.MOV"
VIDEO_PATH = DATA / VIDEO_FILENAME
MASK_PATH  = DATA / "mask.jpg"
MODEL_PATH = MODELS / "best.pt"

LINEA      = (770, 450, 1170, 450)  # ajusta a tu escena
BANDA_Y    = 22                     # tolerancia vertical/horizontal
CONF_TH    = 0.30
IMGSZ      = 576
STREAM_RES = (960, 540)
JPEG_QLTY  = 70

# Ritmo objetivo (evita "fast forward" si el origen es 60 fps)
TARGET_FPS = 24                     # pon None para usar FPS de origen

# Auto-iniciar el procesamiento al levantar el servidor
AUTO_START = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

def rate_limiter_init(cap, target_fps):
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fps = float(target_fps) if target_fps else float(src_fps)
    return {"period": 1.0 / max(1.0, fps), "t_next": time.perf_counter()}

def rate_limiter_sleep(rl):
    rl["t_next"] += rl["period"]
    to_sleep = rl["t_next"] - time.perf_counter()
    if to_sleep > 0:
        time.sleep(to_sleep)
    else:
        rl["t_next"] = time.perf_counter()

def open_video_with_fallbacks(path: Path):
    """Intenta abrir el video probando distintos backends de OpenCV."""
    # 1) CAP_ANY (deja que OpenCV elija)
    cap = cv2.VideoCapture(str(path), cv2.CAP_ANY)
    if cap.isOpened():
        return cap, "CAP_ANY"
    if cap:
        cap.release()
    # 2) Sin especificar backend
    cap = cv2.VideoCapture(str(path))
    if cap.isOpened():
        return cap, "DEFAULT"
    if cap:
        cap.release()
    # 3) CAP_FFMPEG (si tu build lo soporta)
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    if cap.isOpened():
        return cap, "CAP_FFMPEG"
    if cap:
        cap.release()
    return None, None

# --- Worker optimizado (sin hilo de captura para VIDEO) ---
class CounterWorker:
    def __init__(self):
        self.thread: threading.Thread | None = None
        self.running = False
        self.lock = threading.Lock()
        self.latest_jpeg: bytes | None = None
        self.total = 0
        self.start_time: datetime | None = None
        self.last_saved: datetime | None = None
        self._stop_request = threading.Event()

        # Cargar YOLO una sola vez
        self.model = YOLO(str(MODEL_PATH))
        self.model.fuse()
        self.model.to(DEVICE)
        if DEVICE == "cuda":
            self.model.model.half()  # FP16 en GPU

    def _punto_cruza_linea(self, cx, cy, linea, banda_y=10):
        x1, y1, x2, y2 = linea
        if x1 == x2:  # línea vertical
            return (min(y1, y2) <= cy <= max(y1, y2)) and (abs(cx - x1) <= banda_y)
        else:         # línea horizontal
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
            script_db = SCRIPTS / "guardar_en_db.py"
            if script_db.exists():
                subprocess.run([sys.executable, str(script_db), str(self.total)], check=False)
                self.last_saved = datetime.now()
                print(f"[SAVE] Guardado en DB total={self.total} @ {self.last_saved}")
            else:
                print(f"[WARN] No se encontró {script_db}; no se guardó en DB.")
        except Exception as e:
            print(f"[WARN] No se pudo autoguardar en DB: {e}")

    def run(self):
        self.running = True
        self.start_time = datetime.now()
        self.total = 0
        counted_ids = set()

        print("[WORKER] Modo: Video secuencial")
        print(f"[WORKER] VIDEO_PATH: {VIDEO_PATH.resolve()}")
        print(f"[WORKER] MODEL_PATH: {MODEL_PATH.resolve()}")
        print(f"[WORKER] MASK_PATH : {MASK_PATH.resolve()} (si no existe, se ignora)")

        if not VIDEO_PATH.exists():
            print(f"[ERROR] No existe el archivo de video: {VIDEO_PATH.resolve()}")
            self.running = False
            return

        cap, backend = open_video_with_fallbacks(VIDEO_PATH)
        if cap is None or not cap.isOpened():
            print("[ERROR] No se pudo abrir el video con ningún backend (CAP_ANY/DEFAULT/CAP_FFMPEG).")
            print("[HINT] Verifica códecs del .MOV o convierte a MP4 H.264.")
            self.running = False
            return
        else:
            print(f"[WORKER] Video abierto con backend: {backend}")

        # Máscara si existe
        mask = cv2.imread(str(MASK_PATH)) if MASK_PATH.exists() else None

        ok, frame = cap.read()
        if not ok or frame is None:
            print("[ERROR] No se pudo leer el primer frame.")
            cap.release()
            self.running = False
            return

        if mask is not None and mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

        tracker = Sort(max_age=25, min_hits=3, iou_threshold=0.3)
        x1, y1, x2, y2 = LINEA

        rl = rate_limiter_init(cap, TARGET_FPS)

        try:
            while not self._stop_request.is_set():
                roi = cv2.bitwise_and(frame, mask) if mask is not None else frame

                # Ultralytics: usar predict() y dejar el half ya aplicado al modelo
                results = self.model.predict(roi, conf=CONF_TH, imgsz=IMGSZ, verbose=False)[0]

                detecciones = []
                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf > CONF_TH:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0])
                        detecciones.append([bx1, by1, bx2, by2, conf])

                det_np = np.array(detecciones) if detecciones else np.empty((0, 5))
                outputs = tracker.update(det_np)

                # Línea y dibujado
                cv2.line(roi, (x1, y1), (x2, y2), (0, 0, 255), 2)
                for ox1, oy1, ox2, oy2, obj_id in outputs:
                    ox1, oy1, ox2, oy2 = map(int, [ox1, oy1, ox2, oy2])
                    cx, cy = (ox1 + ox2) // 2, (oy1 + oy2) // 2

                    if self._punto_cruza_linea(cx, cy, LINEA, BANDA_Y):
                        if int(obj_id) not in counted_ids:
                            counted_ids.add(int(obj_id))
                            self.total += 1
                            cv2.line(roi, (x1, y1), (x2, y2), (0, 200, 0), 4)

                    cv2.rectangle(roi, (ox1, oy1), (ox2, oy2), (255, 0, 0), 2)
                    cv2.putText(roi, f'ID {int(obj_id)}', (ox1, max(20, oy1 - 6)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                cv2.putText(roi, f'Total: {self.total}', (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.6, (50, 50, 255), 3)

                # Stream más liviano
                disp = cv2.resize(roi, STREAM_RES)
                ok_jpg, jpg = cv2.imencode(".jpg", disp, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QLTY])
                if ok_jpg:
                    with self.lock:
                        self.latest_jpeg = jpg.tobytes()

                # Leer siguiente frame y aplicar rate limiting
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[WORKER] Fin del video.")
                    break
                rate_limiter_sleep(rl)

        except Exception as e:
            print(f"[ERROR] Worker: {e}")

        finally:
            cap.release()
            self.running = False

# --- Flask app ---
app = Flask(__name__, template_folder=str(UI_DIR / "templates"), static_folder=str(UI_DIR / "static"))
worker = CounterWorker()

def fmt(dt: datetime | None) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S") if dt else "-"

@app.route("/")
def index():
    return render_template("index.html") if (UI_DIR / "templates" / "index.html").exists() else (
        "<html><body style='background:#111;color:#eee;font-family:system-ui'>"
        "<h2>Conteo desde VIDEO (fluido y estable)</h2>"
        "<div><img src='/stream' style='max-width:100%;border:1px solid #444'/></div>"
        "<div style='margin-top:12px'>"
        "<button onclick='fetch(\"/start\",{method:\"POST\"}).then(()=>location.reload())'>Iniciar</button>"
        "<button onclick='fetch(\"/stop\",{method:\"POST\"}).then(()=>location.reload())' style='margin-left:8px'>Detener</button>"
        "</div>"
        "<script>setInterval(()=>fetch('/metrics').then(r=>r.json()).then(m=>{document.title=`Total ${m.total}`;}),1000)</script>"
        "</body></html>"
    )

@app.route("/start", methods=["POST"])
def start():
    worker.start()
    return jsonify(ok=True, running=worker.running, start_time=fmt(worker.start_time), total=worker.total)

@app.route("/stop", methods=["POST"])
def stop():
    worker.stop()
    return jsonify(ok=True, running=worker.running, last_saved=fmt(worker.last_saved), total=worker.total)

@app.route("/metrics")
def metrics():
    return jsonify(
        running=worker.running,
        total=worker.total,
        start_time=fmt(worker.start_time),
        last_saved=fmt(worker.last_saved)
    )

@app.route("/stream")
def stream():
    def gen():
        boundary = b"--frame"
        while True:
            with worker.lock:
                frame = worker.latest_jpeg
            if frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                ok_jpg, jpg = cv2.imencode(".jpg", blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
                frame = jpg.tobytes() if ok_jpg else b""
            yield (boundary + b"\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
            time.sleep(0.03)  # ~33 fps en el stream
    return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

if __name__ == "__main__":
    print(f"[CHECK] UI_DIR   -> {UI_DIR}")
    print(f"[CHECK] ROOT     -> {ROOT}")
    print(f"[CHECK] DATA     -> {DATA}  (exists={DATA.exists()})")
    print(f"[CHECK] MODELS   -> {MODELS} (exists={MODELS.exists()})")
    print(f"[CHECK] SCRIPTS  -> {SCRIPTS} (exists={SCRIPTS.exists()})")
    print(f"[CHECK] VIDEO    -> {VIDEO_PATH} (exists={VIDEO_PATH.exists()})")
    print(f"[CHECK] MODEL    -> {MODEL_PATH} (exists={MODEL_PATH.exists()})")
    print(f"[CHECK] MASK     -> {MASK_PATH} (exists={MASK_PATH.exists()})")

    # Ya no se auto-inicia el worker; solo con el botón /start en el navegador
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
