from ultralytics import YOLO
import cv2
import numpy as np
import time
import os

MODEL_COCO = "yolov8n.pt"
MODEL_EPI  = r"C:\Users\guilherme\Desktop\TCC\.venv\runs\detect\train2\weights\best.pt"


CAM_INDEX     = 0
CONF_PESSOA   = 0.35
CONF_EPI      = 0.35
HEAD_FACTOR   = 0.35
CHEST_TOP     = 0.30
CHEST_FACTOR  = 0.45
IOU_MIN_MATCH = 0.15
SHOW_FPS      = True


def iou(a, b):
    xA, yA = max(a[0], b[0]), max(a[1], b[1])
    xB, yB = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter <= 0:
        return 0.0
    areaA = (a[2]-a[0])*(a[3]-a[1])
    areaB = (b[2]-b[0])*(b[3]-b[1])
    return inter / float(areaA + areaB - inter)


print("Carregando modelos...")
coco = YOLO(MODEL_COCO)
epi  = YOLO(MODEL_EPI)

names_epi = {i: n.lower() for i, n in epi.names.items()}
id_capacete = next((i for i,n in names_epi.items() if 'capacete' in n), None)
id_colete   = next((i for i,n in names_epi.items() if 'colete'   in n), None)
print(f"id_capacete={id_capacete} | id_colete={id_colete} | classes={names_epi}")


print("Iniciando webcam...")
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    print("Nao foi possível abrir a webcam. Verifique as permissoes ou tente outro indice")
    raise SystemExit

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# === Configurar pasta e gravador ===
output_dir = r"C:\Users\guilherme\Desktop\TCC\Videos"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"detec_epi_{time.strftime('%Y%m%d_%H%M%S')}.mp4")

fps_out = 20.0
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps_out, (width, height))

print(f"Gravando vídeo em: {output_path}")

cv2.namedWindow("EPI - Webcam (Gravando)", cv2.WINDOW_NORMAL)
t0, frames = time.time(), 0

while True:
    ok, frame = cap.read()
    if not ok:
        print("Frame não lido")
        break


    res_p = coco(frame, classes=[0], conf=CONF_PESSOA, device='cpu', verbose=False)[0]
    pessoas = res_p.boxes.xyxy.cpu().numpy() if res_p.boxes is not None else np.empty((0,4))


    res_e = epi(frame, conf=CONF_EPI, device='cpu', verbose=False)[0]
    epi_boxes = res_e.boxes.xyxy.cpu().numpy() if res_e.boxes is not None else np.empty((0,4))
    epi_cls   = res_e.boxes.cls.cpu().numpy().astype(int) if res_e.boxes is not None else np.empty((0,), dtype=int)
    epi_conf  = res_e.boxes.conf.cpu().numpy() if res_e.boxes is not None else np.empty((0,))


    capacetes = [b for b,c in zip(epi_boxes, epi_cls) if id_capacete is not None and c == id_capacete]
    coletes   = [b for b,c in zip(epi_boxes, epi_cls) if id_colete   is not None and c == id_colete]

    annotated = frame.copy()


    for b, c, cf in zip(epi_boxes, epi_cls, epi_conf):
        if (id_capacete is not None and c == id_capacete) or (id_colete is not None and c == id_colete):
            x1, y1, x2, y2 = map(int, b)
            color = (0,200,0) if c == id_capacete else (255,165,0)
            label = f"{names_epi.get(int(c),'cls')} {cf:.2f}"
            cv2.rectangle(annotated, (x1,y1), (x2,y2), color, 2)
            cv2.putText(annotated, label, (x1, max(20,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


    for p in pessoas:
        x1,y1,x2,y2 = p
        h = y2 - y1
        head_box  = np.array([x1, y1, x2, y1 + h*HEAD_FACTOR], dtype=float)
        chest_box = np.array([x1, y1 + h*CHEST_TOP, x2, y1 + h*(CHEST_TOP + CHEST_FACTOR)], dtype=float)

        tem_capacete = any(iou(head_box,  b) >= IOU_MIN_MATCH for b in capacetes)
        tem_colete   = any(iou(chest_box, b) >= IOU_MIN_MATCH for b in coletes)

        if tem_capacete and tem_colete:
            color, text = (0,200,0), "EPI Completo"
        else:
            faltas = []
            if not tem_capacete: faltas.append("capacete")
            if not tem_colete:   faltas.append("colete")
            color, text = (0,0,255), "Sem " + ", ".join(faltas)

        cv2.rectangle(annotated, (int(x1),int(y1)), (int(x2),int(y2)), color, 2)
        cv2.putText(annotated, text, (int(x1), max(20, int(y1)-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # FPS
    frames += 1
    if SHOW_FPS:
        dt = time.time() - t0
        if dt >= 1.0:
            fps = frames / dt
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            t0, frames = time.time(), 0

    cv2.imshow("EPI - Webcam (Gravando)", annotated)
    out.write(annotated)

    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):  # 'q' ou ESC
        break

# === Encerrar ===
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Encerrado. Vídeo salvo em: {output_path}")
