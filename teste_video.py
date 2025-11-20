from ultralytics import YOLO
import cv2

MODEL_PATH = r"C:\Users\guilherme\Desktop\TCC\.venv\runs\detect\train2\weights\best.pt"

VIDEO_PATH = r"C:\Users\guilherme\Desktop\TCC\Videos\epi-2.mp4"

OUTPUT_PATH = r"C:\Users\guilherme\Desktop\TCC\Videos\resultado_epi-2.mp4"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

out = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

print("Rodando inferência no vídeo...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    results = model(frame, conf=0.25)


    annotated_frame = results[0].plot()

    cv2.imshow("Detecção de EPI - YOLO", annotated_frame)

    out.write(annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Vídeo processado e salvo em: {OUTPUT_PATH}")
