from ultralytics import YOLO

def main():
    modelo = YOLO('yolov8n.pt')
    resultado = modelo.train(
        data=r'C:\Users\guilherme\Desktop\TCC\imagens\data.yaml',
        epochs=30, imgsz=640, device='cpu', workers=0
    )


if __name__== '__main__':
    main()
