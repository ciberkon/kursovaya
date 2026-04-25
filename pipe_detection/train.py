from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
    data="data.yaml",
    epochs=40,
    imgsz=416,
    batch=16,
    device=0,
    patience=15,
    cache=True,
    workers=2
)

if __name__ == "__main__":
    main()