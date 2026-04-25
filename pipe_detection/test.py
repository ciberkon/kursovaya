import os

img_dir = "dataset/val/images"
lbl_dir = "dataset/val/labels"

images = set(os.path.splitext(f)[0] for f in os.listdir(img_dir))
labels = set(os.path.splitext(f)[0] for f in os.listdir(lbl_dir))

print("Images:", len(images))
print("Labels:", len(labels))

print("Нет label для:")
print(images - labels)

print("Лишние label:")
print(labels - images)