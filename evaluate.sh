python val.py --weights runs/train/exp2/weights/best.pt --data dataset/data-augmented.yaml --img-size 640 --batch-size 16

python test.py --weights runs/train/exp2/weights/best.pt --data dataset/data-augmented.yaml --img-size 640 --batch-size 16
