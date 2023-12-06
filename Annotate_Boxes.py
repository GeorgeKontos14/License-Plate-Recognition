import cv2
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from csv import writer
import os

def show_frame(filepath, no_frame, out):
    cap = cv2.VideoCapture(filepath)
    if cap.isOpened()==False:
        print("Error")

    i = 1
    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            if i == no_frame:
                cv2.imwrite(out, frame)
                plt.connect('button_press_event', on_click)
                plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                plt.show()
                break
            i += 1
    cap.release()
    cv2.destroyAllWindows()

def on_click(event):
    global xs
    global ys
    if len(xs) == 4:
        return
    xs.append(round(event.xdata))
    ys.append(round(event.ydata))

# Script for category I
"""count = 1
path = "dataset/TrainingSet/Categorie I/"

for filename in os.scandir(path):
    if filename.is_file():
        ys = []
        xs = []
        out = 'dataset/Frames/Category_I/plate' + str(count) + '.jpg'
        count += 1
        show_frame(filename.path, 1, out)
        max_x = max(xs)
        max_y = max(ys)
        min_x = min(xs)
        min_y = min(ys)
        l = [filename.name, 1, max_y, min_y, min_x, max_y]
        with open('dataset/bounding_boxes.csv', 'a') as f_object:
            writer_object = writer(f_object)
            writer_object.writerow(l)
            f_object.close()"""

# Script for category II
"""count = 1
path = "dataset/TrainingSet/Categorie II"
        
for filename in os.scandir(path):
    if filename.is_file():
        no = 1
        while no < 720:
            ys = []
            xs = []
            out = 'dataset/Frames/Category_II/plate' + str(count) + '.jpg'
            show_frame(filename.path, no, out)
            if len(xs) == 4:
                count += 1
                max_x = max(xs)
                max_y = max(ys)
                min_x = min(xs)
                min_y = min(ys)
                l = [filename.name, no, max_y, min_y, min_x, max_y]
                with open('dataset/bounding_boxes.csv', 'a') as f_object:
                    writer_object = writer(f_object)
                    writer_object.writerow(l)
                    f_object.close()
            no += 1"""

# Script for category III
count = 1
path = "dataset/TrainingSet/Categorie III"


for filename in os.scandir(path):
    c = 0
    if filename.is_file():
        while c < 2:
            ys = []
            xs = []
            out = 'dataset/Frames/Category_III/plate' + str(count) + '.jpg'
            if c == 1:
                count += 1
            show_frame(filename.path, 4, out)
            max_x = max(xs)
            max_y = max(ys)
            min_x = min(xs)
            min_y = min(ys)
            l = [filename.name, 4, max_y, min_y, min_x, max_y]
            with open('dataset/bounding_boxes.csv', 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(l)
                f_object.close()
            c += 1