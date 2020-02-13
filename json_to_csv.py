from os import walk
import json
import glob, os
import random

base = os.getcwd()
f = []
for file in glob.glob("**/**/**/*.json"):
    f.append(file)

train_jsons = [f[i] for i in range(len(f)) if i%10>=1]
eval_jsons = [f[i] for i in range(len(f)) if i%10<1]

def make_csv(csv_path, files):
    with open(csv_path, "w+") as csv:
        csv.write("filename,width,height,class,xmin,ymin,xmax,ymax\n")

        for filename in files:
            
            path = base + '/' + filename
            filename = filename.split('/')
            
            filename[-2] = 'img'
            filename[-1] = filename[-1][:-5]
            with open(path, 'r') as file:
                line = json.loads(file.readlines()[0])
                for obj in line["objects"]:

                    p1, p2 = obj["points"]["exterior"]
                    x1, x2 = sorted([p1[0], p2[0]])
                    y1, y2 = sorted([p1[1], p2[1]])
                    
                    entry = ['/'.join(filename), line["size"]["width"], line["size"]["height"], obj["classTitle"], x1, y1, x2, y2]
                    csv.write(",".join(map(str, entry)) + '\n')
        
make_csv("{}/training/train.csv".format(base), train_jsons)
make_csv("{}/training/eval.csv".format(base), eval_jsons)
