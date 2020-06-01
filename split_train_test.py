import os
import random

directory = os.getcwd()+"/data/barcodes/"


files_img = os.listdir(directory+"images/")
random.shuffle(files_img)

print(files_img)

test_sample = int(0.1*len(files_img))

train_files = files_img[test_sample:]
test_files = files_img[:test_sample]

with open(directory+"train.txt", "w") as ftrain, open(directory+"val.txt", "w") as fval:
    for ifile in train_files:
        fname = directory+"images/"+ifile+"\n"
        ftrain.write(fname)

    for ifile in test_files:
        fname = directory+"images/"+ifile+"\n"
        fval.write(fname)








