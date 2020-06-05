import cv2
import argparse
import os

parser = argparse.ArgumentParser( description="Resize image to a fix vale and rewrite the text files")
parser.add_argument("--old_size", type=str, default="800 600", help="Size of old pictures, this should be x y")
parser.add_argument("--new_size", type=str, default="600 600", help="Size of new pictures, this should be x y")
parser.add_argument("--old_pic_path", type=str, default="data/barcodes/images/", help="Path to old picture")
parser.add_argument("--new_pic_path", type=str, default="data/barcodes/new_images/", help="Path to new images")
#parser.add_argument("--old_label_path", type=str, default="data/barcodes/labels/", help="Path to old labels")
#parser.add_argument("--new_label_path", type=str, default="data/barcodes/new_labels/", help="Path to new labels")

opt = parser.parse_args()

# Make the directories for new pictures and new labels
os.makedirs(opt.new_pic_path, exist_ok=True)
#os.makedirs(opt.new_label_path, exist_ok=True)


files_old = os.listdir(opt.old_pic_path)

for ifile in files_old:
    img_old = cv2.imread(ifile)
    new_img =










