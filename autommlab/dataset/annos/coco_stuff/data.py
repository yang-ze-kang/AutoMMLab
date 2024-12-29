import cv2
import numpy as np
from os.path import splitext
from petrel_client.client import Client

conf_path = '~/petreloss.conf'
client = Client(conf_path) # 若不指定 conf_path ，则从 '~/petreloss.conf' 读取配置文件

root = 'sh1984:s3://openmmlab/datasets/segmentation/coco_stuff164k'


def summary_img2labels():
    anno_root = root+'/annotations/'
    for k in ['train2017','val2017']:
        anno_root = anno_root+k+'/'
        for i in client.list(anno_root):
            print(i)

# summary_img2labels()

img_path = 'sh1984:s3://openmmlab/datasets/segmentation/coco_stuff164k/train2017/000000078621.png'
img_bytes = client.get(img_path)
assert(img_bytes is not None)

# 图片读取
# img_bytes = client.get(img_url)
# assert(img_bytes is not None)
# img_mem_view = memoryview(img_bytes)
# img_array = np.frombuffer(img_mem_view, np.uint8)
# img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

# # 图片处理
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# # 图片存储
# success, img_gray_array = cv2.imencode(img_ext, img_gray)
# assert(success)
# img_gray_bytes = img_gray_array.tostring()
# client.put(img_gray_url, img_gray_bytes)
