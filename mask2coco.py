# convert multiclass mask image to annotation json file 

import os
import json
from collections import OrderedDict
import datetime
import cv2 
import numpy as np
from tqdm.auto import tqdm


############### json comp ###############

def get_coco_info():
    info = {
        'description': 'BCSS custom dataset',
        'url': 'https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss/data',
        'version': '1.0',
        'year': 2024,
        'contributor': '李晨愷',
        'data_created': str(datetime.datetime.now())
    }

    return info

def get_coco_licenses():
    licenses = [
        {
            'url': 'https://creativecommons.org/publicdomain/zero/1.0/',
            'id': 1,
            'name': 'CC0: Public Domain'
        }
    ]

    return licenses

def get_coco_image(file_name, height, width):
    image = {}
    # warning: coco format requires file names to be numeric (ex) 1.png, 1234.png, 0000004.png
    image['license'] = 1
    image['file_name'] = file_name # 파일 확장자 포함 파일명 (ex) '1.png'
    image['coco_url'] = '' 
    image['height'] = height 
    image['width'] = width 
    image['date_captured'] = str(datetime.datetime.now())
    image['flickr_url'] = ''
    image['id'] = int(remove_file_extension(file_name)) # 파일 확장자 제거한 파일명 (ex) 1

    return image

def get_coco_anno(annotations, file_name, mask, category_id, segm_id):

    # find contours
    _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    # print(file_name, len(contours), np.unique(mask))

    for contour in contours:
        annotation_info = {}

        segm_points =  contour.flatten().tolist()

        # prevent TypeError: Argument 'bb' has incorrect type (expected numpy.ndarray, got list)
        if len(segm_points) <= 4:
            continue

        annotation_info['segmentation'] = [segm_points]  # segmetnation = [x1,y1, x2,y2 ,x3,y3...]
        annotation_info['area'] = cv2.contourArea(contour)
        annotation_info['iscrowd'] = 0
        annotation_info['image_id'] = int(remove_file_extension(file_name))
        annotation_info['bbox'] = cv2.boundingRect(contour)
        annotation_info['category_id'] = category_id # TODO

        annotation_info['id'] = segm_id

        segm_id += 1 
        annotations.append(annotation_info)

    return segm_id


############### utils ###############

def get_coco_categories():

    categories = [
        {
            'supercategory': 'cell',
            'id':1,
            'name': 'outside_roi'
        },
        {
            'supercategory': 'cell',
            'id':2,
            'name': 'tumor'
        },
        {
            'supercategory': 'cell',
            'id':3, 
            'name': 'stroma'
        }
    ]

    return categories


def remove_file_extension(file_name):
    
    if file_name.endswith('.jpg'):
        file_name = file_name.replace('.jpg', '')
    elif file_name.endswith('.png'):
        file_name = file_name.replace('.png', '')
    
    return file_name


# FIXME class 개수 받아와서 range에 class 개수 넣어주는걸로 변경?
def create_sub_mask(mask, width, height):
    sub_mask_imgs = [np.zeros_like(mask) for _ in range(3)] # class: 0, 1, 2

    for i in range(width):
        for j in range(height):
            pixel_value = mask[i][j]
            if pixel_value in [0, 1, 2]:
                sub_mask_imgs[pixel_value][i][j] = 255
    
    return sub_mask_imgs


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


############### main ###############

def main():
    coco_json = OrderedDict()

    mask_image_path = 'data/train_mask' 

    ### simple info (모든 이미지 공통사항) ###
    info = get_coco_info()
    licenses = get_coco_licenses()
    categories = get_coco_categories()

    ### image, annotation info (개별 이미지 정보) ###
    images = []
    annotations = [] 

    files = os.listdir(mask_image_path)

    segm_id = 1

    for file in tqdm(files):
        # image load
        file_path = mask_image_path+'/'+file
        mask = cv2.imread(file_path)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        width, height = mask.shape

        # get images info
        image = get_coco_image(file, width, height)
        images.append(image)

        # extract submasks
        sub_masks = create_sub_mask(mask, width, height)
        for category_id, sub_mask in enumerate(sub_masks):
            # get annotations info
            segm_id = get_coco_anno(annotations, file, sub_mask, category_id, segm_id)

    print(segm_id)
    # make coco_json
    coco_json['info'] = info
    coco_json['licenses'] = licenses
    coco_json['images'] = images
    coco_json['annotations'] = annotations
    coco_json['categories'] = categories

    # write json
    with open('annotations.json', 'w', encoding='utf-8') as make_file:
        json.dump(coco_json, make_file, cls=NpEncoder, ensure_ascii=False, indent='\t')
    
    
# RUN
if __name__ == '__main__':
    main()
