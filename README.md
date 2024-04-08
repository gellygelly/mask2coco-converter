# mask2coco converter

multiclass mask convert to coco json file

## Overview
[MMDetection] https://github.com/open-mmlab/mmdetection  
[BCSS dataset] https://www.kaggle.com/datasets/whats2000/breast-cancer-semantic-segmentation-bcss

mmdetection을 활용한 swin mask rcnn 모델 학습을 위해 multiclass mask image를 coco format의 json file로 변환하는 코드


## File Description

**mask2coco-converter**  
  └── mask2coco-converter  
      ├── annotations  
      │   └── coco_annotation_sample.json  
      ├── data  
      │   ├── train  
      │   ├── train_mask  
      │   ├── val  
      │   ├── val_mask  
      │   └── gtruth_codes_512.tsv  
      ├── analysis.py  
      ├── mask2coco.py  
      └── visualize_mask2points.py  

- annotations: coco format의 json file 예시
- data: BCSS dataset sample(image, mask)
- analysis.py: 각 폴더 내 이미지의 class별 object 개수를 카운트하는 코드 / BCSS 224 dataset(0(outside roi), 1(tumor), 2(stroma))기준으로 코드 작성
- **mask2coco.py**: multiclass mask image를 coco format의 json file로 변환하는 코드 / BCSS 224 dataset(0(outside roi), 1(tumor), 2(stroma))기준으로 코드 작성
- visualize_mask2points.py: image, mask image, contour image, each class sub mask images 시각화 코드

## multiclass mask to coco json file

![그림1](https://github.com/gellygelly/mask2coco-converter/assets/54652700/27e59fa8-084e-45fd-b171-33da1681a5b1)

## How to use
1. mask image와 image 쌍으로 이루어진 dataset 준비 (※ 파일명은 반드시 숫자여야 함)
2. mask2coco.py 코드 실행
  - mask 파일 경로, 생성되는 annotation file 경로 수정
  - info, license, categories 정보 필요시 수정
