
URL_LLAMA = "http://127.0.0.1:10069/llama2"
TRAIN_GPU_NUM = 1
RU_MODEL = 'ru-llama2'
HPO_MODEL = 'hpo-llama2'
HPO_MAX_TRY = 3
TENSORBOARD_PORT = 10066
IP_ADDRESS = 'localhost'

DATASET_ZOO = {
    'ImageNet':'sh1984:s3://openmmlab/datasets/classification/imagenet',
    'COCO':'sh1984:s3://openmmlab/datasets/detection/coco',
    'object365': 'sh1984:s3://openmmlab/datasets/detection/Objects365',
    'openimage': 'sh1984:s3://openmmlab/datasets/detection/coco',
    'cityscapes':'s3://openmmlab/datasets/segmentation/cityscapes',
    'ap10k':'sh1986:s3://ap10k/ap-10k/'
}

PATH_LLAMA2 = '/data3/yangzekang/LLMaC/llama_inference/llama_weights/llama-2-7b-hf'
PATH_LORAS = {
    'ru-llama2':'weights/llama2_lora_weights/save_dir_reqparse_v2',
    'hpo-llama2-classification':'weights/llama2_lora_weights/hpo_classification',
    'hpo-llama2-detection':'weights/llama2_lora_weights/hpo_detection',
    'hpo-llama2-segmentation':'weights/llama2_lora_weights/hpo_segmentation',
    'hpo-llama2-pose':'weights/llama2_lora_weights/hpo_pose'
}
