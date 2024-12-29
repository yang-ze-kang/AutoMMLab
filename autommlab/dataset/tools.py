from pycocotools.coco import COCO

def generate_label_num(train_json_path,val_json_path,out_path='label_num.txt'):
    train_coco = COCO(train_json_path)
    val_coco = COCO(val_json_path)
    with open(out_path,'w') as f:
        for cat_id in train_coco.getCatIds():
            name = train_coco.loadCats(ids=[cat_id])[0]['name']
            train_num = len(train_coco.getImgIds(catIds=[cat_id]))
            val_num = len(val_coco.getImgIds(catIds=[cat_id]))
            print(name,train_num,val_num)
            f.write(f"{name},{train_num},{val_num}\n")

def generate_labels_name(json_path):
    coco = COCO(json_path)
    categories = coco.loadCats(coco.getCatIds())
    category_names = [category['name'] for category in categories]
    print(category_names)

if __name__=='__main__':
    # generate_label_num('/data3/yangzekang/LLMaC/llama_inference/dataset/coco/instances_train2017.json','/data3/yangzekang/LLMaC/llama_inference/dataset/coco/instances_val2017.json','/data3/yangzekang/LLMaC/llama_inference/dataset/coco/label_num.txt')
    # generate_label_num('/data3/yangzekang/LLMaC/llama_inference/dataset/lvis/lvis_v1_train.json','/data3/yangzekang/LLMaC/llama_inference/dataset/lvis/lvis_v1_val.json','/data3/yangzekang/LLMaC/llama_inference/dataset/lvis/label_num.txt')
    # generate_label_num('/data3/yangzekang/LLMaC/llama_inference/dataset/cityscapes/instancesonly_filtered_gtFine_train.json','/data3/yangzekang/LLMaC/llama_inference/dataset/cityscapes/instancesonly_filtered_gtFine_val.json','/data3/yangzekang/LLMaC/llama_inference/dataset/cityscapes/label_num.txt')
    generate_labels_name('/data3/yangzekang/LLMaC/llama_inference/dataset/cityscapes/instancesonly_filtered_gtFine_train.json')