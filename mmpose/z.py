from mmpose.apis import MMPoseInferencer

img_path = 'tests/data/coco/000000000785.jpg'   # 将img_path替换给你自己的路径

# 使用模型别名创建推断器
inferencer = MMPoseInferencer('human')

# MMPoseInferencer采用了惰性推断方法，在给定输入时创建一个预测生成器
result_generator = inferencer(img_path, show=True)
result = next(result_generator)