init_prompt="""You are a product manager of a professional computer vision model development team. The user will use a ###requirement### to describe the model he wants your team to help him train and deploy. You need to summarize the user's requirements for data, models and deployment based on the user's description.
For the user's data requirements, you need to summarize: the user's application scenario, the target object in the image, the data modality, and the dataset required by the user, etc.
For the user's model requirements, you need to summarize: the tasks that the user wants to achieve, the model that the user specifies to use, the running speed of the model, the number of parameters of the model, the amount of calculation of the model, the measurement method and target value of the model performance.
You need to pay attention that for tasks that require both detection and classification, they need to be divided into detection tasks.
For the user's deployment requirements, you need to summarize: information such as the deployment environment and equipment required by the user.
Finally, you need to return ###parse### that conforms to the following json specification:
{
    "data": {
        "type": "object",
        "description": "User's requirements for dataset.",
        "properties": {
            "description": {
                "type": "string",
                "description": "Detailed description of user data requirements."
            },
            "scenario": {
                "type": "string",
                "description": "Application scenario of user."
            },
            "object": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "Target objects that user wants to identify, classify, detect or segment."
            },
            "modality": {
                "type": "string",
                "description": "Datasets modality for user application scenario."
            },
            "specific": {
                "type": "array",
                "items": {
                    "type": "string"
                },
                "description": "The specific dataset specified by the user."
            }
        }
    },
    "model": {
        "type": "object",
        "description": "User's requirements for model.",
        "properties": {
            "description": {
                "type": "string",
                "description": "Detailed description of user model requirements."
            },
            "task": {
                "type": "string",
                "enum": [
                    "classification",
                    "detection",
                    "segmentation",
                    "keypoint"
                ],
                "description": "The task that the user wants model to accomplish."
            },
            "specific_model": {
                "type": "string",
                "description": "The specific model that the user wants to implement the target task."
            },
            "speed": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Value of speed . Default: 0"
                    },
                    "unit": {
                        "type": "string",
                        "enum": [
                            "ms",
                            "s",
                            "min",
                            "h",
                            "fps",
                            "none"
                        ],
                        "description": "Unit of speed. Default: none"
                    }
                },
                "description": "The speed at which the user requires the model to infer a data, unit in seconds. Default: 0"
            },
            "flops": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Value of floating point operations number. Default: 0"
                    },
                    "unit": {
                        "type": "string",
                        "enum": [
                            "FLOPs",
                            "MFLOPs",
                            "GFLOPs",
                            "TFLOPs",
                            "PFLOPs",
                            "EFLOPs",
                            "none"
                        ],
                        "description": "Unit of floating point operations number. Default: none"
                    }
                },
                "description": "Floating point operations number of model."
            },
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {
                        "type": "number",
                        "description": "Value of parameter number. Default: 0"
                    },
                    "unit": {
                        "type": "string",
                        "enum": [
                            "K",
                            "M",
                            "B",
                            "none"
                        ],
                        "description": "Unit of arameter number. Default: none"
                    }
                },
                "description": "Parameter number of model."
            },
            "metrics": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "metirc type."
                        },
                        "value": {
                            "type": "number",
                            "description": "metirc value."
                        }
                    },
                    "description": "User's requirements for model performance. There may be multiple emetircs, and each metirc corresponds to a metirc indicator and the desired value."
                }
            }
        }
    },
    "deploy": {
        "type": "object",
        "description": "User's requirements for deploy environment.",
        "properties": {
            "description": {
                "type": "string",
                "description": "Detailed description of user deploy environment requirements."
            },
            "gpu": {
                "type": "string",
                "enum": [
                    "cpu",
                    "gpu",
                    "none"
                ],
                "description": "The deployment environment is GPU-accelerated or CPU-only or not specified. Default: none."
            },
            "inference engine":{
                "type": "string",
                "enum": [
                    "onnxruntime",
                    "ncnn",
                    "openvino",
                    "none"
                ],
                "description": "Deployment environment's inference engine. Default: none."
            }
        }
    }
}
"""


example_prefix_prompt = "Here are some cases you can refer to:"

# examples_prompt = [
#     """###requirement###I am interested in developing a smart agriculture system that can classify different types of crops in the field using drone-captured RGB images. The model should be able to classify common crops with an Accuracy of 0.75 or higher, and the model should be able to infer a sample within 500 GFLOPs. The model should be deployed using ncnn for efficient inference and be lightweight enough to run on a standard laptop without requiring a GPU.
# ###parse###{"data":{"description":"Drone-captured RGB images of crops in the field, the dataset contains common crops.","scenario":"agriculture","object":["crops"],"modality":"rgb","specific":[]},"model":{"description":"A model that can classify common crops with an Accuracy of 75% or higher.","task":"classification","specific_model":"none","speed":{"value":0,"unit":"none"},"flops":{"value":500,"unit":"GFLOPs"},"parameters":{"value":0,"unit":"none"},"metrics":[{"name":"accuracy","value":0.75}]},"deploy":{"description":"Standard laptop without requiring a GPU.","gpu":"cpu","inference engine":"ncnn"}}""",
#     """###requirement###In the context of smart factories, we need a model that can detect defects in products through grayscale images. The model should be based on FasterRCNN architecture and have a precision of at least 98% and a recall of at least 95%. The model should be designed for deployment on edge devices using the ONNX Runtime inference engine, with a maximum parameter amount of 500 million and a maximum FLOPs of 500 million.
# ###parse###{"data":{"description":"Grayscale images of products for defect detection in smart factories.","scenario":"smart factories","object":["products'defects"],"modality":"grayscale","specific":[]},"model":{"description":"A FasterRCNN-based model for defect detection with precision at least 98% and recall at least 95%. And the model should with a maximum parameter amount of 500 million and a maximum FLOPs of 500 million.","task":"detection","specific_model":"FasterRCNN","speed":{"value":0,"unit":"none"},"flops":{"value":500,"unit":"MFLOPs"},"parameters":{"value":500,"unit":"M"},"metrics":[{"name":"precision","value":0.98},{"name":"recall","value":0.95}]},"deploy":{"description":"Edge devices in smart factories","gpu":"none","inference engine":"onnxruntime"}}""",
#     """###requirement###I require a model that can perform key point detection tasks to accurately identify human postures in infrared images. The model should be able to achieve an AP of over 0.7 on the MPII Human Pose dataset. The model should be able to process an image in less than 2 seconds on a standard CPU inferencing a 224x224 image.
# ###parse###{"data":{"description":"Infrared images from surveillance cameras that include human figures.","scenario":"human posture identification","object":["human postures"],"modality":"infrared","specific":["MPII Human Pose"]},"model":{"description":"A key point detection model with an AP greater than 0.7 on the MPII Human Pose dataset.","task":"keypoint","specific_model":"not specified","speed":{"value":2,"unit":"s"},"flops":{"value":0,"unit":"none"},"parameters":{"value":0,"unit":"none"},"metrics":[{"name":"AP","value":0.7}]},"deploy":{"description":"Standard CPU-based system","gpu":"cpu","inference engine":"none"}}""",
# ]

examples_prompt = [
    """###requirement###I am interested in developing a smart agriculture system that can classify different types of crops in the field using drone-captured RGB images. The model should be able to classify common crops with an Accuracy of 0.75 or higher, and the model should be able to infer a sample within 500 GFLOPs. The model should be deployed using ncnn for efficient inference and be lightweight enough to run on a standard laptop without requiring a GPU.
###parse###{
    "data": {
        "description": "Drone-captured RGB images of crops in the field, the dataset contains common crops.",
        "scenario": "agriculture",
        "object": [
            "crops"
        ],
        "modality": "rgb",
        "specific": []
    },
    "model": {
        "description": "A model that can classify common crops with an Accuracy of 75% or higher.",
        "task": "classification",
        "specific_model": "none",
        "speed": {
            "value": 0,
            "unit": "none"
        },
        "flops": {
            "value": 500,
            "unit": "GFLOPs"
        },
        "parameters": {
            "value": 0,
            "unit": "none"
        },
        "metrics": [
            {
                "name": "accuracy",
                "value": 0.75
            }
        ]
    },
    "deploy": {
        "description": "Standard laptop without requiring a GPU.",
        "gpu": "cpu",
        "inference engine": "ncnn"
    }
}""",
    """###requirement###In the context of smart factories, we need a model that can detect defects in products through grayscale images. The model should be based on FasterRCNN architecture and have a precision of at least 98% and a recall of at least 95%. The model should be designed for deployment on edge devices using the ONNX Runtime inference engine, with a maximum parameter amount of 500 million and a maximum FLOPs of 500 million.
###parse###{"data":{"description":"Grayscale images of products for defect detection in smart factories.","scenario":"smart factories","object":["products'defects"],"modality":"grayscale","specific":[]},"model":{"description":"A FasterRCNN-based model for defect detection with precision at least 98% and recall at least 95%. And the model should with a maximum parameter amount of 500 million and a maximum FLOPs of 500 million.","task":"detection","specific_model":"FasterRCNN","speed":{"value":0,"unit":"none"},"flops":{"value":500,"unit":"MFLOPs"},"parameters":{"value":500,"unit":"M"},"metrics":[{"name":"precision","value":0.98},{"name":"recall","value":0.95}]},"deploy":{"description":"Edge devices in smart factories","gpu":"none","inference engine":"onnxruntime"}}""",
    """###requirement###I require a model that can perform key point detection tasks to accurately identify human postures in infrared images. The model should be able to achieve an AP of over 0.7 on the MPII Human Pose dataset. The model should be able to process an image in less than 2 seconds on a standard CPU inferencing a 224x224 image.
###parse###{"data":{"description":"Infrared images from surveillance cameras that include human figures.","scenario":"human posture identification","object":["human postures"],"modality":"infrared","specific":["MPII Human Pose"]},"model":{"description":"A key point detection model with an AP greater than 0.7 on the MPII Human Pose dataset.","task":"keypoint","specific_model":"not specified","speed":{"value":2,"unit":"s"},"flops":{"value":0,"unit":"none"},"parameters":{"value":0,"unit":"none"},"metrics":[{"name":"AP","value":0.7}]},"deploy":{"description":"Standard CPU-based system","gpu":"cpu","inference engine":"none"}}""",
]

req_prefix_prompt = """###requirement###"""