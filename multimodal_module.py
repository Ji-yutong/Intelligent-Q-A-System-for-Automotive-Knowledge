from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json

# 加载配置
with open('config.json', encoding='utf-8') as config_file:  # 打开配置文件
    config = json.load(config_file)  # 从配置文件中读取设置，包括模型路径等

class MultimodalModule:
    def __init__(self):
        self.blip_model = BlipForConditionalGeneration.from_pretrained(config['blip_model_path'])  # 加载BLIP模型
        self.processor = BlipProcessor.from_pretrained(config['blip_model_path'])                   # 加载BLIP处理器

    def process_image(self, image_path):
        image = Image.open(image_path)                              # 打开图像文件
        inputs = self.processor(images=image, return_tensors="pt")  # 使用处理器将图像转换为模型输入的张量格式
        outputs = self.blip_model.generate(**inputs, max_length=50)  # 使用BLIP模型生成图像描述
        description = self.processor.decode(outputs[0], skip_special_tokens=True)  # 解码生成的输出为可读的文本描述
        return description  # 返回图像描述

if __name__ == "__main__":
    multimodal_module = MultimodalModule()                                  # 实例化多模态模块
    test_image_path = r'D:\pytorch project\汽车知识问答系统\data\image1.jpg'    # 图像路径
    description = multimodal_module.process_image(test_image_path)          # 调用process_image方法获取图像的描述

    print("Image Description:", description)  # 打印图像的描述
