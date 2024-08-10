from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import json

# 加载配置
with open('config.json', encoding='utf-8') as config_file:
    config = json.load(config_file)


class MultimodalModule:
    def __init__(self):
        # 加载模型
        self.blip_model = BlipForConditionalGeneration.from_pretrained(config['blip_model_path'])
        self.processor = BlipProcessor.from_pretrained(config['blip_model_path'])

    def process_image(self, image_path):
        image = Image.open(image_path)                              # 打开图像文件
        inputs = self.processor(images=image, return_tensors="pt")  # 使用处理器将图像转换为模型输入的张量格式
        outputs = self.blip_model.generate(**inputs, max_length=50)  # 使用BLIP模型生成图像描述
        description = self.processor.decode(outputs[0], skip_special_tokens=True)  # 解码生成的输出为可读的文本描述
        return description


if __name__ == "__main__":
    multimodal_module = MultimodalModule()
    test_image_path = r'./data/image1.jpg'
    description = multimodal_module.process_image(test_image_path)    # 调用process_image方法获取图像的描述

    print("Image Description:", description)
