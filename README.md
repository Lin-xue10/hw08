# hw08
hw08/
├── app.py              # 单文件主程序
├── requirements.txt    # 依赖清单
├── README.md           # 项目说明
└── report.md           # 项目报告
streamlit>=1.30.0
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms

# 页面设置
st.set_page_config(page_title="图片物体识别")
st.title("🖼️ AI图片物体识别工具")

# 加载预训练AI模型（首次运行自动下载权重）
@st.cache_resource
def load_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.eval()
    return model

# 图片预处理
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    return transform(image).unsqueeze(0)

# 获取类别名称
@st.cache_resource
def get_class_names():
    weights = models.ResNet18_Weights.DEFAULT
    return weights.meta["categories"]

# 核心识别逻辑
def predict(image):
    model = load_model()
    class_names = get_class_names()
    img_tensor = preprocess_image(image)
    
    with torch.no_grad():
        outputs = model(img_tensor)
        _, indices = torch.topk(outputs, 3)
        results = [(class_names[i], torch.softmax(outputs, 1)[0][i].item()) for i in indices[0]]
    return results
pip install -r requirements.txt
# 界面交互
uploaded_file = st.file_uploader("上传一张图片", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=400)
    
    if st.button("开始识别", type="primary"):
        with st.spinner("AI正在识别..."):
            results = predict(image)
        
        st.subheader("识别结果（置信度Top3）")
        for name, prob in results:
            st.write(f"- {name}：{prob*100:.1f}%")
else:
    st.info("请上传一张包含常见物体的图片，AI会识别图片中的主体")
    # AI图片物体识别工具
《人工智能导论》期末作业 HW08

## 项目简介
基于深度学习预训练模型的图片物体识别工具，上传图片即可自动识别图中主体物体，输出置信度最高的3个类别。

## 运行方法
1. 安装依赖
```bash
pip install -r requirements.txt
streamlit run app.py
pip install -r requirements.txt
