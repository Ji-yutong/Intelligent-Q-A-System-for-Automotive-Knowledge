import streamlit as st
from PIL import Image
import json
import os

# 从配置文件加载设置
with open('config.json', encoding='utf-8') as config_file:
    config = json.load(config_file)

@st.cache_resource
def get_agent():
    from agent_module import Agent
    return Agent()

# 初始化 agent
agent = get_agent()

def main():
    st.title("汽车知识问答系统")

    # 显示对话历史
    if 'history' not in st.session_state:
        st.session_state.history = []

    # 用户输入
    user_input = st.text_input("请输入你的问题", "")

    # 上传图片
    image = st.file_uploader("上传图片", type=["jpg", "jpeg", "png"])

    if st.button("提交"):
        if user_input or image:
            image_path = None
            if image:
                image_path = os.path.join('static', 'uploads', image.name)
                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                with open(image_path, "wb") as f:
                    f.write(image.getbuffer())

            response = agent.handle_query(user_input, image_path)
            st.write("系统回复:", response)
            st.session_state.history.append({"user": user_input, "response": response})
        else:
            st.warning("请提供问题或上传图片。")

    # 显示对话历史
    if st.session_state.history:
        for entry in st.session_state.history:
            st.write(f"**用户:** {entry['user']}")
            st.write(f"**系统:** {entry['response']}")

if __name__ == "__main__":
    main()
