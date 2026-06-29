# 智能文档问答助手 完整源码
import gradio as gr
import PyPDF2
import faiss
import numpy as np
import requests
from sentence_transformers import SentenceTransformer

# 1. 初始化全局模型与向量库
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)
text_store = []  # 存储原始文本块
API_KEY = "替换为你的大模型API密钥"
API_URL = "https://api.deepseek.com/v1/chat/completions"

# 2. 文件读取函数
def load_file(file_path):
    content = ""
    if file_path.endswith(".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    elif file_path.endswith(".pdf"):
        pdf_reader = PyPDF2.PdfReader(file_path)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                content += page_text
    # 文本分段
    chunks = []
    chunk_len = 200
    for i in range(0, len(content), chunk_len):
        chunks.append(content[i:i+chunk_len])
    return chunks

# 3. 向量化并存入向量库
def add_to_vector_db(chunks):
    global index, text_store
    embeds = embed_model.encode(chunks)
    embeds = np.array(embeds, dtype=np.float32)
    index.add(embeds)
    text_store.extend(chunks)

# 4. 检索相似文本
def search_text(query, top_k=3):
    query_embed = embed_model.encode([query])
    query_embed = np.array(query_embed, dtype=np.float32)
    dists, ids = index.search(query_embed, top_k)
    res_text = ""
    for idx in ids[0]:
        if idx < len(text_store):
            res_text += text_store[idx] + "\n"
    return res_text

# 5. 调用大模型API生成回答
def llm_answer(context, question):
    prompt = f"""参考文档内容：{context}
请先生成全文简短摘要，再回答用户问题。禁止编造文档没有的信息。
用户问题：{question}"""
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    data = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3
    }
    resp = requests.post(API_URL, headers=headers, json=data)
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    else:
        return "API调用失败，请检查密钥与网络"

# 6. 页面交互逻辑
def upload_file(file):
    global index, text_store
    index.reset()
    text_store.clear()
    chunks = load_file(file.name)
    add_to_vector_db(chunks)
    return "文档上传解析完成，可以开始提问"

def chat_func(question):
    context = search_text(question)
    ans = llm_answer(context, question)
    return ans

# 7. 搭建Gradio网页界面
with gr.Blocks(title="智能文档摘要问答助手") as demo:
    gr.Markdown("# AI智能文档问答工具")
    file_input = gr.File(label="上传PDF/TXT文档")
    upload_msg = gr.Textbox(label="上传状态")
    question_input = gr.Textbox(label="输入你的问题")
    answer_out = gr.Textbox(label="AI摘要与回答", lines=10)
    file_input.upload(upload_file, inputs=[file_input], outputs=[upload_msg])
    question_input.submit(chat_func, inputs=[question_input], outputs=[answer_out])

if __name__ == "__main__":
    demo.launch()