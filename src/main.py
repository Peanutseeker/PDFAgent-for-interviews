# src/main.py (使用 REST API 和代理端点最终版)
import os
import glob
from dotenv import load_dotenv
import time

# LangChain 核心组件
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# LangChain 集成库
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 全局配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
CONFIG = {
    "data_dir": os.path.join(PROJECT_ROOT, "data"),
    "output_dir": os.path.join(PROJECT_ROOT, "outputs"),
    "embedding_model_path": os.path.join(PROJECT_ROOT, "models", "m3e-base"),
    "llm_model": "gemini-1.5-flash-latest",
    "proxy_api_endpoint": "https://api.openai-proxy.org/google",  # 新增代理端点配置
    "chunk_size": 1500,
    "chunk_overlap": 200,
}


def load_documents_for_student(student_folder_path: str) -> tuple[list, str]:
    # ... 此函数内容保持不变 ...
    intro_text = ""
    documents = []
    txt_files = glob.glob(os.path.join(student_folder_path, '*.txt'))
    if txt_files:
        intro_path = txt_files[0]
        try:
            with open(intro_path, 'r', encoding='utf-8') as f:
                intro_text = f.read()
        except UnicodeDecodeError:
            print(f"  - 信息：UTF-8 解码失败，正在尝试使用 GBK 编码读取 {intro_path}...")
            try:
                with open(intro_path, 'r', encoding='gbk') as f:
                    intro_text = f.read()
            except Exception as e:
                print(f"  - 错误：使用 GBK 编码读取也失败: {e}")
        if intro_text:
            from langchain_core.documents import Document
            documents.append(Document(page_content=intro_text, metadata={"source": intro_path}))
    else:
        print(f"  - 警告：在 {student_folder_path} 中未找到 TXT 自我介绍文件。")
    pdf_files = glob.glob(os.path.join(student_folder_path, '*.pdf'))
    if pdf_files:
        pdf_path = pdf_files[0]
        try:
            pdf_loader = PyMuPDFLoader(pdf_path)
            documents.extend(pdf_loader.load())
        except Exception as e:
            print(f"  - 错误：加载 PDF 文件 {pdf_path} 失败: {e}")
    else:
        print(f"  - 警告：在 {student_folder_path} 中未找到 PDF 课题报告。")
    return documents, intro_text


def create_vector_store(documents: list):
    # ... 此函数内容保持不变 ...
    print("  - 正在分割文本...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"]
    )
    docs = text_splitter.split_documents(documents)
    print(f"  - 正在从本地路径 '{CONFIG['embedding_model_path']}' 加载嵌入模型...")
    embeddings = HuggingFaceEmbeddings(model_name=CONFIG["embedding_model_path"])
    print("  - 正在构建 FAISS 向量索引...")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store


def generate_interview_questions(vector_store, student_id: str, intro_text: str) -> str:
    """使用现代化的 LCEL RAG 链生成面试问题清单。"""
    print("  - 准备调用大语言模型生成问题...")
    retriever = vector_store.as_retriever(search_kwargs={'k': 5})

    # --- 【核心修改部分】 ---
    # 1. 定义要传递给 Google SDK 的客户端选项
    client_options = {"api_endpoint": CONFIG["proxy_api_endpoint"]}

    # 2. 在初始化 LLM 时，传入 transport 和 client_options 参数
    llm = ChatGoogleGenerativeAI(
        model=CONFIG["llm_model"],
        temperature=0.7,
        transport="rest",  # 强制使用 HTTPS REST API
        client_options=client_options  # 指定 API 代理端点
    )
    # --- 修改结束 ---

    template = """
    你是一位资深的大学“综合评价”面试官，你的任务是为面试助理（学长学姐）生成一份针对具体学生的、有深度的面试问题清单。

    **学生信息:**
    - 学号: {student_id}
    - 学生的自我介绍: 
    ---
    {intro_text}
    ---

    **根据提问从该学生课题报告中检索出的相关核心内容:**
    ---
    {context}
    ---

    **你的任务要求:**
    1.  **生成 5 个面试问题**：严格基于以上提供的“自我介绍”和“课题报告核心内容”。问题必须具有深度和启发性，能考察学生的学术潜力、思维能力和个人特质。
    2.  **问题类型需多样化**：至少覆盖以下方面：针对自我介绍中的动机和经历提问、针对课题报告中的研究方法/数据/结论提问、考察批判性思维或对课题局限性认识的问题。
    3.  **提供评价要点**：在每个问题下方，清晰地列出 "评价要点"，指导面试官应从学生的回答中重点考察哪些方面（例如：逻辑的严密性、对专业的思考深度、求真精神、表达清晰度等）。
    4.  **格式要求**：请使用 Markdown 格式进行输出，以学号作为主标题。

    请开始生成你的面试问题清单。
    """

    prompt = ChatPromptTemplate.from_template(template)

    # 将文档列表转换为单个字符串
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 定义 RAG 链
    rag_chain = (
            {
                "context": (lambda x: x["intro_text"]) | retriever | RunnableLambda(format_docs),  #注意，接受的是字典，要变换成字符串
                "student_id": RunnablePassthrough(),
                "intro_text": RunnablePassthrough()
            }
            | prompt
            | llm
            | StrOutputParser()
    )

    print("  - 模型正在生成内容，请稍候...")
    # 在 invoke 时，我们只需要传递原始的、会变化的值
    # 因为 context 是由 retriever 自动生成的
    result = rag_chain.invoke({
        "student_id": student_id,
        "intro_text": intro_text
    })

    return result


def process_all_students():
    # ... 此函数内容保持不变 ...
    print("--- 启动综评面试 Agent ---")
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    student_folders = [entry.path for entry in os.scandir(CONFIG["data_dir"]) if entry.is_dir()]

    if not student_folders:
        print(f"错误：在 '{CONFIG['data_dir']}' 目录下未找到任何学生文件夹。")
        return

    for student_folder_path in student_folders:
        student_id = os.path.basename(student_folder_path)
        print(f"\n--- 正在处理学号: {student_id} ---")

        try:
            documents, intro_text = load_documents_for_student(student_folder_path)
            if not documents:
                print(f"  - 跳过学号 {student_id}，因为未能加载到任何有效文档。")
                continue

            vector_store = create_vector_store(documents)
            generated_content = generate_interview_questions(vector_store, student_id, intro_text)

            output_filename = os.path.join(CONFIG["output_dir"], f"{student_id}_面试问题.md")
            with open(output_filename, 'w', encoding='utf-8') as f:
                f.write(generated_content)
            print(f"  - ✅ 成功！面试问题清单已保存到: {output_filename}")

        except Exception as e:
            print(f"  - ❌ 处理学号 {student_id} 时发生严重错误: {e}")

        time.sleep(2)

    print("\n--- 所有学生处理完毕 ---")


if __name__ == '__main__':
    load_dotenv()
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("错误：请在 .env 文件中设置您的 GOOGLE_API_KEY")
    process_all_students()