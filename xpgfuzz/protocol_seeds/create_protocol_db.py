import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# --- 配置 ---
# 加载 .env 文件中的环境变量
load_dotenv()

# 检查 OpenAI API Key 是否设置
if "OPENAI_API_KEY" not in os.environ:
    raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")

# 文件夹和数据库路径配置
BASE_DATA_PATH = "./protocol_deepwiiki" # 包含5个协议子目录的根文件夹
CHROMA_PERSIST_DIR = "./chroma_db_protocols" # ChromaDB持久化存储的路径
CHROMA_COLLECTION_NAME = "protocols_wiki" # 在Chroma中创建的Collection名称


def load_documents_from_protocol(protocol_path: str, protocol_name: str) -> list:
    """
    从指定的协议子目录加载所有Markdown文档，并附加元数据。

    Args:
        protocol_path (str): 协议子目录的路径.
        protocol_name (str): 协议的名称.

    Returns:
        list: 包含加载的文档对象和附加元数据的列表.
    """
    print(f"--- 正在从 '{protocol_path}' 加载协议 '{protocol_name}' 的文档... ---")
    # 使用UnstructuredMarkdownLoader来加载.md文件
    loader = DirectoryLoader(
        protocol_path,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        show_progress=True,
        use_multithreading=True
    )
    
    documents = loader.load()
    
    # 为每个文档添加 'protocol' 元数据，用于后续的过滤
    for doc in documents:
        # 在现有的metadata基础上增加或更新
        doc.metadata["protocol"] = protocol_name
        # 清理一下source路径，使其更可读
        doc.metadata["source"] = os.path.basename(doc.metadata["source"])
        
    print(f"成功加载 {len(documents)} 篇文档 for a protocol '{protocol_name}'.")
    return documents


def split_documents(all_documents: list) -> list:
    """
    将文档列表分割成更小的文本块 (chunks).

    Args:
        all_documents (list): 包含所有协议文档的列表.

    Returns:
        list: 分割后的文本块列表.
    """
    print("\n--- 正在将所有文档分割成文本块... ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个块的最大字符数
        chunk_overlap=200, # 块之间的重叠字符数
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"文档被成功分割成 {len(chunks)} 个文本块.")
    return chunks


def create_and_store_in_chroma(chunks: list, persist_directory: str, collection_name: str):
    """
    创建嵌入向量并将其存储到ChromaDB.

    Args:
        chunks (list): 待处理的文本块列表.
        persist_directory (str): ChromaDB持久化存储的路径.
        collection_name (str): ChromaDB中的Collection名称.
    """
    print("\n--- 正在创建嵌入向量并存入 ChromaDB... ---")
    
    # 清理旧的数据库（可选，如果想每次都重新生成）
    if os.path.exists(persist_directory):
        print(f"发现旧的数据库 '{persist_directory}', 正在删除...")
        shutil.rmtree(persist_directory)
        
    # 初始化OpenAI的嵌入模型
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                  base_url="https://api.chatanywhere.tech/v1") # 推荐使用新且性价比高的模型

    # 从文档创建Chroma数据库
    # Chroma.from_documents 会处理嵌入的创建和存储
    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    print(f"--- 成功! 向量库已创建并持久化存储在 '{persist_directory}' ---")
    print(f"总计 {vectordb._collection.count()} 个向量被添加到 '{collection_name}' collection.")


def main():
    """
    主函数，执行整个向量库构建流程.
    """
    print("=== 开始构建协议 DeepWiki 向量库 ===")
    
    # 检查根数据目录是否存在
    if not os.path.exists(BASE_DATA_PATH):
        print(f"错误: 数据目录 '{BASE_DATA_PATH}' 不存在。请检查路径。")
        return

    protocol_dirs = [d for d in os.listdir(BASE_DATA_PATH) if os.path.isdir(os.path.join(BASE_DATA_PATH, d))]
    
    if not protocol_dirs:
        print(f"错误: 在 '{BASE_DATA_PATH}' 中没有找到任何协议子目录。")
        return
        
    print(f"发现以下协议目录: {protocol_dirs}")
    
    all_protocol_documents = []
    
    # 步骤1: 遍历所有协议目录，加载文档并附加元数据
    for protocol_name in protocol_dirs:
        protocol_path = os.path.join(BASE_DATA_PATH, protocol_name)
        protocol_docs = load_documents_from_protocol(protocol_path, protocol_name)
        all_protocol_documents.extend(protocol_docs)
        
    if not all_protocol_documents:
        print("警告: 未加载到任何文档，程序即将退出。")
        return
        
    # 步骤2: 将所有文档统一分割
    all_chunks = split_documents(all_protocol_documents)
    
    # 步骤3: 创建嵌入并存入ChromaDB
    create_and_store_in_chroma(all_chunks, CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME)
    
    print("\n=== 向量库构建完成！ ===")


if __name__ == '__main__':
    main()