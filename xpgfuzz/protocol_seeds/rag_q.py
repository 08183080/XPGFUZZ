import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# RAG应用中加载已存在的数据库
CHROMA_PERSIST_DIR = "./chroma_db_protocols"
CHROMA_COLLECTION_NAME = "protocols_wiki"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                              base_url="https://api.chatanywhere.tech/v1")

vectordb = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=embeddings,
    collection_name=CHROMA_COLLECTION_NAME
)

# --- 查询示例 ---

# 示例1: 跨协议的通用查询
query1 = "the cves of the protocols"
print(f"\n通用查询: '{query1}'")
# 这将在所有协议中寻找答案
results1 = vectordb.similarity_search(query1, k=4)
for doc in results1:
    print(f"  - [来源: {doc.metadata.get('protocol', 'N/A')}/{doc.metadata.get('source', 'N/A')}]")
    print(f"    '{doc.page_content[:100]}...'")

print("-" * 20)

# 示例2: 只针对特定协议 "protocol_A" 的查询 (假设你的一个子目录叫 protocol_A)
query2 = "What is the consensus mechanism of Exim-exim-DeepWiki"
target_protocol = "Exim-exim-DeepWiki" # 假设这是你的一个协议目录名
print(f"\n针对 '{target_protocol}' 的特定查询: '{query2}'")
# 使用 metadata filter (`where` clause) 来精确查找
results2 = vectordb.similarity_search(
    query2,
    k=4,
    filter={"protocol": target_protocol}
)
for doc in results2:
    print(f"  - [来源: {doc.metadata.get('protocol', 'N/A')}/{doc.metadata.get('source', 'N/A')}]")
    # 验证一下，来源应该都是 protocol_A
    assert doc.metadata.get('protocol') == target_protocol
    print(f"    '{doc.page_content[:100]}...'")