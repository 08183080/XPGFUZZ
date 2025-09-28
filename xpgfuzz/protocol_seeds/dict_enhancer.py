import os
import json
import argparse
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# --- 1. 配置与初始化 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")

# D:\prptocol_seeds\protocol_deepwiiki
smtp_filter = {"protocol": "Exim-exim-DeepWiki"}
proftp_filter = {"protocol": "proftpd-proftpd-DeepWiki"}


def setup_retriever(db_path: str, collection_name: str, filter: str):
    """初始化并返回一个RAG检索器。"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ChromaDB 目录不存在: {db_path}。")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                  base_url='https://api.chatanywhere.tech/v1')
    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    return vectordb.as_retriever(search_kwargs={'k': 15, 'filter': {'protocol':filter}})

# --- 2. 核心AI提取函数 ---
def generate_dictionary_tokens(protocol_name: str, retriever, model: str) -> dict:
    """
    一个简洁的函数，执行RAG提取并返回一个令牌字典。
    """
    # 步骤 A: 一次性检索获取全面的上下文
    logging.info(f"正在为协议 '{protocol_name}' 检索全面的上下文...")
    comprehensive_query = f"all commands, headers, parameters, status codes, and typical example values for the {protocol_name} protocol"
    context_docs = retriever.invoke(comprehensive_query)
    context = "\n\n".join([doc.page_content for doc in context_docs])
    logging.info(f"上下文检索完成，总长度约为 {len(context)} 字符。")

    # 步骤 B: 使用一个简化的、全能的Prompt进行提取
    prompt_template = """
You are a security researcher building a fuzzing dictionary for the {protocol_name} protocol.
Based on the provided context, extract all relevant tokens.
根据上下文提取处命令、头字段、状态码和常见参数值等网络通信令牌。注意不要提取函数名、变量名或源代码相关的内容。


在提取完这些信息后，你再利用大模型内部的知识提取补全遗漏的令牌。

**Task:**
Extract a comprehensive list of protocol keywords. This includes:
- Commands (e.g., USER, RETR)
- Header fields (e.g., Content-Type)
- Status codes (e.g., 200, 404)
- Common parameter values or magic strings (e.g., "application/sdp", "RTSP/1.0")

**Instructions:**
- Your final output must be a single, valid JSON object.
- The JSON keys should be the extracted tokens.
- The JSON values should be a typical example value found in the documentation. If no example is found, use an empty string.
- **Crucial Rule: Only extract tokens that are sent over the network between a client and a server. Do NOT extract internal function names, variable names, or source code artifacts (e.g., do not extract anything like 'smtp_connect' or 'ACL_WHERE_DATA').**

**Provided Documentation Context:**
---
{context}
---

**JSON Output:**
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model=model, temperature=0.1,
                     base_url='https://api.chatanywhere.tech/v1')
    chain = prompt | llm | StrOutputParser()

    logging.info("正在调用AI进行令牌提取...")
    response_str = chain.invoke({
        "protocol_name": protocol_name,
        "context": context
    })

    # 步骤 C: 简单的JSON解析
    try:
        tokens = json.loads(response_str)
        logging.info(f"成功提取 {len(tokens)} 个令牌。")
        return tokens
    except json.JSONDecodeError:
        logging.error("AI未能返回有效的JSON格式，无法生成字典。")
        logging.debug(f"Received malformed string: {response_str}")
        return {}

# --- 3. 字典文件写入函数 ---
def format_and_save_dictionary(token_dict: dict, protocol_name: str, output_file: str):
    """
    将提取的令牌字典格式化为 afl-fuzz (-x) 可用的字典文件。
    """
    logging.info(f"正在将提取的令牌写入到 AFL 字典文件: {output_file}...")
    count = 0
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"# Auto-generated dictionary for {protocol_name} via AI\n\n")
        
        for token, example in token_dict.items():
            clean_token = token.strip().replace('"', '')
            if not clean_token: continue

            # 格式1: 纯关键字
            f.write(f'"{clean_token}"\n')
            
            # 格式2: 关键字=值
            if isinstance(example, str) and example:
                escaped_example = example.replace('\\', '\\\\').replace('"', '\\"')
                f.write(f'{clean_token}="{escaped_example}"\n')
            
            count += 1
            
    logging.info(f"成功写入 {count} 个独特的令牌到 {output_file}。")

# --- 4. 脚本主入口 ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="[简化版] 使用RAG和AI为任何协议自动生成Fuzzing字典。",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("-p", "--protocol-name", required=True, type=str, help="目标协议的名称 (例如 'ProFTPD', 'Exim SMTP')。")
    parser.add_argument("-d", "--db-dir", default="./chroma_db_protocols", type=str, help="ChromaDB持久化存储的路径。")
    parser.add_argument("-c", "--collection-name", default="protocols_wiki", type=str, help="ChromaDB中的Collection名称。")
    parser.add_argument("-o", "--output-file", type=str, help="输出的字典文件名。如果未提供，将根据协议名称自动生成。")
    parser.add_argument("-m", "--model", default="gpt-4-turbo", type=str, help="用于提取的OpenAI模型。")
    parser.add_argument("-f", "--filter", default="none", type=str, help="协议选择过滤")

    args = parser.parse_args()

    # 自动生成输出文件名
    if not args.output_file:
        safe_protocol_name = "".join(c for c in args.protocol_name if c.isalnum() or c in (' ', '_')).rstrip()
        args.output_file = f"{safe_protocol_name.lower().replace(' ', '_')}.dict"

    try:
        # 执行流程
        retriever = setup_retriever(args.db_dir, args.collection_name, args.filter)
        all_tokens = generate_dictionary_tokens(args.protocol_name, retriever, args.model)
        
        if all_tokens:
            format_and_save_dictionary(all_tokens, args.protocol_name, args.output_file)
        else:
            logging.warning("未能提取任何令牌，字典文件未生成。")
        
        logging.info("任务完成。")

    except Exception as e:
        logging.error(f"发生致命错误: {e}")

# python .\dict_enhancer.py -p SMTP/Exim -c protocols_wiki -o smtp_x.dict -f Exim-exim-DeepWiki    
# python .\dict_enhancer.py -p ProFTPD -c protocols_wiki -o proftpd.dict -f proftpd-proftpd-DeepWiki
# python .\dict_enhancer.py -p RTSP -c protocols_wiki -o rtsp_x.dict -f rgaufman-live555-DeepWiki