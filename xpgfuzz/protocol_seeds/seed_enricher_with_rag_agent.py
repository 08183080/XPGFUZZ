import os
import shutil
import random
import argparse
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools.retriever import create_retriever_tool

# ======================================================================
# 模块一: 配置与数据结构
# ======================================================================
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")

@dataclass
class Protocol:
    """封装协议信息的数据类。"""
    name: str
    commands: set

# ======================================================================
# 模块二: 文件与RAG工具
# ======================================================================

def setup_directories(input_dir: str, output_dir: str):
    print(f"--- 正在准备目录 '{input_dir}' -> '{output_dir}' ---")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录 '{input_dir}' 不存在。")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def read_seed_file(filepath: str) -> str:
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"读取文件 '{filepath}' 时出错: {e}")
        return None

def save_enriched_seed(output_dir: str, original_filename: str, content: str, variation_index: int):
    base_name, ext = os.path.splitext(original_filename)
    new_filename = f"enriched_{base_name}_{variation_index}{ext}"
    output_path = os.path.join(output_dir, new_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"  保存文件 '{new_filename}' 时出错: {e}")

def setup_retriever(db_path: str, collection_name: str):
    print(f"--- 正在从 '{db_path}' (collection: '{collection_name}') 加载向量数据库... ---")
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ChromaDB 目录不存在: {db_path}。")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small",
                                  base_url='https://api.chatanywhere.tech/v1')
    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embeddings,
        collection_name=collection_name
    )
    return vectordb.as_retriever(search_kwargs={'k': 5})

# ======================================================================
# 模块三: 核心AI扩充逻辑 (RAG + ReAct)
# ======================================================================

def analyze_missing_commands(sequence: str, all_commands: set) -> set:
    sequence_lower = sequence.lower()
    present_commands = {cmd for cmd in all_commands if cmd.lower() in sequence_lower}
    return all_commands - present_commands

def enrich_sequence_with_react(sequence: str, protocol: Protocol, retriever, temperature: float, max_enrich_types: int, model: str) -> str:
    """
    使用 RAG + ReAct 智能体丰富单条序列。
    """
    missing_commands = analyze_missing_commands(sequence, protocol.commands)
    if not missing_commands:
        print("  序列已包含所有指定命令，无需扩充。")
        return None

    if max_enrich_types > 0 and len(missing_commands) > max_enrich_types:
        commands_to_add = set(random.sample(list(missing_commands), max_enrich_types))
    else:
        commands_to_add = missing_commands

    # --- RAG检索文档 ---
    print(f"  -> RAG: 正在为命令 {commands_to_add} 检索上下文...")
    context_str = ""
    for cmd in commands_to_add:
        query = f"How to use the {cmd} command in {protocol.name} protocol, including syntax and examples."
        docs = retriever.invoke(query)
        context_str += f"\n--- Documentation for {cmd} ---\n"
        context_str += "\n".join([doc.page_content for doc in docs])
        context_str += "\n"

    # --- ReAct智能体 ---
    retriever_tool = create_retriever_tool(
        retriever,
        "search_protocol_docs",
        "Search technical documentation to validate command usage and insertion points."
    )
    tools = [retriever_tool]

    react_prompt = f"""
You are an expert in the {protocol.name} protocol.
You need to intelligently complete an incomplete sequence using authoritative documentation.

Authoritative Context from RAG:
{context_str}

Missing Commands to Insert:
{', '.join(sorted(commands_to_add))}

Rules:
1. Strictly follow the syntax and examples from the context.
2. Only output the raw message sequence.
3. Ensure logical conversation flow for the {protocol.name} protocol.
4. Use ReAct cycle (Thought -> Action -> Observation) to iteratively decide where to insert commands.

Initial Sequence:
{sequence}

Completed Sequence:
"""

    llm = ChatOpenAI(model=model, temperature=temperature, base_url='https://api.chatanywhere.tech/v1')
    agent = create_react_agent(llm, tools, prompt=react_prompt)
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True)

    enriched_sequence = agent_executor.invoke({})['output']
    return enriched_sequence

# ======================================================================
# 模块四: 主流程控制器
# ======================================================================

def main_process(args):
    command_set = {cmd.strip() for cmd in args.commands.split(',') if cmd.strip()}
    protocol = Protocol(name=args.protocol_name, commands=command_set)

    setup_directories(args.input_dir, args.output_dir)
    retriever = setup_retriever(args.db_dir, args.collection_name)
    
    all_seed_files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    if not all_seed_files:
        print(f"警告: 输入目录 '{args.input_dir}' 中没有找到种子文件。")
        return

    if args.max_corpus_size > 0 and len(all_seed_files) > args.max_corpus_size:
        seed_files_to_process = random.sample(all_seed_files, args.max_corpus_size)
    else:
        seed_files_to_process = all_seed_files

    print(f"\n--- 协议 '{protocol.name}' | 开始批量扩充 {len(seed_files_to_process)} 个种子文件... ---")
    
    total_generated = 0
    for filename in seed_files_to_process:
        filepath = os.path.join(args.input_dir, filename)
        print(f"\n处理原始种子: {filename}")
        original_content = read_seed_file(filepath)
        if not original_content: continue
        
        generated_for_this_seed = 0
        for i in range(args.variations):
            temp = random.uniform(0.3, 0.8)
            print(f"  -> 生成变体 {i+1}/{args.variations} (temperature={temp:.2f})...")

            enriched_content = enrich_sequence_with_react(
                original_content, protocol, retriever, temp, args.max_enrich_types, args.model
            )
            
            if enriched_content and enriched_content.strip() != original_content.strip():
                save_enriched_seed(args.output_dir, filename, enriched_content, i + 1)
                generated_for_this_seed += 1
        
        if generated_for_this_seed > 0:
            print(f"  成功为 {filename} 生成了 {generated_for_this_seed} 个新种子。")
        total_generated += generated_for_this_seed

    print(f"\n🎉 批量扩充任务完成！总计生成了 {total_generated} 个新种子，已保存至 '{args.output_dir}' 目录。")

# ======================================================================
# 脚本入口
# ======================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="[RAG+ReAct增强版]通用协议种子丰富器",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--db-dir", default="./chroma_db_protocols", type=str, help="ChromaDB持久化存储路径。")
    parser.add_argument("--collection-name", required=True, type=str, help="ChromaDB Collection名称。")
    parser.add_argument("--protocol-name", required=True, type=str, help="协议名称，例如 'Exim SMTP'。")
    parser.add_argument("--commands", required=True, type=str, help="协议命令全集，用逗号分隔。")
    parser.add_argument("--input-dir", default="in", type=str, help="输入种子文件目录。")
    parser.add_argument("--output-dir", default="out", type=str, help="输出目录。")
    parser.add_argument("--variations", default=5, type=int, help="每个种子生成变体数量。")
    parser.add_argument("--model", default="gpt-4-turbo", type=str, help="使用的OpenAI模型。")
    parser.add_argument("--max-enrich-types", default=2, type=int, help="单次丰富操作最多添加命令类型数。")
    parser.add_argument("--max-corpus-size", default=10, type=int, help="最多处理的种子文件数量。")

    args = parser.parse_args()
    main_process(args)
