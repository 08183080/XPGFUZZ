import os
import shutil
import random
import argparse
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# ==============================================================================
# 模块一: 配置与数据结构
# ==============================================================================
# 加载环境变量
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")

@dataclass
class Protocol:
    """一个用于封装协议信息的数据类，实现解耦。"""
    name: str
    commands: set

# ==============================================================================
# 模块二: 文件工具 (无变化)
# ==============================================================================

def setup_directories(input_dir: str, output_dir: str):
    """检查输入目录是否存在，并创建或清空输出目录。"""
    print(f"--- 正在准备目录 '{input_dir}' -> '{output_dir}' ---")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录 '{input_dir}' 不存在。")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

def read_seed_file(filepath: str) -> str:
    """读取种子文件的内容。"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"读取文件 '{filepath}' 时出错: {e}")
        return None

def save_enriched_seed(output_dir: str, original_filename: str, content: str, variation_index: int):
    """将增强后的种子保存到输出目录。"""
    base_name, ext = os.path.splitext(original_filename)
    new_filename = f"enriched_{base_name}_{variation_index}{ext}"
    output_path = os.path.join(output_dir, new_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"  保存文件 '{new_filename}' 时出错: {e}")

# ==============================================================================
# 模块三: 核心AI扩充逻辑 (已解耦)
# ==============================================================================

def analyze_missing_commands(sequence: str, all_commands: set) -> set:
    """分析序列，返回缺失的命令集。"""
    sequence_lower = sequence.lower()
    present_commands = {cmd for cmd in all_commands if cmd.lower() in sequence_lower}
    return all_commands - present_commands

def enrich_sequence(sequence: str, protocol: Protocol, temperature: float, max_enrich_types: int, model: str) -> str:
    """
    使用LLM增量式扩充单个序列（协议通用版本）。
    """
    missing_commands = analyze_missing_commands(sequence, protocol.commands)
    
    if not missing_commands:
        print("  序列已包含所有指定命令，无需扩充。")
        return None

    if max_enrich_types > 0 and len(missing_commands) > max_enrich_types:
        commands_to_add = set(random.sample(list(missing_commands), max_enrich_types))
    else:
        commands_to_add = missing_commands

    prompt_template = """
You are an expert in the {protocol_name} protocol. Your task is to intelligently complete an incomplete message sequence for fuzzing purposes.

**Analysis:**
The provided sequence is missing several commands. We will focus on inserting the following commands in this step: {commands_to_add_str}

**Task:**
Rewrite the sequence, inserting the specified missing commands in logically correct positions to form a valid and more comprehensive conversation.

**Rules:**
1.  The final output must be ONLY the raw message sequence.
2.  Do not add any explanations or comments.
3.  Ensure the conversation flow is logical for the {protocol_name} protocol.
4.  Use plausible arguments for commands.

**Incomplete Sequence:**
{original_sequence}

**Completed Sequence:**
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model=model, temperature=temperature, base_url="https://api.chatanywhere.tech/v1")
    enrichment_chain = prompt | llm | StrOutputParser()
    
    enriched_sequence = enrichment_chain.invoke({
        "protocol_name": protocol.name,
        "commands_to_add_str": ", ".join(sorted(list(commands_to_add))),
        "original_sequence": sequence
    })
    
    return enriched_sequence

# ==============================================================================
# 模块四: 主流程控制器 (已解耦)
# ==============================================================================

def main_process(args):
    """
    自动化处理整个目录的种子文件的主流程。
    """
    # 从命令行参数构造协议对象
    command_set = {cmd.strip() for cmd in args.commands.split(',') if cmd.strip()}
    protocol = Protocol(name=args.protocol_name, commands=command_set)

    setup_directories(args.input_dir, args.output_dir)
    
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
            
            enriched_content = enrich_sequence(
                original_content, protocol, temp, args.max_enrich_types, args.model
            )
            
            if enriched_content and enriched_content.strip() != original_content.strip():
                save_enriched_seed(args.output_dir, filename, enriched_content, i + 1)
                generated_for_this_seed += 1
        
        if generated_for_this_seed > 0:
            print(f"  成功为 {filename} 生成了 {generated_for_this_seed} 个新种子。")
        total_generated += generated_for_this_seed

    print(f"\n🎉 批量扩充任务完成！总计生成了 {total_generated} 个新种子，已保存至 '{args.output_dir}' 目录。")

# ==============================================================================
# 脚本入口: 使用 argparse 进行命令行解析
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="通用协议种子丰富器 (Generic Protocol Seed Enricher)",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--protocol-name",
        required=True,
        type=str,
        help="协议的名称 (例如 'Exim SMTP', 'FTP')，将用于AI Prompt。"
    )
    parser.add_argument(
        "--commands",
        required=True,
        type=str,
        help="该协议的命令全集，用逗号分隔 (例如 'HELO,EHLO,MAIL FROM,...')。"
    )
    parser.add_argument(
        "--input-dir",
        default="in",
        type=str,
        help="包含原始种子文件的输入目录 (默认: 'in')。"
    )
    parser.add_argument(
        "--output-dir",
        default="out",
        type=str,
        help="用于存放生成种子的输出目录 (默认: 'out')。"
    )
    parser.add_argument(
        "--variations",
        default=20,
        type=int,
        help="每个原始种子要生成的变体数量 (默认: 20)。"
    )
    parser.add_argument(
        "--model",
        default="gpt-4-turbo",
        type=str,
        help="要使用的OpenAI模型 (默认: 'gpt-4-turbo')。"
    )
    parser.add_argument(
        "--max-enrich-types",
        default=2,
        type=int,
        help="单次丰富操作中最多添加的命令类型数量 (0为无限制, 默认: 2)。"
    )
    parser.add_argument(
        "--max-corpus-size",
        default=10,
        type=int,
        help="从输入目录中最多处理的种子文件数量 (0为无限制, 默认: 10)。"
    )

    args = parser.parse_args()
    main_process(args)
