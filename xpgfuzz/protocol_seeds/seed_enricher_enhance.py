import os
import shutil
import random
import argparse
from dataclasses import dataclass
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# ====================================================================
# 模块一: 配置与数据结构
# ====================================================================
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")

@dataclass
class Protocol:
    name: str
    commands: set

STRATEGIES = ["normal_completion", "boundary_conditions", "randomized_order", "special_characters"]

# ====================================================================
# 模块二: 文件工具
# ====================================================================
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

def save_enriched_seed(output_dir: str, strategy: str, original_filename: str, content: str, variation_index: int):
    base_name, ext = os.path.splitext(original_filename)
    # 在文件名中加策略前缀
    new_filename = f"enriched_{strategy}_{base_name}_{variation_index}{ext}"
    output_path = os.path.join(output_dir, new_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"保存文件 '{new_filename}' 时出错: {e}")

# ====================================================================
# 模块三: 核心AI扩充逻辑
# ====================================================================
def analyze_missing_commands(sequence: str, all_commands: set) -> set:
    sequence_lower = sequence.lower()
    present_commands = {cmd for cmd in all_commands if cmd.lower() in sequence_lower}
    return all_commands - present_commands

def enrich_sequence(sequence: str, protocol: Protocol, temperature: float, max_enrich_types: int, model: str, strategy: str) -> str:
    missing_commands = analyze_missing_commands(sequence, protocol.commands)
    if not missing_commands:
        print("  序列已包含所有命令，无需扩充。")
        return None

    if max_enrich_types > 0 and len(missing_commands) > max_enrich_types:
        # 可重复插入同一命令，增加 fuzzing 覆盖
        commands_to_add = random.choices(list(missing_commands), k=max_enrich_types)
    else:
        commands_to_add = list(missing_commands)

    prompt_template = f"""
You are an expert in the {protocol.name} protocol. Your task is to intelligently complete an incomplete message sequence for fuzzing.

**Analysis:**
The provided sequence is missing several commands. Insert the following commands: {', '.join(commands_to_add)}

**Strategy:**
{strategy}
- normal_completion: logical completion
- boundary_conditions: very long addresses, whitespace, repeated commands
- randomized_order: shuffle command order
- special_characters: include non-ASCII characters, unusual symbols

**Rules:**
1. Only output the raw message sequence.
2. Do not add explanations or comments.
3. Ensure logical flow for {protocol.name}.
4. Use plausible arguments, include edge cases if strategy specifies.

**Incomplete Sequence:**
{sequence}

**Completed Sequence:**
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model=model, temperature=temperature, base_url="https://api.chatanywhere.tech/v1")
    enrichment_chain = prompt | llm | StrOutputParser()
    enriched_sequence = enrichment_chain.invoke({})
    return enriched_sequence

# ====================================================================
# 模块四: 主流程控制器
# ====================================================================
def main_process(args):
    command_set = {cmd.strip() for cmd in args.commands.split(',') if cmd.strip()}
    protocol = Protocol(name=args.protocol_name, commands=command_set)

    setup_directories(args.input_dir, args.output_dir)

    all_seed_files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    if not all_seed_files:
        print(f"警告: 输入目录 '{args.input_dir}' 没有种子文件。")
        return

    if args.max_corpus_size > 0 and len(all_seed_files) > args.max_corpus_size:
        seed_files_to_process = random.sample(all_seed_files, args.max_corpus_size)
    else:
        seed_files_to_process = all_seed_files

    print(f"\n--- 协议 '{protocol.name}' | 批量扩充 {len(seed_files_to_process)} 个种子文件 ---")

    total_generated = 0
    for filename in seed_files_to_process:
        filepath = os.path.join(args.input_dir, filename)
        original_content = read_seed_file(filepath)
        if not original_content: continue

        for i in range(args.variations):
            strategy = random.choice(STRATEGIES)
            # 根据策略调整温度
            if strategy == "boundary_conditions": temp = random.uniform(0.6, 0.9)
            elif strategy == "randomized_order": temp = random.uniform(0.5, 0.8)
            elif strategy == "special_characters": temp = random.uniform(0.6, 0.85)
            else: temp = random.uniform(0.3, 0.5)
            print(f"  -> {filename} 生成变体 {i+1}/{args.variations} | 策略: {strategy} | temperature={temp:.2f}")

            enriched_content = enrich_sequence(
                original_content, protocol, temp, args.max_enrich_types, args.model, strategy
            )
            if enriched_content and enriched_content.strip() != original_content.strip():
                save_enriched_seed(args.output_dir, strategy, filename, enriched_content, i+1)
                total_generated += 1

    print(f"\n🎉 批量扩充完成！总生成种子: {total_generated}，已保存至 '{args.output_dir}' 目录。")

# ====================================================================
# 脚本入口
# ====================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="通用协议种子丰富器 (Generic Protocol Seed Enricher)")
    parser.add_argument("--protocol-name", required=True, type=str, help="协议名称，例如 'Exim SMTP'")
    parser.add_argument("--commands", required=True, type=str, help="协议命令全集，逗号分隔")
    parser.add_argument("--input-dir", default="in", type=str, help="原始种子目录")
    parser.add_argument("--output-dir", default="out", type=str, help="生成种子输出目录")
    parser.add_argument("--variations", default=20, type=int, help="每个种子生成变体数量")
    parser.add_argument("--model", default="gpt-4-turbo", type=str, help="OpenAI模型")
    parser.add_argument("--max-enrich-types", default=3, type=int, help="单次最多添加命令类型数")
    parser.add_argument("--max-corpus-size", default=10, type=int, help="最多处理种子文件数量")
    args = parser.parse_args()

    main_process(args)
