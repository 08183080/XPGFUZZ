import os
import shutil
import random
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

# ==============================================================================
# 模块一: 配置 (Configuration) - 更新为Exim特定命令
# ==============================================================================
# 加载环境变量
load_dotenv()
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("请在 .env 文件中设置 OPENAI_API_KEY")

# --- 可调整的参数 ---
INPUT_DIR = "in-smtp"
OUTPUT_DIR = "in-smtp-x"
NUM_VARIATIONS_PER_SEED = 44
LLM_MODEL = "gpt-4-turbo"

# --- ChatAFL策略参数 ---
MAX_ENRICHMENT_MESSAGE_TYPES = 2
MAX_ENRICHMENT_CORPUS_SIZE = 10

# ===============================================================
# 更新：使用您提供的Exim实现的完整命令和扩展列表
# ===============================================================
EXIM_SMTP_COMMANDS = {
    # 基本命令
    "HELO",
    "EHLO",
    "MAIL FROM",
    "RCPT TO",
    "DATA",
    "QUIT",
    "RSET",
    "NOOP",
    # 扩展命令
    "VRFY",
    "ETRN",
    "BDAT",
    # 支持的扩展 (作为命令使用)
    "CHUNKING",
    "DSN",
    "PIPELINING",
    "SIZE",
    "STARTTLS"
}
# ===============================================================

# ==============================================================================
# 模块二: 文件工具 (File Utilities)
# ==============================================================================

def setup_directories(input_dir: str, output_dir: str):
    """检查输入目录是否存在，并创建或清空输出目录。"""
    print("--- 正在准备目录... ---")
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"输入目录 '{input_dir}' 不存在。请创建并放入种子文件。")
    if os.path.exists(output_dir):
        print(f"输出目录 '{output_dir}' 已存在，正在清空...")
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    print(f"成功创建空的输出目录 '{output_dir}'。")

def read_seed_file(filepath: str) -> str:
    """读取种子文件的内容。"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"读取文件 '{filepath}' 时出错: {e}")
        return None

def save_enriched_seed(output_dir: str, original_filename: str, content: str, variation_index: int):
    """将增强后的种子保存到输出目录，使用 "enriched_" 前缀。"""
    base_name, ext = os.path.splitext(original_filename)
    new_filename = f"enriched_{base_name}_{variation_index}{ext}"
    output_path = os.path.join(output_dir, new_filename)
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    except Exception as e:
        print(f"  保存文件 '{new_filename}' 时出错: {e}")

# ==============================================================================
# 模块三: 核心AI扩充逻辑 (Core AI Enrichment)
# ==============================================================================

def analyze_missing_commands(sequence: str, all_commands: set) -> set:
    """分析序列，返回缺失的Exim命令集。"""
    # 检查时忽略大小写以提高匹配率
    sequence_lower = sequence.lower()
    present_commands = {cmd for cmd in all_commands if cmd.lower() in sequence_lower}
    return all_commands - present_commands

def enrich_exim_sequence(sequence: str, temperature: float) -> str:
    """
    使用LLM增量式扩充单个Exim SMTP序列。
    """
    missing_commands = analyze_missing_commands(sequence, EXIM_SMTP_COMMANDS)
    
    if not missing_commands:
        print("  序列已包含所有已知Exim命令，无需扩充。")
        return None

    if MAX_ENRICHMENT_MESSAGE_TYPES > 0 and len(missing_commands) > MAX_ENRICHMENT_MESSAGE_TYPES:
        num_to_add = MAX_ENRICHMENT_MESSAGE_TYPES
        commands_to_add = set(random.sample(list(missing_commands), num_to_add))
    else:
        commands_to_add = missing_commands

    prompt_template = """
You are an expert in the Exim SMTP server protocol. Your task is to intelligently complete an incomplete SMTP message sequence, making it more complex for fuzzing purposes.

**Analysis:**
The provided sequence is missing several Exim-supported commands. We will focus on inserting the following commands in this step: {commands_to_add_str}

**Task:**
Rewrite the sequence, inserting the specified missing commands in logically correct positions to form a valid and more comprehensive Exim SMTP conversation.

**Rules:**
1.  The final output must be ONLY the raw SMTP sequence.
2.  Do not add any explanations or comments.
3.  Ensure the conversation flow is logical for an Exim server.
4.  Use plausible arguments for commands (e.g., `test@example.com`, `STARTTLS`).

**Incomplete Sequence:**
{original_sequence}
**Completed Sequence:**
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model=LLM_MODEL, temperature=temperature, base_url="https://api.chatanywhere.tech/v1")
    enrichment_chain = prompt | llm | StrOutputParser()
    
    enriched_sequence = enrichment_chain.invoke({
        "commands_to_add_str": ", ".join(sorted(list(commands_to_add))),
        "original_sequence": sequence
    })
    
    return enriched_sequence

# ==============================================================================
# 模块四: 主流程控制器 (Main Orchestrator)
# ==============================================================================

def process_directory(input_dir: str, output_dir: str, variations_per_seed: int):
    """
    自动化处理整个目录的种子文件。
    """
    setup_directories(input_dir, output_dir)
    
    all_seed_files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    
    if not all_seed_files:
        print(f"警告: 输入目录 '{input_dir}' 中没有找到种子文件。")
        return

    if MAX_ENRICHMENT_CORPUS_SIZE > 0 and len(all_seed_files) > MAX_ENRICHMENT_CORPUS_SIZE:
        print(f"发现 {len(all_seed_files)} 个种子，超过了 {MAX_ENRICHMENT_CORPUS_SIZE} 的限制。将随机选取一部分进行处理。")
        seed_files_to_process = random.sample(all_seed_files, MAX_ENRICHMENT_CORPUS_SIZE)
    else:
        seed_files_to_process = all_seed_files
    
    print(f"\n--- 将处理 {len(seed_files_to_process)} 个种子文件，开始批量扩充... ---")
    
    total_generated = 0
    for filename in seed_files_to_process:
        filepath = os.path.join(input_dir, filename)
        print(f"\n处理原始种子: {filename}")
        
        original_content = read_seed_file(filepath)
        if not original_content:
            continue
        
        generated_for_this_seed = 0
        for i in range(variations_per_seed):
            temp = random.uniform(0.3, 0.8)
            print(f"  -> 生成变体 {i+1}/{variations_per_seed} (temperature={temp:.2f})...")
            
            enriched_content = enrich_exim_sequence(original_content, temperature=temp)
            
            if enriched_content and enriched_content.strip() != original_content.strip():
                save_enriched_seed(output_dir, filename, enriched_content, i + 1)
                generated_for_this_seed += 1
        
        if generated_for_this_seed > 0:
            print(f"  成功为 {filename} 生成了 {generated_for_this_seed} 个新种子。")
        else:
            print(f"  未能为 {filename} 生成有效的新种子。")
            
        total_generated += generated_for_this_seed

    print("\n" + "="*50)
    print("🎉 批量扩充任务完成！")
    print(f"总计生成了 {total_generated} 个新种子，已保存至 '{output_dir}' 目录。")
    print("="*50)

# ==============================================================================
# 脚本入口
# ==============================================================================

if __name__ == '__main__':
    # 自动创建示例文件
    if not os.path.exists(INPUT_DIR) or not os.listdir(INPUT_DIR):
        print(f"未找到或空的'{INPUT_DIR}'目录，将创建并写入示例种子文件用于演示。")
        os.makedirs(INPUT_DIR, exist_ok=True)
        seed1_content = "EHLO client.example.com\r\nMAIL FROM:<sender@example.com>\r\n"
        with open(os.path.join(INPUT_DIR, "exim_incomplete_1.raw"), "w") as f: f.write(seed1_content)
        seed2_content = "HELO mail.server.com\r\n"
        with open(os.path.join(INPUT_DIR, "exim_greeting_only.raw"), "w") as f: f.write(seed2_content)
        print(f"已创建示例种子到 '{INPUT_DIR}' 目录。")

    # 执行主流程
    process_directory(INPUT_DIR, OUTPUT_DIR, NUM_VARIATIONS_PER_SEED)