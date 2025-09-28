import json
import re
import os
import random
import sys
from openai import OpenAI
from typing import List, Dict, Optional, Any
from pathlib import Path

# ==============================================================================
# 模块1: 细粒度变异器 (Fine-Grained Mutator)
# ==============================================================================

class FineGrainedMutator:
    """根据数据类型生成细粒度的变异数据。"""

    def mutate(self, type_str: str, original_value: str) -> List[str]:
        type_str = type_str.lower()
        if type_str == 'integer':
            return self._mutate_integer(original_value)
        elif type_str == 'string':
            return self._mutate_string(original_value)
        elif type_str == 'path':
            return self._mutate_path(original_value)
        elif type_str == 'ip':
            return self._mutate_ip(original_value)
        elif type_str == 'port':
            return self._mutate_port(original_value)
        elif type_str == 'enum' or type_str == 'header_enum' or type_str == 'method_enum':
            return self._mutate_enum(original_value)
        elif type_str == 'email':
            return self._mutate_email(original_value)
        elif type_str == 'uri' or type_str == 'sip_uri':
            return self._mutate_uri(original_value)
        elif type_str == 'kv_pair':
            return self._mutate_kv_pair(original_value)
        elif type_str == 'multiline':
            return self._mutate_multiline(original_value)
        elif type_str == 'sessionid':
            return self._mutate_sessionid(original_value)
        else:
            return self._mutate_any(original_value)

    # ----------------- 整数变异 -----------------
    def _mutate_integer(self, value: str) -> List[str]:
        if random.random() < 0.5:
            return self._mutate_integer_constraints(value)
        else:
            return self._mutate_integer_attacks(value)

    def _mutate_integer_constraints(self, value: str) -> List[str]:
        mutations = []
        try:
            num = int(value)
            mutations.extend([str(num + 1), str(num - 1), str(num * 2), str(num // 2 if num != 0 else 0)])
        except (ValueError, TypeError):
            mutations.extend(["0", "1", "-1"])
        return list(set(mutations))

    def _mutate_integer_attacks(self, value: str) -> List[str]:
        return list(set([
            "0", "1", "-1",
            str(random.randint(2, 100)), str(random.randint(-100, -2)),
            str(2**15 - 1), str(2**16), str(2**31 - 1), str(2**32),
            str(sys.maxsize)
        ]))

    # ----------------- 字符串变异 -----------------
    def _mutate_string(self, value: str) -> List[str]:
        if random.random() < 0.5:
            return ["", "A" * 100, "A" * 1024, "' OR 1=1 --", "<script>alert(1)</script>",
                    "../" * 10, "%s%s%s%n%n", "null", "undefined"]
        else:
            return [value[:1], value * 2, value[::-1], value + "_test"]

    # ----------------- 路径变异 -----------------
    def _mutate_path(self, value: str) -> List[str]:
        if random.random() < 0.5:
            return ["/", "/etc/passwd", "/bin/sh", "C:\\Windows\\system32\\kernel32.dll",
                    "../../../../../../../../etc/hosts", "file:///dev/null", "nul"]
        else:
            return [value + "_bak", value + ".tmp", "/tmp/" + os.path.basename(value)]

    # ----------------- IP地址变异 -----------------
    def _mutate_ip(self, value: str) -> List[str]:
        return [
            "0.0.0.0", "127.0.0.1", "255.255.255.255",
            "::1", "2001:db8::1",
            "999.999.999.999", "1.1.1"
        ]

    # ----------------- 端口变异 -----------------
    def _mutate_port(self, value: str) -> List[str]:
        return ["0", "1", "65535", "65536", str(random.randint(-1000, 100000))]

    # ----------------- 枚举变异 -----------------
    def _mutate_enum(self, value: str) -> List[str]:
        mutations = [
            value.upper(), value.lower(), value[::-1],
            value + "!!!", "INVALID", "UNKNOWN"
        ]
        return list(set(mutations))

    # ----------------- Email变异 -----------------
    def _mutate_email(self, value: str) -> List[str]:
        return [
            "test@example.com",
            "a"*256 + "@example.com",
            "invalid-email",
            "test@ex!ample.com",
            "user@localhost",
            ""
        ]

    # ----------------- URI / SIP URI 变异 -----------------
    def _mutate_uri(self, value: str) -> List[str]:
        return [
            "rtsp://127.0.0.1:554/stream",
            "sip:user@domain:5060",
            "rtssp://invalid",
            value + "/A"*100,
            "sipss://user@:invalid"
        ]

    # ----------------- KV Pair 变异 -----------------
    def _mutate_kv_pair(self, value: str) -> List[str]:
        return [
            "key=value",
            "key=",
            "=value",
            "key1=value1;key2=value2",
            value + ";expires=0"
        ]

    # ----------------- 多行内容变异 -----------------
    def _mutate_multiline(self, value: str) -> List[str]:
        return [
            value + "\r\nExtra: line",
            "\r\n".join([value] * 10),
            value.replace("\n", "\r\n"),
            "\x00" + value,
            ""
        ]

    # ----------------- 会话ID变异 -----------------
    def _mutate_sessionid(self, value: str) -> List[str]:
        return [
            "1", "1234567890",
            "A"*1024,
            value + ";DROP TABLE",
            ""
        ]

    # ----------------- 通用变异 -----------------
    def _mutate_any(self, value: str) -> List[str]:
        if not value:
            return ["MUTATED_ANY"]
        if random.random() < 0.5:
            return [value * 2, value[:len(value)//2] if len(value) > 1 else "A"]
        else:
            return ["MUTATED_ANY", "FUZZ_" + value, "ZZZ"]



# ==============================================================================
# 模块2: 协议语法提取器 (Grammar Extractor)
# ==============================================================================

class GrammarExtractor:
    """负责从LLM提取、解析、加载和保存协议语法。"""
    def __init__(self, protocol_name: str, base_output_dir: str, openai_api_key: str, base_url: Optional[str] = None):
        self.protocol_name = protocol_name
        self.output_dir = Path(base_output_dir) / self.protocol_name / "protocol-grammars"
        self.grammar_filepath = self.output_dir / "grammars.json"
        self.client = OpenAI(api_key=openai_api_key, base_url=base_url or "https://api.openai.com/v1")
        self.grammars: List[Dict[str, Any]] = []
        
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_grammars(self) -> bool:
        """从文件加载结构化的语法。"""
        if self.grammar_filepath.exists():
            with open(self.grammar_filepath, 'r', encoding='utf-8') as f:
                self.grammars = json.load(f)
            print(f"Loaded {len(self.grammars)} grammars for '{self.protocol_name}' from {self.grammar_filepath}")
            return True
        return False

    def save_grammars(self):
        """保存结构化的语法到JSON文件。"""
        with open(self.grammar_filepath, 'w', encoding='utf-8') as f:
            json.dump(self.grammars, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(self.grammars)} grammars to: {self.grammar_filepath}")

    def setup_llm_grammars(self, consistency_count: int = 2):
        """主要的语法提取流程。"""
        print(f"Starting fine-grained grammar extraction for {self.protocol_name}...")
        first_prompt_str = self._construct_prompt_for_templates()
        first_question_content = json.loads(first_prompt_str)[1]['content']
        
        combined_llm_output = ""
        for i in range(consistency_count):
            print(f"Running iteration {i + 1}/{consistency_count}...")
            answer = self._chat_with_llm(first_prompt_str)
            if answer:
                combined_llm_output += answer + "\n"
                remaining_prompt = self._construct_prompt_for_remaining_templates(first_question_content, answer)
                remaining_answer = self._chat_with_llm(remaining_prompt)
                if remaining_answer:
                    combined_llm_output += remaining_answer + "\n"
        
        self._parse_and_store_grammars(combined_llm_output)
        unique_grammars = {g['name']: g for g in self.grammars}
        self.grammars = list(unique_grammars.values())
        self.save_grammars()
        print(f"Grammar extraction complete! Extracted {len(self.grammars)} unique grammar rules.")

    def _construct_prompt_for_templates(self) -> str:
        prompt_rtsp_example = 'For the RTSP protocol, the DESCRIBE client request template is:\nDESCRIBE: ["DESCRIBE <<uri:path>>\\r\\n", "CSeq: <<cseq:integer>>\\r\\n", "User-Agent: <<agent:string>>\\r\\n", "Accept: <<accept_type:string>>\\r\\n", "\\r\\n"]'
        prompt_http_example = 'For the HTTP protocol, the GET client request template is:\nGET: ["GET <<url:path>> HTTP/1.1\\r\\n", "Host: <<host:string>>\\r\\n", "Connection: <<connection:any>>\\r\\n", "\\r\\n"]'
        message = f"{prompt_rtsp_example}\n{prompt_http_example}\nFor the {self.protocol_name} protocol, all of client request templates are [我要协议的全部命令] :"
        return json.dumps([
            {"role": "system", "content": "You are a helpful assistant for fuzzing. Provide network protocol templates using the format KEY: [\"line1\", \"line2\"]. Use typed placeholders like <<variable_name:type>> where possible. Supported types are: integer, string, path, any."},
            {"role": "user", "content": message}
        ])

    def _construct_prompt_for_remaining_templates(self, first_question: str, first_answer: str) -> str:
        second_question = f"For the {self.protocol_name} protocol, what are some other templates for client requests?"
        return json.dumps([{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": first_question},
                           {"role": "assistant", "content": first_answer}, {"role": "user", "content": second_question}])

    def _chat_with_llm(self, prompt: str, model: str = "gpt-3.5-turbo", max_retries: int = 5, temperature: float = 0.5) -> Optional[str]:
        messages = json.loads(prompt)
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=2048)
                return response.choices[0].message.content
            except Exception as e:
                print(f"LLM call failed (attempt {attempt + 1}/{max_retries}): {e}")
        return None

    def _parse_and_store_grammars(self, llm_response: str):
        """解析LLM响应，生成灵活、非贪婪、无头部锚点的正则表达式。"""
        pattern = re.compile(r'([A-Z_]+):\s*(\[.*?\])', re.DOTALL)
        matches = pattern.findall(llm_response)
        for name, body in matches:
            try:
                template_lines = json.loads(body)
                if not isinstance(template_lines, list): continue
                full_template_str = "".join(template_lines)
                
                placeholder_find_pattern = r'<<([a-zA-Z0-9_]+):([a-zA-Z0-9]+)>>'
                placeholders = re.findall(placeholder_find_pattern, full_template_str)
                placeholder_split_pattern = r'(<<(?:[a-zA-Z0-9_]+):(?:[a-zA-Z0-9]+)>>)'
                
                parts = re.split(placeholder_split_pattern, full_template_str)
                regex_parts = []
                for part in parts:
                    if re.fullmatch(placeholder_split_pattern, part):
                        regex_parts.append('(.*?)') # 非贪婪匹配
                    elif part:
                        regex_parts.append(re.escape(part))
                
                final_regex = "".join(regex_parts)
                final_regex = final_regex.replace('\\r\\n', '\\r?\\n') # 兼容 \n 和 \r\n
                
                # 不再使用 ^ 锚点
                self.grammars.append({"name": name.strip(), "template": template_lines, "regex": final_regex,
                                      "placeholders": [{"name": p[0], "type": p[1]} for p in placeholders]})
            except (json.JSONDecodeError, re.error) as e:
                print(f"Skipping grammar block '{name}' due to parsing error: {e}")


# ==============================================================================
# 模块3: 模糊测试器 (Fuzzer)
# ==============================================================================

class Fuzzer:
    """模糊测试执行器，负责加载种子、变异和保存结果。"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.protocol_name = config["protocol_name"]
        
        self.base_input_dir = Path(config["base_input_dir"])
        self.base_output_dir = Path(config["base_output_dir"])
        
        self.seed_dir = self.base_input_dir / self.protocol_name
        self.mutation_dir = self.base_output_dir / self.protocol_name / "mutations"
        
        self.grammars = config["grammars"]
        self.mutator = FineGrainedMutator()

        self.mutation_dir.mkdir(parents=True, exist_ok=True)

    def _load_seeds(self) -> Dict[str, str]:
        """从指定目录批量加载所有种子文件。"""
        seeds = {}
        if not self.seed_dir.exists():
            print(f"Seed directory not found: {self.seed_dir}")
            return seeds
            
        print(f"Loading seeds from: {self.seed_dir}")
        for filepath in self.seed_dir.iterdir():
            if filepath.is_file():
                try:
                    with open(filepath, "r", encoding="utf-8", newline="") as f:
                        seeds[filepath.name] = f.read()
                except Exception as e:
                    print(f"Could not read seed file {filepath.name}: {e}")
        print(f"Loaded {len(seeds)} seeds.")
        return seeds

    def _save_mutations(self, original_filename: str, mutations: List[str]):
        """将一个种子的所有变异结果批量保存，命名规则: {原始种子名}_mut_{id}{扩展名}"""
        base_name, ext = os.path.splitext(original_filename)
        if not ext:  # 没有扩展名时默认使用 .raw
            ext = ".raw"

        for i, mutation_content in enumerate(mutations):
            mutation_filename = f"{base_name}_mut_{i+1:03d}{ext}"
            mutation_filepath = self.mutation_dir / mutation_filename
            try:
                with open(mutation_filepath, 'w', encoding='utf-8', newline='') as f:
                    f.write(mutation_content)
            except Exception as e:
                print(f"Could not save mutation file {mutation_filepath}: {e}")
        
        print(f"Saved {len(mutations)} mutations for '{original_filename}' to '{self.mutation_dir}'")


    def mutate_seed(self, seed_data: str) -> List[str]:
        """对单个种子应用所有匹配的语法规则，并对每个规则的所有出现位置进行变异。"""
        mutations = set()
        
        for grammar in self.grammars:
            try:
                regex = re.compile(grammar['regex'], re.DOTALL)
                
                for match in re.finditer(regex, seed_data):
                    print(f"  - Seed matched grammar: '{grammar['name']}' at position {match.start()}")
                    captured_values = match.groups()
                    placeholders = grammar['placeholders']

                    if len(captured_values) != len(placeholders):
                        print(f"    Warning: Mismatch in captures for grammar '{grammar['name']}'. Skipping.")
                        continue

                    for i, original_value in enumerate(captured_values):
                        ph = placeholders[i]
                        new_values = self.mutator.mutate(ph['type'], original_value)
                        
                        for new_val in new_values:
                            start, end = match.span(i + 1)
                            mutated_seed = seed_data[:start] + new_val + seed_data[end:]
                            mutations.add(mutated_seed)

            except re.error as e:
                print(f"Regex error for grammar '{grammar['name']}': {e}")
        
        return list(mutations)

    def run(self):
        """执行模糊测试的主流程。"""
        print("\n" + "="*50)
        print(f"Starting Fuzzer for protocol: {self.protocol_name}")
        print("="*50)

        seeds = self._load_seeds()
        if not seeds:
            print("No seeds found. Exiting.")
            return

        for filename, content in seeds.items():
            print(f"\n--- Mutating seed: {filename} ---")
            mutations = self.mutate_seed(content)
            
            if mutations:
                self._save_mutations(filename, mutations)
            else:
                print("  - No mutations were generated. The seed did not match any loaded grammars.")
        
        print("\n" + "="*50)
        print("Fuzzing process complete.")
        print("="*50)

# ==============================================================================
# 4. 主程序入口 (Main Execution)
# ==============================================================================

def main():
    """主函数，负责配置和启动整个流程。"""
    
    # --- 在这里集中配置所有参数 ---
    CONFIG = {
        "protocol_name": 'SIP',
        "base_input_dir": "./input_seeds",   # 存放所有协议种子父目录
        "base_output_dir": "./output_results", # 存放所有协议结果的父目录
        "openai_api_key": os.getenv("OPENAI_API_KEY", None), # 优先从环境变量获取
        "base_url": os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    }
    # --------------------------------

    # 为方便测试，如果环境变量中没有，可以在此处填入备用key
    if not CONFIG["openai_api_key"]:
        # CONFIG["openai_api_key"] = "sk-..." # 在此填入你的key
        print("Error: OPENAI_API_KEY is not set in environment variables or in the script.")
        print("Please set it before running.")
        return

    # 1. 初始化语法提取器
    extractor = GrammarExtractor(
        protocol_name=CONFIG["protocol_name"],
        base_output_dir=CONFIG["base_output_dir"],
        openai_api_key=CONFIG["openai_api_key"],
        base_url=CONFIG["base_url"]
    )

    # 2. 加载或生成语法
    if not extractor.load_grammars():
        print("No existing grammars found. Starting extraction process...")
        extractor.setup_llm_grammars()
        extractor.load_grammars()

    if not extractor.grammars:
        print("Failed to load or generate any grammars. Exiting.")
        return
    
    # 3. 配置并启动 Fuzzer
    fuzzer_config = {
        "protocol_name": CONFIG["protocol_name"],
        "base_input_dir": CONFIG["base_input_dir"],
        "base_output_dir": CONFIG["base_output_dir"],
        "grammars": extractor.grammars
    }
    fuzzer = Fuzzer(fuzzer_config)
    fuzzer.run()

if __name__ == "__main__":
    main()
