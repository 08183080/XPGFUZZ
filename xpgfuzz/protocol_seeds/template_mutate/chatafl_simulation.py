import json  
import re  
import os  
from openai import OpenAI  
from typing import List, Dict, Tuple, Optional  
  
class ChatAFLGrammarExtractor:  
    def __init__(self, protocol_name: str, openai_api_key: str, output_dir: str,   
                 base_url: Optional[str] = None, api_version: Optional[str] = None):  
        self.protocol_name = protocol_name  
        self.output_dir = output_dir  
          
        # 使用新的OpenAI客户端初始化方式  
        self.client = OpenAI(  
            api_key=openai_api_key,  
            base_url=base_url or "https://api.openai.com/v1"  
        )  
          
        # 创建输出目录  
        os.makedirs(f"{output_dir}/protocol-grammars", exist_ok=True)  
        os.makedirs(f"{output_dir}/regex-patterns", exist_ok=True)  
      
    def construct_prompt_for_templates(self) -> str:  
        """构建语法模板提取的提示词"""  
        prompt_rtsp_example = """For the RTSP protocol, the DESCRIBE client request template is:  
DESCRIBE: ["DESCRIBE <<VALUE>>\\r\\n",  
"CSeq: <<VALUE>>\\r\\n",  
"User-Agent: <<VALUE>>\\r\\n",  
"Accept: <<VALUE>>\\r\\n",  
"\\r\\n"]"""  
          
        prompt_http_example = """For the HTTP protocol, the GET client request template is:  
GET: ["GET <<VALUE>>\\r\\n"]"""  
          
        message = f"{prompt_rtsp_example}\n{prompt_http_example}\nFor the {self.protocol_name} protocol, all of client request templates are :"  
          
        return json.dumps([  
            {"role": "system", "content": "You are a helpful assistant."},  
            {"role": "user", "content": message}  
        ])  
      
    def construct_prompt_for_remaining_templates(self, first_question: str, first_answer: str) -> str:  
        """构建获取剩余模板的提示词"""  
        second_question = f"For the {self.protocol_name} protocol, other templates of client requests are:"  
          
        return json.dumps([  
            {"role": "system", "content": "You are a helpful assistant."},  
            {"role": "user", "content": first_question},  
            {"role": "assistant", "content": first_answer},  
            {"role": "user", "content": second_question}  
        ])  
      
    def chat_with_llm(self, prompt: str, model: str = "gpt-3.5-turbo",   
                     max_retries: int = 5, temperature: float = 0.5) -> str:  
        """与LLM交互获取响应，使用新的OpenAI API"""  
        messages = json.loads(prompt)  
          
        for attempt in range(max_retries):  
            try:  
                # 根据模型类型选择不同的API调用方式  
                if model == "instruct" or "instruct" in model:  
                    # 对应原始代码中的instruct模式，使用completions端点  
                    response = self.client.completions.create(  
                        model="gpt-3.5-turbo-instruct",  
                        prompt=messages[1]["content"] if len(messages) > 1 else prompt,  
                        max_tokens=2048,  
                        temperature=temperature  
                    )  
                    return response.choices[0].text.strip()  
                else:  
                    # 对应原始代码中的chat模式，使用chat/completions端点  
                    response = self.client.chat.completions.create(  
                        model=model,  
                        messages=messages,  
                        temperature=temperature,  
                        max_tokens=2048  
                    )  
                    return response.choices[0].message.content  
            except Exception as e:  
                print(f"LLM调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")  
                if attempt == max_retries - 1:  
                    return None  
        return None  
      
    def extract_message_grammars(self, llm_response: str) -> List[Dict]:  
        """从LLM响应中提取语法模板"""  
        grammars = []  
        ptr = 0  
          
        while ptr < len(llm_response):  
            start = llm_response.find('[', ptr)  
            if start == -1:  
                break  
              
            end = llm_response.find(']', start)  
            if end == -1:  
                break  
              
            json_str = llm_response[start:end+1]  
            ptr = end + 1  
              
            try:  
                grammar_obj = json.loads(json_str)  
                grammars.append(grammar_obj)  
            except json.JSONDecodeError:  
                continue  
          
        return grammars  
      
    def convert_to_regex_pattern(self, template: str) -> str:  
        """将模板转换为正则表达式模式"""  
        # 移除引号  
        if template.startswith('"') and template.endswith('"'):  
            template = template[1:-1]  
          
        # 转换 <<VALUE>> 为捕获组  
        pattern = re.sub(r'<<.*?>>', r'(.*)', template)  
          
        # 转义特殊字符  
        pattern = pattern.replace('\\r\\n', r'\r\n')  
        pattern = pattern.replace('\\n', r'\n')  
        pattern = pattern.replace('\\r', r'\r')  
          
        return f"^{pattern}"  
      
    def save_templates_to_file(self, templates: str, iteration: int):  
        """保存模板到本地文件"""  
        filepath = f"{self.output_dir}/protocol-grammars/llm-grammar-output-{iteration}.txt"  
        with open(filepath, 'w', encoding='utf-8') as f:  
            f.write(templates)  
        print(f"模板已保存到: {filepath}")  
      
    def save_regex_patterns(self, patterns: Dict[str, str]):  
        """保存正则表达式模式到文件"""  
        filepath = f"{self.output_dir}/regex-patterns/patterns.json"  
        with open(filepath, 'w', encoding='utf-8') as f:  
            json.dump(patterns, f, indent=2, ensure_ascii=False)  
        print(f"正则表达式模式已保存到: {filepath}")  
      
    def load_regex_patterns(self) -> Dict[str, str]:  
        """从文件加载正则表达式模式"""  
        filepath = f"{self.output_dir}/regex-patterns/patterns.json"  
        if os.path.exists(filepath):  
            with open(filepath, 'r', encoding='utf-8') as f:  
                return json.load(f)  
        return {}  
      
    def setup_llm_grammars(self, consistency_count: int = 5) -> Dict[str, str]:  
        """主要的语法提取流程"""  
        print(f"开始为 {self.protocol_name} 协议提取语法模板...")  
        print(f"使用API端点: {self.client.base_url}")  
          
        # 构建一致性表  
        consistency_table = {}  
        regex_patterns = {}  
          
        # 构建初始提示  
        first_question = self.construct_prompt_for_templates()  
          
        for iteration in range(consistency_count):  
            print(f"执行第 {iteration + 1} 次迭代...")  
              
            # 获取初始模板  
            templates_answer = self.chat_with_llm(first_question, "gpt-3.5-turbo")  
            if not templates_answer:  
                print(f"第 {iteration + 1} 次迭代失败：无法获取初始模板")  
                continue  
              
            # 获取剩余模板  
            remaining_prompt = self.construct_prompt_for_remaining_templates(  
                json.loads(first_question)[1]["content"], templates_answer  
            )  
            remaining_templates = self.chat_with_llm(remaining_prompt, "gpt-3.5-turbo")  
            if not remaining_templates:  
                print(f"第 {iteration + 1} 次迭代失败：无法获取剩余模板")  
                continue  
              
            # 合并模板  
            combined_templates = f"{templates_answer}\n{remaining_templates}"  
              
            # 保存到文件  
            self.save_templates_to_file(combined_templates, iteration)  
              
            # 提取语法  
            grammars = self.extract_message_grammars(combined_templates)  
              
            # 处理每个语法模板  
            for grammar in grammars:  
                if isinstance(grammar, list) and len(grammar) > 0:  
                    header = grammar[0]  
                      
                    # 更新一致性表  
                    if header not in consistency_table:  
                        consistency_table[header] = {}  
                      
                    # 生成正则表达式模式  
                    pattern = self.convert_to_regex_pattern(header)  
                    regex_patterns[header] = pattern  
                      
                    # 处理字段  
                    for field in grammar[1:]:  
                        if field not in consistency_table[header]:  
                            consistency_table[header][field] = 0  
                        consistency_table[header][field] += 1  
          
        # 保存正则表达式模式  
        self.save_regex_patterns(regex_patterns)  
          
        print(f"语法提取完成！共提取 {len(regex_patterns)} 个模式")  
        return regex_patterns  
      
    def mutate_seed_with_grammar(self, seed_data: str, patterns: Dict[str, str]) -> List[str]:  
        """使用语法模式引导种子变异"""  
        mutations = []  
          
        for message_type, pattern in patterns.items():  
            try:  
                # 编译正则表达式  
                regex = re.compile(pattern)  
                  
                # 尝试匹配种子数据  
                match = regex.search(seed_data)  
                if match:  
                    # 生成变异  
                    for i in range(5):  # 生成5个变异  
                        mutated = seed_data  
                        for group_idx, group_value in enumerate(match.groups(), 1):  
                            # 简单的变异策略：修改捕获组的值  
                            new_value = f"MUTATED_{group_idx}_{i}"  
                            mutated = mutated.replace(group_value, new_value, 1)  
                        mutations.append(mutated)  
            except re.error as e:  
                print(f"正则表达式错误 {pattern}: {e}")  
          
        return mutations  
  
# 使用示例  
def main():  
    # 示例1: 使用默认OpenAI API  
    extractor1 = ChatAFLGrammarExtractor(  
        protocol_name="FTP/proftpd",  
        openai_api_key='sk-AIQaabGrgQ4TYkxAU6msW77ojWeHLf1MUpwvaOkz9Ql0LYvo',  
        output_dir="./chatafl_output",
        base_url='https://api.chatanywhere.tech/v1'  
    )  
      
    # 执行语法提取  
    patterns = extractor1.setup_llm_grammars()  
      
    # 加载已保存的模式  
    loaded_patterns = extractor1.load_regex_patterns()  
    print(f"加载的模式: {loaded_patterns}")  
      
    # 使用模式进行种子变异  
    seed_example = "USER ubuntu\r\nPASS 123\r\n\r\n"  
    mutations = extractor1.mutate_seed_with_grammar(seed_example, loaded_patterns)  
      
    print(f"生成 {len(mutations)} 个变异种子")  
    for i, mutation in enumerate(mutations[:3]):  # 显示前3个  
        print(f"变异 {i+1}: {mutation}")  
  
if __name__ == "__main__":  
    main()