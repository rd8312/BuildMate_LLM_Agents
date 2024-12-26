import openai
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re

class Agent():
    
    def __init__(self, agent_name, prompt, model):
        
        self.agent_name = agent_name
        self.memory = []
        
        template = ChatPromptTemplate.from_template(prompt)
        model = ChatOpenAI(model=model, temperature=0.7)
        parser = StrOutputParser()

        self.llm = template | model | parser
        
    def forward(self, user_message):
        
        # ### Design your logic here ###
        
        # llm_message = self.llm.invoke(user_message)
        # self.memory.append(llm_message)
        
        # return message
        
        # ##############################
        
        pass
    
    def __call__(self, component_dict):
        
        component_dict = self.forward(component_dict)
        
        return component_dict
    
    def get_component_list(self, component_dict):

        component_list = ''

        for i, (key, value) in enumerate(component_dict.items()):
            
            num = i + 1
            if 'name' in value and 'price' not in value:
                component_list += f'{num}. {key}: None | price: {value["price"]}\n'
            elif 'name' in value and 'price' in value:
                component_list += f'{num}. {key}: {value["name"]} | price: {value["price"]}\n'
            else:
                component_list += f'{num}. {key}: None | price: None \n'

        return component_list
    
class Planning_Agent(Agent):
    
    def __init__(self, agent_name, model, require, budget):
        
        self.require = require
        self.budget = budget

        prompt = """
            你現在是一個電腦的零件專家，現在，有一個使用者有自己組裝機器的需求，
            請根據它的需求內容，分配總預算給每一個零件做個別的預算。
            
            使用者需求：{require}
            使用者預算：{budget}
            
            電腦的零件總共有：Mother board, Case, CPU, GPU, Memory, Storage, Power, Fan
            請根據上述八個零件，分配總預算給每一個零件
            
            輸入的範例：
            
            使用者需求：我想要一個50000元的遊戲機器
            使用者預算：65000
            
            輸出的範例：
            1.[ Mother board ]：2500
            2.[ Case ]：1500
            3.[ CPU ]：30000
            4.[ GPU ]：20000
            5.[ Memory ]：5000
            6.[ Storage ]：3500
            7.[ Power ]：2000
            8.[ Fan ]：500
            
            另外有一點請注意：
            產品的價格，請回覆一個完整的數字，中間不需要加入逗號
        """
        
        super().__init__(agent_name, prompt, model)
        
    def forward(self, component_dict):
        
        user_message = {}
        user_message['require'] = self.require
        user_message['budget'] = self.budget
        
        message = self.llm.invoke(user_message)
        self.memory.append(message)
        
        component_dict = self.parser_message(message)
        
        return component_dict
    
    def parser_message(self, message):

        pattern = r"\[\s*(.*?)\s*\]\s*[:\uff1a]?\s*(\d+)"
        matches = re.findall(pattern, message)

        component_dict = {match[0]: {'price': int(match[1])} for match in matches}

        return component_dict
    
class Component_Agent(Agent):
    
    def __init__(self, component_name, price_list, model, require, budget):
        
        self.price_list = price_list
        self.require = require
        self.budget = budget
        
        self.component_info = {
            'Mother board': {
                'description': '主機板',
                'examples': 'ASUS ROG, MSI MEG, Gigabyte AORUS',
                'key_features': '晶片組、PCIe規格、記憶體支援、擴充插槽'
            },
            'Case': {
                'description': '機殼',
                'examples': 'Lian Li, Fractal Design, NZXT',
                'key_features': '散熱空間、擴充性、風扇安裝位置、Storage位置'
            },
            'CPU': {
                'description': '處理器',
                'examples': 'Intel Core i9, AMD Ryzen 9',
                'key_features': '核心數、時脈速度、快取、功耗'
            },
            'GPU': {
                'description': '顯示卡',
                'examples': 'NVIDIA RTX 4090, RTX 4080, A5000',
                'key_features': '深度學習性能、CUDA核心數、顯存容量(建議>=12GB)、FP32/FP16效能'
                },
            'Memory': {
                'description': '記憶體',
                'examples': 'Corsair, G.SKILL, Crucial',
                'key_features': '容量、頻率、延遲、代數'
            },
            'Storage': {
                'description': '儲存裝置',
                'examples': 'Samsung SSD, WD Black, Seagate',
                'key_features': '容量、讀寫速度、耐用度、介面'
            },
            'Power': {
                'description': '電源供應器',
                'examples': 'Seasonic Prime 1000W, Corsair AX1200i',
                'key_features': '瓦數(深度學習建議>=1000W)、80 Plus認證、模組化、穩定度'
            },
            'Fan': {
                'description': '機殼風扇',
                'examples': 'Noctua NF-A12x25, Arctic P12, be quiet! Silent Wings',
                'key_features': '尺寸(通常是12cm或14cm)、風量、噪音、軸承類型、RGB燈效'
            }
        }
        component_prompt = self.component_prompt_maker(component_name)
        super().__init__(component_name, component_prompt, model)
        self.component_chinese_name = self.component_info[component_name]['description']
        
    def forward(self, component_dict):
        
        component_list = self.get_component_list(component_dict)
        
        user_message = {}
        user_message['component_chinese_name'] = self.component_chinese_name
        user_message['budget'] = component_dict[self.agent_name]['price']
        user_message['require'] = self.require
        user_message['component_list'] = component_list
        
        message = self.llm.invoke(user_message)
        self.memory.append(message)
        
        component_dict[self.agent_name] = self.parser_message(message)
        
        return component_dict
        
    def component_prompt_maker(self, component_name):

        info = self.component_info[component_name]
        prompt = f"""
        你是一個專業的電腦-{info['description']}銷售專家。
        你只能推薦 {component_name} 類別的產品，例如：{info['examples']}。
        評估產品時要考慮：{info['key_features']}。
        
        另外，這邊提供了一組{info['description']}的選擇清單，
        請從以下選擇清單中，挑選低於預算的產品：
        {self.price_list}
        """
        
        prompt = prompt + """
        客戶對於{component_chinese_name}的預算為：{budget}
        客戶的電腦需求為：{require}
        目前已經有的電腦組合清單為：{component_list}

        請說明推薦原因，並提供產品名稱，範例的回饋訊息如下：
        [ 推薦原因 ] ： 耐用性強，可以使用到最新規格的Intel CPU...
        [ 產品名稱 ] ： Inter 第八代 i9
        [ 產品價格 ] ： 13000
        
        另外有幾點請注意：
        1. 推薦原因的說明請限制在50個字以內
        2. 產品的價格，請回覆一個完整的數字，中間不需要加入逗號
        3. 請從選擇清單中，挑出一個價格低於預算，且適合客戶電腦需求的產品
        """

        return prompt
    
    def parser_message(self, message):
        pattern = {
            "reason": r"\[ 推薦原因 \]\s*：\s*(.*?)\s*\n",
            "name": r"\[ 產品名稱 \]\s*：\s*(.*?)\s*\n",
            "price": r"\[ 產品價格 \]\s*：\s*(\d+)"
        }

        results = {key: re.search(regex, message).group(1) for key, regex in pattern.items()}

        return results

class Reducing_Agent(Agent):
    
    def __init__(self, agent_name, model, require, budget):
        
        prompt ="""
            你是一個專業的電腦銷售機器人，目前客戶遇到了一個難題，
            它有一組電腦的清單，清單中含有產品的名稱與價格，
            但清單的總價格超出了預算。
            
            身為專業的電腦銷售機器人，請挑出這個電腦清單中，
            你覺得哪一個產品可以被替換成更低價格的產品，從而使得整體的價格可以低於或者等於客戶的預算。
            
            客戶的需求為：{require}
            客戶的預算為：{budget}
            目前已經有的電腦組合清單為：{component_list}
            目前已經有的電腦組合清單的總價格：{total_price}
            距離客戶預算的差價：{difference}

            請指出哪一個種類的產品可以被替換，
            請說明推薦原因，範例的回饋訊息如下：
            [ 替換產品 ] ： CPU
            [ 替換原因 ] ： 耐用性強，可以使用到最新規格的Intel CPU...
            
            另外有幾點請注意：
            1. 替換原因的說明請限制在50個字以內
            2. 替換產品的選項只能限制於右側八個產品：Mother board, Case, CPU, GPU, Memory, Storage, Power, Fan
            """       
        
        super().__init__(agent_name, prompt, model)
        
        self.require = require
        self.budget = budget
    
    def forward(self, component_dict):
            
        total_price = self.summary_price(component_dict)
        difference = self.budget - total_price
        
        component_list = self.get_component_list(component_dict)
        
        user_message = {}
        user_message['component_list'] = component_list
        user_message['total_price'] = total_price
        user_message['difference'] = difference
        user_message['require'] = self.require
        user_message['budget'] = self.budget
        
        message = self.llm.invoke(user_message)
        self.memory.append(message)
        product_list = ['Mother board', 'Case', 'CPU', 'GPU', 'Memory', 'Storage', 'Power', 'Fan']
        
        for product_name in product_list:
            if product_name in message:
                return product_name
        
        return None
        
    def parser_replace_message(self, message):
        
        patterns = {
            "replace_component": r"\[ 替換產品 \] ：\s*(.*?)\s*\n",
            "reason": r"\[ 推薦原因 \] ：\s*(.*?)\s*\n",
            "name": r"\[ 產品名稱 \] ：\s*(.*?)\s*\n",
            "price": r"\[ 產品價格 \] ：\s*(\d+)"
        }

        replace_component = re.search(patterns["replace_component"], message).group(1)

        results = {
            "reason": re.search(patterns["reason"], message).group(1),
            "name": re.search(patterns["name"], message).group(1),
            "price": re.search(patterns["price"], message).group(1)
        }

        return replace_component, results
    
    def summary_price(self, component_dict):
    
        return sum([int(component['price']) for component in component_dict.values()])

class Power_Agent(Agent):
    
    def __init__(self, agent_name, prompt, model):
        super().__init__(agent_name, prompt, model)

    def forward(self, component_dict):
        
        component_list = self.get_component_list(component_dict)
        
        user_message = {}
        user_message['component_list'] = component_list
        
        message = self.llm.invoke(user_message)
        self.memory.append(message)

        if '[ 結論 ]：電源供應充足' in message:
            pass
        elif '[ 結論 ]：電源供應不充足' in message:
            replace_component, results = self.parser_replace_message(message)
            component_dict[replace_component] = results
        else:
            raise Exception('There return message does not follow the rule.')
        
        return component_dict
        
    def parser_replace_message(self, message):
        
        patterns = {
            "replace_component": r"\[ 替換產品 \] ：\s*(.*?)\s*\n",
            "reason": r"\[ 推薦原因 \] ：\s*(.*?)\s*\n",
            "name": r"\[ 產品名稱 \] ：\s*(.*?)\s*\n",
            "price": r"\[ 產品價格 \] ：\s*(\d+)"
        }

        replace_component = re.search(patterns["replace_component"], message).group(1)

        results = {
            "reason": re.search(patterns["reason"], message).group(1),
            "name": re.search(patterns["name"], message).group(1),
            "price": re.search(patterns["price"], message).group(1)
        }

        return replace_component, results
    
class Heat_Dissipation_Agent(Agent):
    
    def __init__(self, agent_name, prompt, model):
        super().__init__(agent_name, prompt, model)

    def forward(self, component_dict):
        
        component_list = self.get_component_list(component_dict)
        
        user_message = {}
        user_message['component_list'] = component_list
        
        message = self.llm.invoke(user_message)
        self.memory.append(message)

        if '[ 結論 ]：散熱充足' in message:
            pass
        elif '[ 結論 ]：散熱不充足' in message:
            replace_component, results = self.parser_replace_message(message)
            component_dict[replace_component] = results
        else:
            raise Exception('There return message does not follow the rule.')
        
        return component_dict
        
    def parser_replace_message(self, message):
        
        patterns = {
            "replace_component": r"\[ 替換產品 \] ：\s*(.*?)\s*\n",
            "reason": r"\[ 推薦原因 \] ：\s*(.*?)\s*\n",
            "name": r"\[ 產品名稱 \] ：\s*(.*?)\s*\n",
            "price": r"\[ 產品價格 \] ：\s*(\d+)"
        }

        replace_component = re.search(patterns["replace_component"], message).group(1)

        results = {
            "reason": re.search(patterns["reason"], message).group(1),
            "name": re.search(patterns["name"], message).group(1),
            "price": re.search(patterns["price"], message).group(1)
        }

        return replace_component, results
    
class Debate_Agents(Agent):
    
    def __init__(self, agent_name, product, first_company, second_company, model, limited_time=2):
        
        self.first_company = first_company
        self.second_company = second_company
        self.product = product
        
        prompt = """
            你現在是{A_company}的廠商代表，對於自己生產的產品瞭若指掌，且非常有自信。
            你的對面坐著{B_company}，他是另一個{product}的廠商代表。
            你們正在進行著挑選哪一個{product}的辯論。
            
            現在，請根據user的問題，試著去說服user採用你們家廠商生產的產品，
            說服他你的觀點才是更好的。
        """
        
        rules_prompt = """
            下面有三個規則要遵守：
            1. 字數內容100字為上限
            2.提出產品名稱
            3.提出自身產品優勢內容
            4.提出另一牌子的缺點
        """
        
        first_prompt = prompt.format(A_company=self.first_company, 
                                     B_company=self.second_company, 
                                     product=self.product) + """
            用戶需求為：{topic}
            {Opposition}廠商的發言：{previous_message}
        """ + rules_prompt
        
        self.first_agent = Agent(self.first_company, first_prompt, model)
        
        second_prompt = prompt.format(A_company=self.second_company, 
                                     B_company=self.first_company, 
                                     product=self.product) + """
            用戶需求為：{topic}
            {Opposition}廠商的發言：{previous_message}
        """ + rules_prompt
        
        self.second_agent = Agent(self.second_company, second_prompt, model)
        

        prompt = f"""
                你是一個專業的{self.product}專家，現在，有兩個廠商代表著自己家的產品正在進行辯論。
                目前有：{self.first_company}的代表與{self.second_company}的代表。\n
        """
        
        summary_prompt = prompt + """
                以下為他們的爭辯過程：
                {memory}

                請根據他們的爭辯內容，根據客戶的需求：{topic}，
                選出你覺得最合適客戶需求的廠商與它推薦的產品，並預估該產品的價格。

                輸出範例：
                [ 推薦原因 ] ： 耐用性強，可以使用到最新規格的Intel CPU...
                [ 產品名稱 ] ： Inter 第八代 i9
                [ 產品價格 ] ： 13000

                另外有幾點請注意：
                1. 推薦原因的說明請限制在50個字以內
                2. 產品的價格，請回覆一個完整的數字，中間不需要加入逗號
        """
        
        super().__init__(agent_name, summary_prompt, model)
        
        self.limited_time = limited_time

    def forward(self, component_dict):

        previous_message = ''
        time = 1
        
        question = f'我目前預算在{component_dict[self.product]["price"]}的附近，想要買一個{self.product}，請幫我推薦'

        while self.limited_time != time:

            round_message = f' ======= 第 {time} 回合 ======= \n題目：{question}'

            first_response = self.first_agent.llm.invoke({"topic":question,
                                                      "previous_message":previous_message,
                                                      "Opposition":self.second_agent})
            previous_message = f'{self.first_agent.agent_name}廠商的發言：{first_response}\n'
            self.memory.append(previous_message)

            second_response = self.second_agent.llm.invoke({"topic":question,
                                                        "previous_message":previous_message,
                                                        "Opposition":self.first_agent})
            previous_message = f'{self.second_agent.agent_name}廠商的發言：{second_response}\n'
            self.memory.append(previous_message)

            time += 1
        
        memory_string = ''.join(self.memory)

        recommendation = self.llm.invoke({"topic":question, "memory":memory_string})
        
        component_dict[self.product] = self.parser_message(recommendation)
        
        return component_dict
        
    def parser_message(self, message):
        pattern = {
            "reason": r"\[ 推薦原因 \] ：\s*(.*?)\s*\n",
            "name": r"\[ 產品名稱 \] ：\s*(.*?)\s*\n",
            "price": r"\[ 產品價格 \] ：\s*(\d+)"
        }

        results = {key: re.search(regex, message).group(1) for key, regex in pattern.items()}

        return results
    
