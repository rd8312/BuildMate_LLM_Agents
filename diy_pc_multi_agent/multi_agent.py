from agent import Component_Agent, Reducing_Agent, Power_Agent, Heat_Dissipation_Agent
from agent import Planning_Agent, Debate_Agents
import re
import requests
from bs4 import BeautifulSoup
import nest_asyncio
import asyncio

class Multi_Agent():
    
    def __init__(self):
        
        pass
    
    def execute(self, data):
        
        pass
    
class PC_DIY_Debate_System(Multi_Agent):
    
    def __init__(self, model, require, budget):
        super().__init__()
        
        self.agent_dict = {}
        
        ### Build planning agent ###
        self.planning_agent = Planning_Agent('planning_agent', 'gpt-4o-mini', '我想要一個45000元的遊戲電腦', 45000)
            
        ### Build debate agents ###
        product_and_companies = {'Mother board': ['華碩', '技嘉'],
                         'Case': ['酷碼', '全漢'],
                         'CPU': ['Intel', 'AMD'],
                         'GPU': ['微星', '技嘉'],
                         'Memory': ['美光', '金士頓'],
                         'Storage': ['三星', '威剛'],
                         'Power': ['海韻', '華碩'],
                         'Fan': ['威剛', '聯力']}

        self.debate_agents_list = []

        for product_name, companies in product_and_companies.items():
            
            debate_cpu_agents = Debate_Agents(f'debate_{product_name}', product_name, companies[0], companies[1], 'gpt-4o-mini', 2)
            self.debate_agents_list.append(debate_cpu_agents)
            
    def execute(self, component_dict):
        
        ### Budget allocation###
        
        component_dict = self.planning_agent({})
            
        # ### Debate with components ###
        
        for debate_agents in self.debate_agents_list:
            component_dict = debate_agents(component_dict)
        
        return component_dict
    
class PC_DIY_System(Multi_Agent):
    
    def __init__(self, model, require, budget):
        super().__init__()
        
        self.agent_dict = {}
        
        ### Build panning agent ###
        coolpc_dict = {
            'CPU': [4, "處理器 CPU"],
            'Mother board': [5, "主機板 MB"],
            'Memory': [6, "記憶體 RAM"],
            'Power': [15, "電源供應器"],
            'GPU': [12, "顯示卡 VGA"],
            'Storage': [7, "固態硬碟 M.2｜SSD"],
            'Case': [14, "機殼 CASE"],
            'Fan': [16, "機殼 風扇"]
        }
        
        self.planning_agent = Planning_Agent('planning agent', model, require, budget)
        
        ### Build component agents ###
        
        component_name_list = ['Mother board', 'Case', 'CPU', 'GPU', 'Memory', 'Storage', 'Power', 'Fan']
        for component_name in component_name_list:
            
            product_indext, class_name = coolpc_dict[component_name]
            price_list = self.item_crawler(product_indext, class_name)
            agent = Component_Agent(component_name, price_list, model, require, budget)
            self.agent_dict[component_name] = agent
            
        ### Build goalkeepers agents ###
        
        self.reducing_agent = Reducing_Agent('Reducing_Agent', model, require, budget)
    
        replacing_power_prompt ="""
            針對以下電腦的組合清單，請計算電源供應是否足夠：

            1. 計算除了機殼與電源供應器之外，所有電腦物件「需要的電源瓦數」，加總成一個總電源瓦數。
            2. 查詢電源供應器的「供應瓦數」
            3. 檢查「需要的電源瓦數」是否低於「供應瓦數」，如果是，則回答「電源供應充足」，如果不是，請更換其中一個電腦零件，使得「需要的電源瓦數」，可以低於電源供應器的「供應瓦數」
            3.1  請指出哪一個種類的產品可以被替換，推薦的一個需要更低電源瓦數的替代方案。
            範例：
            如果電源供應充足，回覆：
            [ 結論 ]：電源供應充足
            如果電源供應不充足，回覆：
            [ 結論 ]：電源供應不充足
            [ 替換產品 ] ： CPU
            [ 推薦原因 ] ： 耐用性強，可以使用到最新規格的Intel CPU...
            [ 產品名稱 ] ： Inter 第八代 i9
            [ 產品價格 ] ： 13000
            電腦清單：
            {component_list}

            最後一點請注意：回覆的文字內容一定要有[ 結論 ]
            """   
        self.power_agent = Power_Agent('Power_Agent', replacing_power_prompt, model)

        replacing_heat_dissipation_prompt ="""
            針對以下電腦的組合清單，請計算電腦的散熱是否足夠：

            1. 大概評估所有零件的發熱程度，用1~10分盡可能量化出來
            2. 大概評估風散與機殼的散熱程度，與1~10分盡可能量化出來
            3. 檢查「所有零件的發熱程度」是否低於「風散與機殼的散熱程度」，如果是，則回答「散熱充足」，如果不是，則回答「散熱不充足」，並請更換其中一個電腦零件，使得「所有零件的發熱程度」，可以低於的「風散與機殼的散熱程度」
            3.1  請指出哪一個種類的產品可以被替換，推薦的一個使得的替代方案。
            範例：
            如果散熱充足，回覆：
            [ 結論 ]：散熱充足
            如果散熱不充足，回覆：
            [ 結論 ]：散熱不充足
            [ 替換產品 ] ： CPU
            [ 推薦原因 ] ： 耐用性強，可以使用到最新規格的Intel CPU...
            [ 產品名稱 ] ： Inter 第八代 i9
            [ 產品價格 ] ： 13000
            電腦清單：
            {component_list}

            最後一點請注意：回覆的文字內容一定要有[ 結論 ]
            """   
        self.heat_dissipation_agent = Heat_Dissipation_Agent('Heat_Dissipation_Agent', replacing_heat_dissipation_prompt, model)
    
        ### Build debate agents ###
        
        self.debate_gpu_agents = Debate_Agents('debate', 'GPU', '微星', '技嘉', 'gpt-4o-mini', 2)
        self.debate_cpu_agents = Debate_Agents('debate', 'CPU', 'Intel', 'AMD', 'gpt-4o-mini', 2)
    
    def execute(self, component_dict):
        
        ### Distribute the budget ###
        
        component_dict = self.planning_agent({})
        
        ### Component selection ###
        
        for agent in self.agent_dict.values():
            
            component_dict = agent(component_dict)
            
        # ### Debate with component ###
        
        # component_dict = self.debate_cpu_agents(component_dict)
        # component_dict = self.debate_gpu_agents(component_dict)
            
        # ### Goalkeepers ###
        
        total_price = self.reducing_agent.summary_price(component_dict)
        difference = self.reducing_agent.budget - total_price
        accumulated_times = 0

        while difference < 0 and accumulated_times < 5:
            
            print(f'總預算：{self.reducing_agent.budget} | 清單價格：{total_price} | 差值：{difference}')
            
            replace_component = self.reducing_agent(component_dict)
            
            replace_component_price = component_dict[replace_component]['price']
            occupy_budget = total_price - float(replace_component_price)
            reset_budget = self.reducing_agent.budget - occupy_budget
            
            component_dict[replace_component] = {'price': reset_budget}
            component_dict = self.agent_dict[replace_component](component_dict)
            
            total_price = self.reducing_agent.summary_price(component_dict)
            difference = self.reducing_agent.budget - total_price
            
            accumulated_times += 1
        
        # component_dict = self.reducing_agent(component_dict)
        # component_dict = self.power_agent(component_dict)
        # component_dict = self.heat_dissipation_agent(component_dict)
        
        return component_dict
    
    def item_crawler(self, value, class_string):
        
        headers = {'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'}
        res = requests.get("https://coolpc.com.tw/evaluate.php", headers=headers)
        soup = BeautifulSoup(res.text, 'lxml')
        
        data_list = []  # 用於存儲清單

        for item in soup.select('#tbdy > tr:nth-child('+str(value)+')'):
            for opt in item.select('td:nth-child(3) > select'):
                for opt_item in opt.find_all(value=True, disabled=False):
                    total_result = re.sub(r"共有.*\n", "", opt_item.text, 0, re.MULTILINE)
                    blank_result = re.sub(r"^\s*\n", "", total_result, 0, re.MULTILINE)
                    if len(blank_result) != 0:
                        name_string = blank_result.split(',')[0]
                        price_string = blank_result.split("$").pop().split(" ")[0]
                        data_list.append({'class': class_string, 'name': name_string, 'price': price_string})
        
        # print(f'{class_string}_crawler')
        # print ( f'result_0:{data_list[0]}')
        if data_list:
            
            product_texts = []
            for p in data_list:
                product_info = f"品名：{p['name']}\n價格：{p['price']}元\n"
                product_texts.append(product_info)
            context_text = "\n".join(product_texts)
            return context_text
        else:
            return "查不到" 
        # return data_list


