import openai
#import gradio as gr
from langchain import LLMChain, OpenAI, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import logging
import os
import re
import requests
from bs4 import BeautifulSoup
#os.environ['OPENAI_API_KEY'] = 'Enter OpenAI Key'
from langchain.globals import set_debug, set_verbose
set_verbose(True)
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import rich
class Agent():
    
    def __init__(self, agent_name, prompt, model, require, budget):
        
        self.agent_name = agent_name
        self.memory = []
        self.require = require
        self.budget = budget
        
        
        template = ChatPromptTemplate.from_template(prompt)
        #rich.print(template)
        model = ChatOpenAI(model=model, temperature=0.7)
        parser = StrOutputParser()

        self.llm = template | model | parser
        
    def receive_message(self, messages):
        self.memory += messages
        
    def send_message(self, content, receive_obj):
        pass
    
    def action(self, component_dict):
        
        component_list = self.get_component_list(component_dict)
        
        user_message = {}
        user_message['component_list'] = component_list
        user_message['require'] = self.require
        user_message['budget'] = self.budget
        
        message = self.llm.invoke(user_message)
        self.memory.append(message)
        print (self.agent_name, message)
        component_dict[self.agent_name] = self.parser_message(message)
        
        return component_dict
        
    def check_target(self, target):
        pass
    
    def end_stage(self):
        pass
    
    def get_component_list(self, component_dict):

        component_list = ''

        for i, (key, value) in enumerate(component_dict.items()):
            
            num = i + 1
            if value:
                component_list += f'{num}. {key}: {value["name"]} | price: {value["price"]}\n'
            else:
                component_list += f'{num}. {key}: None | price: None \n'

        return component_list
    
    def parser_message(self, message):
        logging.debug(f'message: {message}, {type(message)}' )
        pattern = {
            "reason": r"\[ 推薦原因 \] ：\s*(.*?)\s*\n",
            "name": r"\[ 產品名稱 \] ：\s*(.*?)\s*\n",
            "price": r"\[ 產品價格 \] ：\s*(\d+)"
        }

        results = {key: re.search(regex, message).group(1) for key, regex in pattern.items()}

        return results

class Reducing_Agent(Agent):
    
    def __init__(self, agent_name, prompt, model, require, budget):
        super().__init__(agent_name, prompt, model, require, budget)

    def action(self, component_dict):
        
        component_list = self.get_component_list(component_dict)
        total_price = self.summary_price(component_list)
        
        difference_price = self.budget - total_price
        
        if difference_price > 0:
            return False
        else: 
            user_message = {}
            user_message['component_list'] = component_list
            user_message['total_price'] = total_price
            user_message['difference_price'] = difference_price
            user_message['require'] = self.require
            user_message['budget'] = self.budget
            
            message = self.llm.invoke(user_message)
            self.memory.append(message)
        
            replace_component, results = self.parser_replace_message(message)
            component_dict[replace_component] = results
        
            return component_dict
    
    def summary_price(self, component_list):
        
        price_pattern = r"price: (\d+)"

        prices = [int(price) for price in re.findall(price_pattern, component_list)]

        return sum(prices)

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

def get_component_list(component_dict):
    print('get_component_list...')
    component_list = ''

    for i, (key, value) in enumerate(component_dict.items()):
        
        num = i + 1
        if value:
            component_list += f'{num}. {key}: {value["name"]} | price: {value["price"]}\n'
        else:
            component_list += f'{num}. {key}: None | price: None \n'

    return component_list

def parser_message(message):
    pattern = {
        "reason": r"\[ 推薦原因 \] ：\s*(.*?)\s*\n",
        "name": r"\[ 產品名稱 \] ：\s*(.*?)\s*\n",
        "price": r"\[ 產品價格 \] ：\s*(\d+)"
    }
    # albert 修改的版本，因為 LLM 的輸出有時會缺少空格
    patterns = {
    "reason": r"\[\s*推薦原因\s*\]([^@]+)",
    "name": r"\[\s*產品名稱\s*\] ：\s*(.*?)\s*\n",
    "price": r"\[\s*產品價格\s*\] ：\s*(\d+)"
    }   
    try:
        results = {key: re.search(regex, message).group(1) for key, regex in pattern.items()}
    except Exception:
        print(f"Exception in parser_message: {parser_message}")
        results = 'not found'
    return results

def summary_price(component_list):
    print ('summary_price...')
    try:
        price_pattern = r"price: (\d+)"

        prices = [int(price) for price in re.findall(price_pattern, component_list)]
        sum_prices = sum(prices)
    except Exception:
        sum_prices = "sum_prices not found"
        print('Exception in summary_price')
    return sum_prices

def parser_replace_message(message):
    
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

def component_prompt_maker(component_name, context_text, require):

    prompt = f"""
    你是一個專業的電腦-{component_name}銷售機器人，請根據你的專業，
    根據客戶的需求，提供你覺得客戶可能需要的電腦-{component_name}給他。
    挑選的 {component_name} 的價格要非常近使用者在 {component_name} 的預算分配。
    """
    
    prompt = prompt + """
    客戶整台電腦的總預算為：{budget}
    客戶的需求為：{require}
    目前已經有的電腦組合清單為：{component_list}

    請說明推薦原因與產品介紹，並提供產品名稱，範例的回饋訊息如下，請確保格式完全一致：
    [ 零組件類別 ] ： CPU
    [ 推薦原因 ] ： 耐用性強，可以使用到最新規格的Intel CPU，與使用者在 CPU 的預算上僅差距 500元...
    [ 產品名稱 ] ： Intel 第八代 i9 8C/16T 
    [ 產品價格 ] ： 13000
    [ 產品價格與使用者預算分配的價差 ] ： 500
    
    另外有幾點請注意：
    1. 推薦產品要和已有的電腦組合清單相容！
    2. 推薦原因的說明請限制在50個字以內
    3. 產品的價格，請回覆一個完整的數字，中間不需要加入逗號

    """

    prompt = prompt + f"""
    {component_name} 的報價資訊如下，請參考報價單裡的產品價格。
    注意！不是 {component_name} 的不要挑選。
    注意！挑選的 {component_name} 的價格要非常接近使用者在 {component_name} 的預算分配。
    零組件報價：\n{context_text}\n
    """

    return prompt

def get_budget_from_string(text):
    
    text = text.split('。')[0] # 取出第一句話
    match = re.search(r'\d+', text) # 取數字
    if match:
        budget = int(match.group())
    else:
        print("Not found the budget")
        budget = "Not found the budget!"
    
    return budget

def print_results(component_dict):
    component_list = get_component_list(component_dict)
    print(f'組合清單：{component_list}')

    total_price = summary_price(component_list)
    print(f'總價格：{total_price}')