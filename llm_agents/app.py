from typing import Literal
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langchain.schema.runnable.config import RunnableConfig
from langchain_core.messages import HumanMessage
from langchain_community.document_loaders import JSONLoader
from langchain.globals import set_debug, set_verbose

import json
import numpy as np
import logging
from typing import TypedDict, Annotated, Sequence, Type, Optional, List, Dict, Any
from pydantic import BaseModel, Field, PrivateAttr
from openai import OpenAI
from langchain.tools.base import BaseTool
from langgraph.prebuilt import ToolExecutor, ToolInvocation
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
import rich
import chainlit as cl
from langgraph.checkpoint.memory import MemorySaver
import asyncio
import faiss
import pickle
from langchain.callbacks.tracers.langchain import LangChainTracer
from answer_grader import answer_grader
from agents import Agent,print_results,parser_message,parser_replace_message,get_budget_from_string,get_component_list, component_prompt_maker, summary_price, Reducing_Agent
import yaml
import pickle

# 載入設定檔
def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

config = load_config()
    
memory = MemorySaver()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class GetComponentListInput(BaseModel):
    """電腦組裝清單推薦的輸入模型"""
    require: str = Field(description="使用者的電腦組裝需求，例如 '使用者總預算: 40000元，用途：遊戲機器，各零組件的分配到的預算為 CPU:8000元, Memory:4000元'")

class GetComponentPriceInput(BaseModel):
    """匯率查詢工具的輸入模型"""
    component: str = Field(description="要查詢價格的零組件名稱，例如 'CPU'")

class GetExchangeRateInput(BaseModel):
    """匯率查詢工具的輸入模型"""
    currency_pair: str = Field(description="要查詢匯率的貨幣 pair，例如 'USD/TWD'")

class GetAverageExchangeRateInput(BaseModel):
    """平均匯率查詢工具的輸入模型"""
    currency_pair: str = Field(description="要查詢匯率的貨幣 pair，例如 'USD/TWD'")
    period: str = Field(default="1 month", description="計算平均匯率的時間 period，例如 '1 month' 或 '3 months'")

class CalculateInput(BaseModel):
    """計算工具的輸入模型"""
    expression: str = Field(description="要計算的數學表達式，例如 '(31.5 - 31.0) / 31.0 * 100'")

# 工具實現
def item_crawler(value, class_string):
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
    
    print(f'{class_string}_crawler')
    print ( f'result_0:{data_list[0]}')
    if data_list:
        
        product_texts = []
        for p in data_list:
            product_info = f"品名：{p['name']}\n價格：{p['price']}元\n"
            product_texts.append(product_info)
        context_text = "\n".join(product_texts)
        return context_text
    else:
        return "查不到" 

    
# recommend tool
class GetRequireListRecommendationTool(BaseTool):
    """工具：獲取電腦零組件推薦清單"""
    name: str = "get_require_list_recommendation"
    description: str = "取得電腦組裝推薦清單。"
    args_schema: Type[BaseModel] = GetComponentListInput

    def _run(self, require: str) -> str:
        """執行工具邏輯"""
        try:
            component_dict = {'Mother board': '', '機殼': '', 'CPU': '', 'GPU': '', 
                  'Memory': '', 'Device': '', 'Power': '', 'Fan': ''}
            component_dict = {'Mother board': '', '機殼': '', 'CPU': '', 'GPU': '', 
                  'Memory': '', '硬碟/SSD': '', 'Power': '','Fan': ''}
            agent_dict = {}

            for component_name in component_dict.keys():
                if component_name in ["CPU"]:
                    price_list = item_crawler(4, "處理器 CPU")
                    
                elif component_name in ["Mother board", "motherboard"]:
                    price_list = item_crawler(5, "主機板 MB")
                elif component_name in ["Memory"]: 
                    price_list = item_crawler(6, "記憶體 RAM")
                elif component_name in ["Power"]: 
                    price_list = item_crawler(15, "電源供應器")
                elif component_name in ["顯示卡","顯卡","GPU"]: 
                    price_list = item_crawler(12, "顯示卡 VGA")
                elif component_name in ["硬碟/SSD"]: 
                    price_list = item_crawler(7, "固態硬碟 M.2｜SSD")
                elif component_name in ["機殼"]: 
                    price_list = item_crawler(14, "機殼 CASE")
                elif component_name in ["Fan"]: 
                    price_list = item_crawler(16, "機殼 風扇")

                budget = -1
                budget = get_budget_from_string(require)
                logger.debug(f'require: {require} budget: {budget}')
                component_prompt = component_prompt_maker(component_name, price_list, require)
                model = 'gpt-4o-mini-2024-07-18'
                agent = Agent(component_name, component_prompt, model, require, budget)
                agent_dict[component_name] = agent

            for agent_name, agent in agent_dict.items():
                logger.debug(agent_name)
                component_dict = agent.action(component_dict)
            
            rich.print(component_dict)
            
            # Reducing strategy
            logger.debug('Reducing strategy')
            reducing_prompt ="""
            你是一個專業的電腦銷售機器人，目前客戶遇到了一個難題，
            它有一組電腦的清單，清單中含有產品的名稱與價格，
            但清單的總價格超出了預算。
            
            身為專業的電腦銷售機器人，請把它挑出這個電腦清單中，
            哪一個產品可以被替換成更低價格的產品，從而使得整體的價格可以低於或者等於客戶的預算。
            
            客戶的需求為：{require}
            客戶的預算為：{budget}
            目前已經有的電腦組合清單為：{component_list}
            目前已經有的電腦組合清單的總價格：{total_price}
            距離客戶預算的差價：{difference_price}

            請指出哪一個種類的產品可以被替換，
            推薦的一個更低價格的替代方案。
            請說明推薦原因，並提供產品名稱，範例的回饋訊息如下：
            [ 替換產品 ] ： CPU
            [ 推薦原因 ] ： 耐用性強，可以使用到最新規格的Intel CPU...
            [ 產品名稱 ] ： Inter 第八代 i9
            [ 產品價格 ] ： 13000
            
            另外有幾點請注意：
            1. 推薦原因的說明請限制在50個字以內
            2. 產品的價格，請回覆一個完整的數字，中間不需要加入逗號
            """       

            reducing_agent = Reducing_Agent('Reducing_agent', reducing_prompt, model, require, budget)
            response = reducing_agent.action(component_dict)
            if response:
                component_dict = response
            # count = 0
            # while True:
            #     response = reducing_agent.action(component_dict)
            #     if response:
            #         component_dict = response
            #     else:
            #         break
            #     count +=1
            #     #print_results(component_dict)
            
            component_list = get_component_list(component_dict)
            total_price = summary_price(component_list)


            recommend_results = f'組合清單：{component_list}\n 總價格：{total_price}'
            rich.print(recommend_results)

    


            if recommend_results:
                return f"針對使用者推薦的電腦硬體組裝清單是： {recommend_results}"
            else:
                return f"電腦硬體組裝清單是 for {requirement} is not available"
        except Exception as e:
            logger.error(f"Error in GetComponentListRecommendationTool: {str(e)}")
            raise

    async def _arun(self, currency_pair: str) -> str:
        """異步執行（未實現）"""
        raise NotImplementedError("異步執行尚未實現")

class GetPriceTool(BaseTool):
    """工具：獲取當前零組件價格清單"""
    name: str = "get_component_price_list"
    description: str = "取得指定零組件類別的價錢清單。"
    args_schema: Type[BaseModel] = GetComponentPriceInput

    def _run(self, component: str) -> str:
        """執行工具邏輯"""
        try:
            if not component:
                raise ValueError("component 不能為空。")
            
            # component_prices = {
                
            #     "CPU": item_crawler(4, "處理器 CPU"),
            #     "主機板": item_crawler(5, "主機板 MB"),
            #     "motherboard": item_crawler(5, "主機板 MB"),
            #     "Motherboard": item_crawler(5, "主機板 MB"),
                
            # }
            if component.lower() in ["cpu"]:
                price_list = item_crawler(4, "處理器 CPU")
                
            elif component.lower() in ["主機板", "motherboard"]:
                price_list = item_crawler(5, "主機板 MB")
            elif component.lower() in ["ram"]: 
                price_list = item_crawler(6, "記憶體 RAM")
            elif component.lower() in ["power","電源", "電供","電源供應器", "電源供應"]: 
                price_list = item_crawler(15, "電源供應器")
            elif component.lower() in ["顯示卡","顯卡","gpu"]: 
                price_list = item_crawler(12, "顯示卡 VGA")
            elif component.lower() in ["儲存裝置", "ssd"]: 
                price_list = item_crawler(7, "固態硬碟 M.2｜SSD")
            elif component.lower() in ["機殼", "case"]: 
                price_list = item_crawler(14, "機殼 CASE")
            else:
                price_list = False
            
            if price_list:
                return f"[PRICE_TOOL_OUTPUT]\nThe current price list for {component} is {price_list}"
            else:
                return f"[PRICE_TOOL_OUTPUT]\nPrice list for {component} is not available"
        except Exception as e:
            logger.error(f"Error in GetPriceTool: {str(e)}")
            raise

    async def _arun(self, component: str) -> str:
        """異步執行（未實現）"""
        raise NotImplementedError("異步執行尚未實現")

class GetExchangeRateTool(BaseTool):
    """工具：獲取當前匯率"""
    name: str = "get_exchange_rate"
    description: str = "取得指定貨幣對的當前匯率。"
    args_schema: Type[BaseModel] = GetExchangeRateInput

    def _run(self, currency_pair: str) -> str:
        """執行工具邏輯"""
        try:
            if not currency_pair:
                raise ValueError("貨幣 pair 不能為空。")
            
            exchange_rates = {
                "USD/TWD": 31.5,
                "EUR/TWD": 34.2,
                "JPY/TWD": 0.21,
                "TWD/USD": 0.031,
            }
            rate = exchange_rates.get(currency_pair.strip(), None)
            if rate:
                return f"The current exchange rate for {currency_pair} is {rate}"
            else:
                return f"Exchange rate for {currency_pair} is not available"
        except Exception as e:
            logger.error(f"Error in GetExchangeRateTool: {str(e)}")
            raise

    async def _arun(self, currency_pair: str) -> str:
        """異步執行（未實現）"""
        raise NotImplementedError("異步執行尚未實現")

class GetAverageExchangeRateTool(BaseTool):
    """工具：獲取平均匯率"""
    name: str = "get_average_exchange_rate"
    description: str = "取得指定貨幣 pair 在特定時間段內的平均匯率。"
    args_schema: Type[BaseModel] = GetAverageExchangeRateInput

    def _run(self, currency_pair: str, period: str = "1 month") -> str:
        """執行工具邏輯"""
        try:
            if not currency_pair:
                raise ValueError("貨幣 pair 不能為空。")
            if not period:
                raise ValueError("時間 period 不能為空。")

            average_rates = {
                ("USD/TWD", "1 month"): 31.0,
                ("USD/TWD", "3 months"): 30.8,
                ("USD/TWD", "6 months"): 30.5,
                ("EUR/TWD", "1 month"): 33.8,
                ("EUR/TWD", "3 months"): 33.5,
                ("JPY/TWD", "1 month"): 0.208,
            }
            rate = average_rates.get((currency_pair.strip(), period.strip()), None)
            if rate:
                return f"The average exchange rate for {currency_pair} over {period} is {rate}"
            else:
                return f"Average exchange rate for {currency_pair} over {period} is not available"
        except Exception as e:
            logger.error(f"Error in GetAverageExchangeRateTool: {str(e)}")
            raise

    async def _arun(self, currency_pair: str, period: str = "1 month") -> str:
        """異步執行（未實現）"""
        raise NotImplementedError("異步執行尚未實現")

class CalculateTool(BaseTool):
    """工具：執行數學計算"""
    name: str = "calculate"
    description: str = "根據提供的數學表達式執行計算。例如：'(31.5 - 31.0) / 31.0 * 100' 計算漲幅百分比，或是 '3999 + 1891' 計算總價。"
    args_schema: Type[BaseModel] = CalculateInput

    def _run(self, expression: str) -> str:
        """執行工具邏輯"""
        try:
            if not expression:
                raise ValueError("表達式不能為空。")
            
            allowed_names = {"__builtins__": None}
            result = eval(expression.strip(), allowed_names, {})
            return f"計算結果為: {result}"
        except Exception as e:
            logger.error(f"Error in CalculateTool: {str(e)}")
            raise

    async def _arun(self, expression: str) -> str:
        """異步執行（未實現）"""
        raise NotImplementedError("異步執行尚未實現")


# 讀入 RAG 所需的 JSON
output_json_path =config['rag']['paths']['output_json']
print (output_json_path)
loader = JSONLoader(
    file_path = output_json_path,
    jq_schema = '.text',
    json_lines=True, text_content=True)
documents = []
documents_load = loader.load()
for d in documents_load:
    documents.append(d.page_content)



class RAGRetrieveInput(BaseModel):
    query: str
    top_k: int = config['rag']['retrieval']['top_k'] # 默認返回前 20 個結果

# 定義 RAGRetrieveTool
with open(config['rag']['paths']['embedding_data'], 'rb') as f:
    embeddings = pickle.load(f)
    embeddings = np.array(embeddings, dtype=np.float32)
    
    # embeddings shape: (len(data_list), 1536)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)


# 定義 RAGRetrieveTool
class RAGRetrieveTool(BaseTool):
    """工具：使用 Faiss 和 OpenAI 的嵌入模型進行論壇檢索。"""
    name: str = "rag_retrieve"
    description: str = "根據用戶的查詢，檢索論壇最相關的文檔。"
    args_schema: Type[RAGRetrieveInput] = RAGRetrieveInput  # 正確的類型註解

    # 使用 PrivateAttr 定義私有屬性，名稱以單個下劃線開頭
    _index: faiss.IndexFlatL2 = PrivateAttr()
    _documents: List[str] = PrivateAttr()

    def __init__(self, index: faiss.IndexFlatL2, documents: List[str]):
        super().__init__()
        self._index = index
        self._documents = documents





    def _run(self, query: str, top_k: int = 20) -> str:
        """同步執行 RAG 檢索。"""
        try:
            # 生成查詢的嵌入向量
            response = OpenAI().embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)

            # 搜索最相似的文檔
            distances, indices = self._index.search(query_embedding, top_k)
            results = [self._documents[i] for i in indices[0]]

            # 將結果轉換為 JSON 字符串
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            print(f"Error in RAGRetrieveTool (sync _run): {str(e)}")
            raise

    async def _arun(self, query: str, top_k: int = 20) -> str:
        """異步執行 RAG 檢索"""
        try:
            # 生成查詢的嵌入向量
            response = OpenAI().embeddings.create(
                input=query,
                model="text-embedding-3-small"
            )
            query_embedding = np.array([response.data[0].embedding], dtype=np.float32)

            # 搜索最相似的文檔
            distances, indices = self._index.search(query_embedding, top_k)
            results = [self._documents[i] for i in indices[0]]

            # 將結果轉換為 JSON 字符串
            return json.dumps(results, ensure_ascii=False)
        except Exception as e:
            print(f"Error in RAGRetrieveTool: {str(e)}")
            raise

class AgentToolkit:
    """Agent Toolkit"""
    
    def __init__(self):
        """初始化 Toolkit"""
        self.tools: List[BaseTool] = [
            GetExchangeRateTool(),
            GetAverageExchangeRateTool(),
            CalculateTool(),
            GetPriceTool(),
            RAGRetrieveTool(index=index, documents=documents),
            GetRequireListRecommendationTool()
        ]

    def get_tools(self) -> List[BaseTool]:
        """獲取 tool list"""
        return self.tools

exchange_toolkit = AgentToolkit()
tools = exchange_toolkit.get_tools()
headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.60 Safari/537.36'}
#tools = [get_weather]
model = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0)
final_model = ChatOpenAI(model_name="gpt-4o-mini-2024-07-18", temperature=0)

model = model.bind_tools(tools)
# NOTE: this is where we're adding a tag that we'll can use later to filter the model stream events to only the model called in the final node.
# This is not necessary if you call a single LLM but might be important in case you call multiple models within the node and want to filter events
# from only one of them.
final_model = final_model.with_config(tags=["final_node"])
tool_node = ToolNode(tools=tools)

from typing import Annotated
from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph, START
from langgraph.graph.message import MessagesState
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage

from typing import List, TypedDict
RETRIEVE = "retrieve"
GRADE_DOCUMENTS = "grade_documents"
GENERATE = "generate"
WEBSEARCH = "websearch"

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: bool
    documents: List[str]

def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    filtered_docs = []
    web_search = False
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score.binary_score
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = True
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def should_continue(state: MessagesState) -> Literal["tools", "final"]:
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return "final"

def grade_generation_grounded_in_documents_and_question(state: MessagesState) -> str:
    messages = state["messages"]
    last_message = messages[-1]
    user_question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    
    if not user_question:
        print("---WARNING: No user question found---")
        return "useful"  # 如果找不到問題，預設返回 useful

    score = answer_grader.invoke({"question": user_question, "generation": last_message})
    feedback_message = f"""
    評分回饋：
    {score.feedback}

    改進建議：
    {score.suggestions}
    """
    
    state["messages"].append(
        SystemMessage(content=f"Previous response's feedback: {score.feedback} Please consider this feedback in your response.")
    )
    if score.binary_score:
        print("---DECISION: GENERATION ADDRESSES QUESTION---")
        return "useful"
    else:
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return "not useful"
    pass

def initialize_system_prompt(state: MessagesState):
    
    if not any(isinstance(msg, dict) and msg.get("role") == "system" for msg in state["messages"]):
        system_message= """我是一個專業的電腦零組件顧問，我的名字是 BuildMate，我只回答和電腦零組件相關的問題。我能夠：
    1. 查詢零組件的即時價格 （使用原價屋查詢系統）
    2. 搜尋相關的論壇討論文章  (PTT, Mobile01)
    3. 計算總價等數學運算
    4. 根據預算和需求提供購買清單建議
    5. 進行數學運算，例如計算商品總價，打折價，商品價格與使用者預算的差額。
    
    注意： 在使用電腦零組件清單推推薦清單工具時：
    - 當使用者有給整台電腦的總預算要詢問清單推薦時，你要先做預算的分配，將總預算分配至以下零組件：
        - Mother board
        - 機殼
        - CPU
        - GPU
        - Memory
        - 硬碟/SSD
        - Power
        - Fan (機殼)
    - 不同用途的電腦，各零組件的預算分配的比例也會有所不同！
    - 各零組件分配到的預算加總後要小於使用者的總預算!
    - 使用者的預算要用數字顯示，如：50000


    注意： 在使用零組件價格查詢工具時：
    - 查詢 CPU 時，應該只使用關鍵字 "CPU"，不要加上品牌名稱
    - 查詢 GPU 時，應該只使用關鍵字 "GPU"，不要加上品牌名稱，例如：ASUS
    - 查詢主機板時，使用 "主機板" 或 "motherboard" 作為關鍵字
    - 避免使用具體的型號名稱，因為價格查詢工具只接受通用類別

    例如：
    ✅ 正確的查詢：
    - component: "CPU"
    - component: "主機板"

    ❌ 避免的查詢：
    - component: "AMD CPU"
    - component: "Intel i9"
    - component: "華碩主機板

    注意： 使用論壇查詢工具查回的資訊要用分別由不同網站來源做整理，如 Mobile01 來源， PTT 來源。

        """
        state["messages"].append({"role": "system", "content": system_message})
    
    return state

def call_model(state: MessagesState):
    #messages = state["messages"]
    #rich.print ('Before clean call_model:\n', state["messages"])

    messages = filter_messages(state["messages"])
    #rich.print ('After clean call model:\n', state["messages"])
    response = model.invoke(messages)
    
    # state["messages"] = [
    #     msg for msg in state["messages"]
        
    #     if not (isinstance(msg, BaseMessage) and "[PRICE_TOOL_OUTPUT]" in msg.content)
    # ]
    # rich.print ('After clean call model:\n', state["messages"])
    # We return a list, because this will get added to the existing list
    return {"messages": [response]}

def filter_messages(messages: list):
    # This is a simple helper function which only ever uses the last six messages
    if len(messages) > 10:
        return messages[-10:]
    else:
        return messages

def call_final_model(state: MessagesState):
    messages = state["messages"]
    last_ai_message = messages[-1]
    response = final_model.invoke(
        [
            SystemMessage("""
        你是一名專業的電腦零組件組裝銷售顧問，請用專業，友善與熱情的口吻重新表達。"""),
            HumanMessage(last_ai_message.content),
        ]
    )
    # overwrite the last AI message from the agent
    rich.print(response.id)
    response.id = last_ai_message.id
    return {"messages": [response]}

def call_final_model_2(state: MessagesState):
    rich.print('call_final_model_2.....')
    messages = state["messages"]
    last_ai_message = messages[-1]
    print("\n=== Messages in call_final_model_2 ===")
    for i, msg in enumerate(messages):
        print(f"\nMessage {i}:")
        print(f"Type: {type(msg)}")
        print(f"Content: {msg.content}")
        if hasattr(msg, "role"):
            print(f"Role: {msg.role}")
    print("=====================================\n")
    
    # 找出最後一個 grader 的評估結果
    grader_result = None
    for msg in reversed(messages):
        if (isinstance(msg, SystemMessage) and 
            msg.content.startswith("Previous response's feedback:")):
            grader_result = msg.content
            break
    rich.print(grader_result)    
    response = final_model.invoke(
        [
            SystemMessage(f"""
            你是一名專業的電腦零組件組裝銷售顧問，請用英語重新表達。
   
            注意：
            - 若有產品的價錢資訊，請記得提供給使用者。
            - 請直接依據 feedback 和 suggestions 做 revise /update ，並回應給使用者就好；不用提到你是依據 feedback & suggestions 做 revise 或 update.
            - 不要提到類似的句子：Certainly! Below is the revised response with the requested improvements
            - 論壇查詢工具查回的資訊要用分別由不同網站來源做整理，如 Mobile01 來源， PTT 來源。

            """),
            # Grader的評估結果：{grader_result if grader_result else "無"}
            HumanMessage(last_ai_message.content),
        ]
    )
    response.id = last_ai_message.id
    return {"messages": [response]}

def process_grader_result(state: MessagesState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # 找出最後一個 HumanMessage
    user_question = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_question = msg.content
            break
    
    score = answer_grader.invoke({
        "question": user_question,
        "generation": last_message.content
    })
    
    # 返回更新後的訊息
    return {
        "messages": [SystemMessage(content=f"Previous response's feedback: {score.feedback}. Suggestions: {score.suggestions}. Please consider this feedback and the suggestionins your response.")],
        "grader_result": "useful" if score.binary_score else "not useful"
    }

def update_state_with_feedback(state: MessagesState, grader_feedback: str) -> MessagesState:
    state["messages"].append(
        SystemMessage(content=f"Previous response's feedback: {grader_feedback}. Please consider this feedback in your response.")
    )
    return state

def route_based_on_grader(state: MessagesState) -> str:
    return state["grader_result"]

builder = StateGraph(MessagesState)
builder.add_node("initialize_system_prompt", lambda state: initialize_system_prompt(state))
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)

builder.add_node("process_grader", process_grader_result)  # 新增節點

builder.add_node("final", call_final_model)
builder.add_node("final_2", call_final_model_2)

builder.add_edge(START, "initialize_system_prompt")
builder.add_edge("initialize_system_prompt", "agent")


builder.add_conditional_edges(
    "agent",
    should_continue,
)

builder.add_edge("tools", "agent")
builder.add_edge("final", "process_grader") 
builder.add_conditional_edges(
    "process_grader",
    route_based_on_grader,
    {
        "useful": "final_2",
        "not useful": "final_2",
    },
)

builder.add_edge("final_2", END)

graph = builder.compile(checkpointer=memory)




@cl.on_message
async def on_message(msg: cl.Message):
    config = {"configurable": {"thread_id": cl.context.session.id}}
    cb = cl.LangchainCallbackHandler()
    final_answer = cl.Message(content="")
    
    for msg, metadata in graph.stream({"messages": [HumanMessage(content=msg.content)]}, stream_mode="messages", config=RunnableConfig(callbacks=[cb], **config)):
        if (
            msg.content
            and not isinstance(msg, HumanMessage)
            and metadata["langgraph_node"] in [ "final_2"]
        ):
            await final_answer.stream_token(msg.content)

    await final_answer.send()
