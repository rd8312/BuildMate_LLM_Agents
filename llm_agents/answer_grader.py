from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI


class GradeAnswer(BaseModel):

    binary_score: bool = Field(
        description="回答是否解決問題，'yes'或 'no'"
    )
    feedback: str = Field(description="關於答案優點和改進空間的詳細反饋")
    suggestions: str = Field(description="具體建議如何改進或擴展答案（如有需要")


llm = ChatOpenAI(temperature=0, model='gpt-4o-2024-11-20')
structured_llm_grader = llm.with_structured_output(GradeAnswer)


system = """你是一名電腦硬體組裝評分員，負責評估回答裡的產品組合是否相容，或是是否有回答到使用者的問題\n 給予二進制分數'yes'或 'no'。 
'yes'表示產品組合相容，或是有回答到使用者的問題。
總是提供建設性的反饋和建議，即使對於好的答案也是如此。"""
answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "User question: \n\n {question} \n\n LLM generation: {generation}"),
    ]
)

answer_grader: RunnableSequence = answer_prompt | structured_llm_grader