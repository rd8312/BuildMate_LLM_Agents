import re
import json

def get_component_list(component_dict):

    component_list = ''

    for i, (key, value) in enumerate(component_dict.items()):
        
        num = i + 1
        if value:
            component_list += f'{num}. {key}: {value["name"]} | price: {value["price"]}\n'
        else:
            component_list += f'{num}. {key}: None | price: None \n'

    return component_list

def get_budget_from_string(text):
    
    text = text.split('。')[0] # 取出第一句話
    match = re.search(r'\d+', text) # 取數字
    if match:
        budget = int(match.group())
    else:
        print("Not found the number")
    
    return budget

def summary_price(component_dict):

    return sum([int(component['price']) for component in component_dict.values()])

def print_results(component_dict):
    component_list = get_component_list(component_dict)
    print(f'組合清單：\n{component_list}')

    total_price = summary_price(component_dict)
    print(f'總價格：{total_price}')