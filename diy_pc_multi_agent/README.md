# PC DIY Multi-Agent System

### Multi-Agent Graph
The repository is based on a multi-agent system.

It's like a Tensorflow and Pytorch mechanism to flow the data on a graph.

Because it depends on a structure to transmit the data. 

### Installation
```
conda create --name pc_diy --file requirements.txt
```

### Agent building

Build an agent with the __init__ and forward implement.

Or any functions you need.

1. __init__: Define the variables

2. forward: Design how to operate the data

There is an example.

```
class Component_Agent(Agent):
    
    def __init__(self, component_name, model, require, budget):
        
        component_prompt = self.component_prompt_maker(component_name)
        super().__init__(component_name, component_prompt, model)
        
        self.require = require
        self.budget = budget
        
    def forward(self, component_dict):
        
        component_list = self.get_component_list(component_dict)
        
        user_message = {}
        user_message['component_list'] = component_list
        user_message['require'] = self.require
        user_message['budget'] = self.budget
        
        message = self.llm.invoke(user_message)
        self.memory.append(message)
        
        component_dict[self.agent_name] = self.parser_message(message)
        
        return component_dict
        
```

### Multi-agent system

Design the cooperate pattern with agents.

1. __init__: Define the variables

2. execute: Design the architecture

It shows how using the agent after building a multi-agent object.

```
pc_diy = PC_DIY_System(model, require, budget)

component_dict = {'Mother board': '', 'Case': '', 'CPU': '', 'GPU': '', 
                'Memory': '', 'Device': '', 'Power': '', 'Fan': ''}

component_dict = pc_diy.execute(component_dict)
print_results(component_dict)

# 組合清單：
# 1. Mother board: ASUS ROG Strix B550-F Gaming | price: 8590
# 2. Case: Cooler Master MasterBox Q300L | price: 2990
# 3. CPU: AMD Ryzen 7 5800X | price: 11500
# 4. GPU: NVIDIA GeForce GTX 1660 Super | price: 12000
# 5. Memory: Corsair Vengeance LPX 16GB (2 x 8GB) DDR4 3200MHz | price: 3500
# 6. Device: Corsair RM750 750W 80 Plus Gold | price: 4000
# 7. Power: Noctua NH-U12S Redux CPU Cooler | price: 2900
# 8. Fan: Cooler Master SickleFlow 120 V2 風扇 | price: 750

# 總價格：46230
```

### Component_dict

The component_dict is saved in a component_dict (json)

```python
{
    '品項': {
        'reason': '推薦原因', 
        'name': '產品名稱', 
        'price': '價格'
        }
}
### An example
{
    'Mother board': {
        'reason': '支援最新的AMD CPU與PCIe 4.0，適合高效能遊戲需求。',
        'name': 'ASUS ROG Strix B550-F Gaming',
        'price': '5999'
        },

    'Case': {
        'reason': '這款機殼擁有優秀的散熱性能，且設計美觀，適合高效能遊戲組合。',
        'name': 'Cooler Master MasterBox Q300L',
        'price': '1890'
        }
  ...
}
```


