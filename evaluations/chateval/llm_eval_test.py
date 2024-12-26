import os
os.environ["OPENAI_API_KEY"] = "***"

# always remember to put these lines at the top of your code if you are using clash
# os.environ["http_proxy"] = "http://127.0.0.1:7890"
# os.environ["https_proxy"] = "http://127.0.0.1:7890"
# os.environ["all_proxy"] = "socks5://127.0.0.1:7890"


import json
from eval_helper.get_evaluation import get_evaluation

from agentverse.agentverse import AgentVerse
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument("--config", type=str, default="config.yaml")
parser.add_argument("--reverse_input", default=False, action="store_true")


args = parser.parse_args()

agentverse, args_data_path, args_output_dir = AgentVerse.from_task(args.config)

print(args)

os.makedirs(args_output_dir, exist_ok=True)
with open(os.path.join(args_output_dir, "args.txt"), "w") as f:
    f.writelines(str(args))

# uncomment this line if you don't want to overwrite your output_dir
# if os.path.exists(args_output_dir) and len(os.listdir(args_output_dir)) > 1 :
#
#     raise ValueError("the output_dir is not empty, check if is expected.")

with open(args_data_path, encoding='utf-8') as f:
    data = json.load(f)

pair_comparison_output = []

for num, ins in enumerate(data[:10]):

    print(f"================================instance {num}====================================")

    # reassign the text to agents, and set final_prompt to null for debate at first round
    for agent_id in range(len(agentverse.agents)):
        agentverse.agents[agent_id].source_text = ins["question"]

        if args.reverse_input:
            agentverse.agents[agent_id].compared_text_one = ins["response"]["BuildMate"]
            agentverse.agents[agent_id].compared_text_two = ins["response"]["chatGPT o1"]
        else:
            agentverse.agents[agent_id].compared_text_one = ins["response"]["chatGPT o1"]
            agentverse.agents[agent_id].compared_text_two = ins["response"]["BuildMate"]

        agentverse.agents[agent_id].final_prompt = ""

    agentverse.run()

    evaluation = get_evaluation(setting="every_agent", messages=agentverse.agents[0].memory.messages, agent_nums=len(agentverse.agents))

    pair_comparison_output.append({"question": ins["question"],
                                    "response": {"chatGPT o1": ins["response"]["chatGPT o1"],
                                                "BuildMate": ins["response"]["BuildMate"]},
                                    "evaluation": evaluation})

    os.makedirs(args_output_dir, exist_ok=True)
    with open(os.path.join(args_output_dir, "pair_comparison_results_test.json"), "w") as f:
        json.dump(pair_comparison_output, f, indent=4)
    # with open(os.path.join(args_output_dir, "gt_origin_results.json"), "w") as f:
    #     json.dump(gt_origin_output, f, indent=4)
