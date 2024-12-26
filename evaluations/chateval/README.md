# 🚀 Getting Started

## Installation

```bash
git clone https://github.com/chanchimin/ChatEval.git
cd ChatEval
pip install -r requirements.txt
```

We basically call the OpenAI's API for our LLMs, so you also need to export your OpenAI key as follows before running our code.

## Using Environment Variable

### On Linux/macOS:
```bash
export OPENAI_API_KEY="your_api_key_here"
```

### On Windows:
```powershell
set OPENAI_API_KEY "your_api_key_here"
```

### Or, directly specifying in the Python file:
```python
# llm_eval_test.py
import os
os.environ["OPENAI_API_KEY"] = "your_api_key_here"
```

### Prepare Dataset

You can provide the evaluation examples in `test_data/test_o1_preview_results.json`.

Your custom data should look like

```json
[
    {
        "question_id": 1,
        "question": "Please provide me with a computer build list for training deep learning models on a 100k budget.",
        "response": {
            "chatGPT o1": "
            ....,
            ",
            "BuildMate": "
            .....
            "
        }
    }
]
```

The `test_o1_preview_results.json` file should contain a list of evaluation examples, each with a unique `question_id`, a `question`, and a `response` object. The `response` object should include the responses from different agents for the given question.

### Configuration

The `agentverse/tasks/llm_eval/config.yaml` file contains the configuration for the evaluation task. It specifies the data path, output directory, prompts, environment settings, and agent details. Make sure to review and update the configuration as needed.

- `data_path`: Path to the test data file. Modify this to point to your test data.
- `output_dir`: Directory where the output results will be saved.
- `prompts`: Contains the prompt template used for the evaluation.
  - `[System]`: Provides instructions to the agents on how to evaluate the responses, focusing on criteria such as practicality, performance, cost-efficiency, compatibility, and completeness of the recommendations.
- `environment`: Defines the environment settings such as type, max turns, and rules.
- `agents`: Specifies the agents involved in the evaluation, including their types, names, roles, and LLM configurations.
  - `Hard Critic`: Focuses on whether the PC build recommendations meet the purpose of the question, fit within the given budget, consider power consumption, and evaluate the quality of the cooling solution.
  - `Soft Critic`: Focuses on criteria such as component longevity, warranty coverage, aesthetics, brand reputation, and energy efficiency.

### Run the scripts

Now, you are good to run the experiments.
Try out the following lines first, it employs **one-by-one communication** and **2 agent roles** for **2 discussion turns** in the paper.
```shell
python llm_eval_test.py --config agentverse/tasks/llm_eval/config.yaml
```

