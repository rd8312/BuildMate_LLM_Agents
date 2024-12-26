# Multi-Agent PC Build Recommendation Evaluation System

## Overview

A multi-agent system for evaluating PC build recommendations using three specialized agents and a coordinator. The system conducts multiple rounds of evaluation to achieve consensus on recommendation quality.

## System Components

### Evaluation Agents

#### Agent1 (Performance & Stability Expert)

- Evaluates overall system performance
- Assesses component compatibility
- Reviews cooling solutions
- Checks power supply quality
- Analyzes system balance

#### Agent2 (Price-Performance Expert)

- Evaluates cost-effectiveness
- Assesses component pricing
- Reviews budget allocation
- Analyzes value proposition

#### Agent3 (Requirements Expert)

- Matches components to user requirements
- Verifies budget compliance
- Assesses component suitability
- Reviews special requirements

### Coordinator

- Manages evaluation process
- Determines consensus
- Controls evaluation rounds
- Makes final decisions



# Usage

## Initialize the system
```
input_recommendation_file = './recommendations.json'
output_report = "evaluation_report.json"
```

## Load recommendations
```
with open(input_recommendation_file, "r", encoding="utf-8") as f:
    recommendation_data = json.load(f)
```
## Choose LLM backend
```
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3)
```
## or
```
llm = ChatOpenAI(model="gpt-4", temperature=0.3)
```
## Create workflow
```
app = create_workflow(llm)
```


