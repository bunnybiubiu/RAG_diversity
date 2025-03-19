import os
import requests
import pandas as pd
from datasets import load_dataset

def get_nq():
    # URL to the raw file
    url = "https://raw.githubusercontent.com/google-research-datasets/natural-questions/master/nq_open/NQ-open.efficientqa.dev.1.1.jsonl"
    os.makedirs("datasets", exist_ok=True)
    
    if not os.path.exists("datasets/NQ-open.efficientqa.dev.1.1.jsonl"):
        response = requests.get(url)
        
        if response.status_code == 200:
            with open("datasets/NQ-open.efficientqa.dev.1.1.jsonl", "wb") as f:
                f.write(response.content)
            print("File downloaded successfully.")
        else:
            print(f"Failed to download file. Status code: {response.status_code}")
    nq_ds = pd.read_json("datasets/NQ-open.efficientqa.dev.1.1.jsonl", lines=True)
    return nq_ds



def get_tqa():
    if not os.path.exists("datasets/raw_data/trivia_qa.json"):
        tqa = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation")
        tqa.to_json("datasets/raw_data/trivia_qa.json")
    
    # create processed tqa dataset
    if not os.path.exists("tqa.json"):
        tqa = pd.read_json("datasets/raw_data/trivia_qa.json", lines=True)
        data = pd.DataFrame()
        data["question"] = tqa["question"]
        data["answer"] = tqa.loc[:,"answer"].apply(lambda x: x["aliases"])
        data.to_json("datasets/tqa.json", index=False)
    
    tqa_ds = pd.read_json("datasets/tqa.json")
    return tqa_ds



def get_squad():
    if not os.path.exists("datasets/raw_data/squad_dev-v1.1.json"):
        squad = load_dataset("squad", split="validation")
        squad.to_json("datasets/raw_data/squad_dev-v1.1.json")
        
    if not os.path.exists("datasets/squad.csv"):
        squad = pd.read_json("datasets/raw_data/squad_dev-v1.1.json", lines=True)
        data = pd.DataFrame()
        data["question"] = squad["question"]
        data["answer"] = squad["answers"].apply(lambda x: x["text"])
        data.to_json("datasets/squad.json", index=False)
        
    squad_ds = pd.read_json("datasets/squad.json")
    return squad_ds



def get_asqa():
    if not os.path.exists("datasets/asqa_dev.json"):
        asqa = load_dataset("din0s/asqa")
        asqa_dev = asqa["dev"]
        data = []
        for item in asqa_dev["qa_pairs"]:
            for qa_pair in item:
                q = qa_pair["question"]
                a = qa_pair["short_answers"]
                data.append({"question":q, "answer":a})
        data = pd.DataFrame(data)
        data.to_json("datasets/asqa_dev.json", index=False)
    
    asqa_ds = pd.read_json("datasets/asqa_dev.json")
    return asqa_ds
    

    
