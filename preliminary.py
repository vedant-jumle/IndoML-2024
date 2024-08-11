import os
import ollama
import openai
import json
import random
import dotenv
import regex as re
import pandas as pd
import time
import enlighten

from tqdm import tqdm
from typing import Union, List, Any
from threading import Thread
from yaml import safe_load_all, safe_load

dotenv.load_dotenv()

system_prompt = """You are an helpful assistant who only answers in JSON and nothing else.
You do not answer in Markdown, and not use " or ' within strings"""

user_prompt = """Task: Attribute-Value Prediction From E-Commerce Product Descriptions
Example 1:
{example_1}

Example 2:
{example_2}

Example 3:
{example_3}

Based on the above examples:

I want you to help me find attribute-value from the following description:
Product data:
Title: {title}
Store: {store}
Manufacturer: {details_Manufacturer}
Attribute-values:
"""

examples_prompt = """Product data:
Title: {title}
Store: {store}
Manufacturer: {details_Manufacturer}
Attribute-values:
{attribute_values}"""

def get_client(service="ollama") -> Union[ollama.Client, openai.Client]:
    assert service in ['ollama', 'openai'], "'service' should be either 'ollama' or 'openai'"

    if service == "ollama":
        return ollama.Client(host=os.environ['OLLAMA_ENDPOINT'],)
    
    if service == "openai":
        return openai.Client(api_key=os.environ['OPENAI_API_KEY'])
    
def call_llm(messages: list[dict],
             model:str='llama3.1',
             temperature:float=0.2,
             top_k:Union[None, int]=None) -> str:

    if 'gpt' in model:
        # call openai
        response = get_client('openai').chat.completions.create(
            messages=messages,
            model=model,
            temperature=temperature,
        ).choices[0].message.content
    
    else:
        response = get_client('ollama').chat(
            model=model,
            messages=messages,
            options={
                "top_k": top_k,
                "temperature": temperature
            }
        )['message']['content']

    return response

def parse_json(input_string:str) -> dict:
    # extract JSON patterned string
    # to understand this regex: https://regex101.com/r/to8x5X/1
    pattern = re.compile(r'\{(?:[^{}]|(?R))*\}')

    # load JSON using python's JSON utility
    return safe_load(pattern.findall(input_string)[0])

def load_jsonl(path):
    with open(path) as input_file:
        lines = list(map(json.loads, input_file))

    return lines

def load_data(root_path):
    files = os.listdir(root_path)
    dataset= {}

    for filename in files:
        filepath = root_path + f'/{filename}'

        dataset[filename] = load_jsonl(filepath)

    return dataset

def get_random_example(inputs, labels, n=3):
    idx = random.choices(list(range(len(inputs))), k=n)

    random_inputs = [inputs[i] for i in idx]
    random_labels = [labels[i] for i in idx]

    return random_inputs, random_labels

def prepare_prompts(question):
    reference_inputs, reference_labels = get_random_example(train_inputs, train_labels)

    examples = {}
    select_l = ['details_Brand', 'L0_category', 'L1_category', 'L2_category', 'L3_category', 'L4_category']

    for i, (inputs, labels) in enumerate(zip(reference_inputs, reference_labels)):
        labels = str({item: labels[item] for item in select_l})

        examples[f'example_{i+1}'] = examples_prompt.format(
            attribute_values=labels, **inputs
        )

    return user_prompt.format(**examples, **question)

if __name__ == '__main__':
    dataset = load_data('data/L1')

    models = ['gemma2:2b']

    test_from = 'attrebute_val'
    train_from = 'attrebute_train'

    train_inputs = train_from + '.data'
    train_labels = train_from + '.solution'

    test_inputs = test_from + '.data'
    test_labels = test_from + '.solution'

    test_inputs = dataset[test_inputs]
    test_labels = dataset[test_labels]
    train_inputs = dataset[train_inputs]
    train_labels = dataset[train_labels]

    with enlighten.get_manager() as pbar_manager:

        def run_model(model, sub_pbar):
            predictions, labels = [], []
            for test_x, test_y in tqdm(zip(test_inputs, test_labels), total=len(test_inputs)):
                response=call_llm(
                    messages=[
                        dict(role='system', content=system_prompt),
                        dict(role='user', content=prepare_prompts(test_x))
                    ],
                    model=model,
                    temperature=0
                )

                predictions.append(parse_json(response))
                labels.append(test_y)
                sub_pbar.update()

            pd.DataFrame.from_records(predictions).to_csv(f'./preds/{model}_preds.csv')
            pd.DataFrame.from_records(test_labels).to_csv(f'./preds/{model}_labels.csv')

        threads = [
                Thread(target=run_model, args=(model, pbar_manager.counter(total=len(test_labels), desc=f'Running {model}', units='ticks')))
        for model in models]

        [thread.start() for thread in threads]
        [thread.join() for thread in threads]