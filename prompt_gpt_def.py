import openai
openai.api_type = "azure"
openai.api_base = "YOUR_API_BASE"
openai.api_version = "YOUR_API_VERSION"
openai.api_key = "YOUR_API_KEY"

import asyncio
from typing import List, Dict, Any
import argparse
import os 
from tqdm import tqdm
import re 
import time
import json 
import random 

def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9(),.!?\"\']", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()


async def dispatch_openai_requests(
    messages_list: List[List[Dict[str, Any]]],
    temperature: float,
    max_tokens: int,
) -> List[str]:
    """Dispatches requests to OpenAI API asynchronously.
    
    Args:
        messages_list: List of messages to be sent to OpenAI ChatCompletion API.
        model: OpenAI model to use.
        temperature: Temperature to use for the model.
        max_tokens: Maximum number of tokens to generate.
        top_p: Top p to use for the model.
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = [ 
            openai.ChatCompletion.acreate(
            engine="your_engine",
            messages=x,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
            frequency_penalty=0,
            presence_penalty=0,
            stop=None
        )
        for x in messages_list
    ]

    return await asyncio.gather(*async_responses)

def call_api_async(msg_lst,  temperature, max_tokens):
    print("===================================")
    print(f"call APIs, {len(msg_lst)} in total, t= {temperature}.")
    l = len(msg_lst)

    response = asyncio.run(
        dispatch_openai_requests(
            messages_list = msg_lst,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )
    ans = [x['choices'][0]['message']['content'] for x in response]
    print(f"API returns {len(ans)} in total.")
    print("===================================")
    return ans 

def build_prompt(example, domain, remove):
    prompt = f"""Suppose you are working on a health-related phenotyping task and need to get relevant information for the given {domain}. Here are some relevant information:\nDisease Name: {example["name"]}"""
    name = f"{example['name']}"
    defs = []
    if remove == 'all':
        pass
    else:
        for k in example:
            if "def" in k and remove not in k and example[k]:
                defs.append(example[k])
    for i, definition in enumerate(defs):
        prompt += f"\nRelevant Information {i+1}: {definition}"
    prompt += f"\nBased on the above information, Could you generate 1 sentences to summarize the knowledge for the {domain} '{name}' that are useful for health phenotyping task?"
    return prompt 


parser = argparse.ArgumentParser("")
parser.add_argument("--temperature", default=0.0, type=float, help="which seed to use")
parser.add_argument("--dataset", default='mimic', type=str, help="which model to use")
parser.add_argument("--domain", default='disease_id', type=str, help="")
parser.add_argument("--remove", default='', type=str, help="which field to remove")

parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--output_dir", default='gpt_summary', type=str, help="the folder for saving the generated text")
parser.add_argument("--api_key", default='', type=str, help="the api key for your openai account. See https://platform.openai.com/account/api-keys for how to get the key.")

args = parser.parse_args()


with open(f"{args.dataset}_{args.domain}_name_merge.json", 'r') as f_out:      
    data = json.load(f_out)

examples = []
idxs = []
names = []
prompt_lst = []
length = 0
return_dict = {}
examples = []
total_len = len(data)
for key in data:
    example = data[key]
    idxs.append(key)
    names.append(example["name"])
    examples.append(example)
    prompt_input = build_prompt(example, args.domain, args.remove)
    prompt_lst.append(
                    [{"role": "user", "content": prompt_input}]
            )
    length += 1
    if length % 5 == 0:
        success = False
        while not success:
            try:
                ans = call_api_async(prompt_lst, args.temperature, max_tokens = 200)
                success = True
            except openai.error.RateLimitError:
                print(f"RateLimitError.")
                time.sleep(10)
                continue
            except  openai.error.APIError:
                print(f"APIError.")
                time.sleep(5)
                continue
            except openai.error.InvalidRequestError:
                print("InvalidRequestError!")
                ans = [""] * len(prompt_lst)
                success = True
                continue
            except openai.error.ServiceUnavailableError:
                print("ServiceUnavailableError")
                time.sleep(5)
                continue
            except openai.error.Timeout:
                print("TimeoutError")
                time.sleep(5)
                continue
            except openai.error.APIConnectionError:
                print("APIConnectionError")
                time.sleep(5)
                continue

        for id, n, a, e in zip(idxs, names, ans, examples):
            e["gpt_sum_all"] = a
            return_dict[id] = e
            # print(return_dict[id])
        print(f'{len(return_dict)}/{total_len}')
        length = 0
        prompt_lst = []
        idxs, names, examples = [], [], []

if  prompt_lst:
    success = False
    while not success:
        try:
            ans = call_api_async(prompt_lst, args.temperature, max_tokens = 200)
            success = True
        except openai.error.RateLimitError:
            print(f"RateLimitError.")
            time.sleep(10)
        except  openai.error.APIError:
            print(f"APIError.")
            time.sleep(5)
            continue
        except openai.error.InvalidRequestError:
            print("InvalidRequestError!")
            time.sleep(5)
            ans = [""] * len(prompt_lst)
            success = True
            continue
        except openai.error.ServiceUnavailableError:
            print("ServiceUnavailableError")
            time.sleep(5)
            continue
        except openai.error.Timeout:
            print("TimeoutError")
            time.sleep(5)
            continue
        except openai.error.APIConnectionError:
            print("APIConnectionError")
            time.sleep(5)
            continue

    for id, n, a, e in zip(idxs, names, ans, examples):
        e["gpt_sum_all"] = a
        return_dict[id] = e
        
if args.remove:
    file_name = f"{args.output_dir}/{args.dataset}_{args.domain}_name_gpt_summary_remove_{args.remove}.json"
else:
    file_name = f"{args.output_dir}/{args.dataset}_{args.domain}_name_gpt_summary.json"
with open(file_name, 'w') as f_out:      
    json.dump(return_dict, f_out, indent = 2)
