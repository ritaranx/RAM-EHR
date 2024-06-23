import json 
from tqdm import tqdm 
import csv 
from fuzzywuzzy import fuzz
import re


field_name = "prescription"

with open(f"mimic_{field_name}_name.json", 'r') as f:
    data = json.load(f)
    returned_data = {}
    visit = 0
    success = 0
    for (idx, name) in tqdm(data.items()):
        if field_name == "prescription":
            idx, name = name, idx
        print(idx, name)
        returned_data[idx] = name 

with open("drugbank_drugs_info.csv", 'r') as f:
    f.readline()
    data = {}
    reader = csv.reader(f, delimiter = ',')
    for line in reader:
        drug_title = line[3]
        dbid = line[10]
        drug_name = line[11]
        desc = line[13]
        data[drug_title] = {"drugbank": dbid, "description": desc}
        data[drug_name] = {"drugbank": dbid, "description": desc}
    print(len(data))


def clean_string(s):
    # Remove "NEO*IV*"
    s = s.replace("NEO*IV*", "")
    s = s.replace("*NF*", "")
    s = re.sub(r'\d+(\.\d+)?%', '', s)
    
    # Remove content within brackets and the brackets themselves
    s = re.sub(r'\(.*?\)', '', s)
    return s.strip()

save_data = {}
for idx in tqdm(returned_data):
    name = returned_data[idx]
    score_max = -1
    name = clean_string(name)
    for drug_name in data:
        score = fuzz.ratio(name, clean_string(drug_name))
        if score > score_max:
            best_name = drug_name
            best_def = data[drug_name]["description"].split("\n")[0]
            score_max = score 
    
    if score_max >= 80:
        print(name, best_name, best_def, score_max)
        save_data[idx] = {"name": returned_data[idx], "drugbank_name": best_name, "def": best_def, "score": score_max}
    else:
        save_data[idx] = {"name": returned_data[idx], "drugbank_name": "", "def": "", "score": ""}

with open(f"mimic_{field_name}_name_drugbank.json", 'w') as f_out:      
    json.dump(save_data, f_out, indent = 2)