from Bio import Entrez
import pickle 
import math 
from tqdm import tqdm, trange
import json
import re 


def load_keywords(field_name):
    file_name = f"mimic_{field_name}_name.json"
    with open(file_name, 'r') as f:
        data = json.load(f)
        returned_data = {}
        name_to_id = {}
        queries = []
        visit = 0
        success = 0
        for (idx, name) in tqdm(data.items()):
            if field_name == "prescription":
                idx, name = name, idx
            print(idx, name)
            name_to_id[name] = idx 
            queries.append(name)
    return name_to_id, queries


def split_ids(ids_strings):
    pubmed = []
    pmc = []
    for id_string in ids_strings:
        if id_string.startswith('PMC'):
            pmc.append(id_string[3:])  # Drop PMC prefix since it is no longer needed to distinguish between PubMed/PMC
        else:
            pubmed.append(id_string)
    return pubmed, pmc


def get_abstract_text(subrec):
    if "Abstract" in subrec:
        abstract_text_lines = subrec["Abstract"]["AbstractText"]
        return "\n".join(subrec["Abstract"]["AbstractText"]) + "\n"
    else:
        return ""


def get_abstract_dict(abstract_record):
    pmid = str(abstract_record["MedlineCitation"]["PMID"])
    text = get_abstract_text(abstract_record["MedlineCitation"]["Article"])
    # print(abstract_record)
    # year = int(abstract_record["MedlineCitation"]["DateCompleted"]["Year"])
    return pmid, {"text": text}


def retrieve_all_abstracts(id_list, database):
    max_query_size = 200  # PubMed only accepts 200 IDs at a time when retrieving abstract text
    print('Retrieval will require {} queries'.format(math.ceil(len(id_list)/float(max_query_size))))
    texts = {}

    total_texts = 0
    for i in trange(0, len(id_list), max_query_size):

        start = i
        end = min(len(id_list), start+max_query_size)
        cur_ids = id_list[start:end]

        handle = Entrez.efetch(database, id=cur_ids, retmod="xml")
        record = Entrez.read(handle, validate=False)


        d = map(get_abstract_dict, record["PubmedArticle"])
        cur_texts = dict((x, y) for x, y in d if y["text"]!="")
        total_texts += len(cur_texts)

        texts.update(cur_texts)
    
    return texts

# Set your email and API key
Entrez.email = ""
Entrez.api_key = ""

# Define your query
field_name = "prescription" # "prescription"
name_to_id, qtext = load_keywords(field_name)
qtext= qtext
# query = "Other bacterial pneumonia"

id_list = []
for query in tqdm(qtext):
    # Use ESearch to find article IDs related to the query
    if query.strip():
        search_results = Entrez.esearch(db="pubmed", term=query.strip(), retmax=30) # retmax sets the number of results to retrieve
        success = False
        while not success:
            try:
                record = Entrez.read(search_results, validate=False)
                success = True
            except RuntimeError:
                print("Runtime Error!")
                pass
        # Fetch article details based on their IDs
        num_papers = int(record["Count"])
        id_list += [str(x) for x in record["IdList"]]
pubmed_ids, pmc_ids = split_ids(id_list)
texts = retrieve_all_abstracts(pubmed_ids, 'pubmed')
articles_ids = set()
print(f"{num_papers} papers, {len(id_list)} ids found.")

with open(f"{field_name}_pubmed.txt", 'w') as f:
    for text in texts:
        text= texts[text]["text"]
        text = re.sub(r"[^A-Za-z0-9=(),!?\'\`]", " ", text)
        f.write(text.strip() + "\n")

