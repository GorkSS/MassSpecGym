import json
import pandas as pd
from pathlib import Path

def load_chemont_dict(path: Path) -> dict:
    df = pd.read_csv(path, sep='\t')
    df.drop(['chemont_id', 'parent_name'], axis=1, inplace=True)
    j =  json.loads(df.to_json())
    numeric_ids = j['numeric_id']
    chemont_d = dict()
    for i in range(1, len(numeric_ids)):
        chemont_key = str(i)
        new_key = j['numeric_id'][chemont_key]
        chemont_d[new_key] = {
            'name': j['name'][chemont_key],
        }
        parent_id = j['parent_numeric_id'][chemont_key]
        chemont_d[new_key]['parent_numeric_id'] = int(parent_id) if parent_id != None else None

    return chemont_d

print(json.dumps(load_chemont_dict(Path("data/chemont_dictionary.tsv")), indent=4))