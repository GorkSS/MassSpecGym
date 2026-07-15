import json
import pandas as pd
from pathlib import Path

class ChemontDict:
    def __init__(self, dict_path: Path) -> None:
        self.d = load_chemont_dict(dict_path)

    def get_name_of(self, id : int):
        try:
            return self.d[id]["name"]
        except KeyError:
            raise Exception(f"There's no ChemONT classification with id {id}")
        
    def get_parent_id_of(self, id : int):
        try:
            return self.d[id]["parent_id"]
        except KeyError:
            raise Exception(f"There's no ChemONT classification with id {id}")
    
    def get_parent_name_of(self, id : int):
        try:
            return self.d[self.d[id]["parent_id"]]["name"]
        except KeyError:
            raise Exception(f"There's no ChemONT classification with id {id}")

def load_chemont_dict(path: Path) -> dict:
    df = pd.read_csv(path, sep='\t')
    df.drop(['chemont_id', 'parent_name'], axis=1, inplace=True)
    j = json.loads(df.to_json())
    numeric_ids = j['numeric_id']
    chemont_d = dict()
    for i in range(len(numeric_ids)):
        chemont_key = str(i)
        new_key = j['numeric_id'][chemont_key]
        chemont_d[new_key] = {
            'name': j['name'][chemont_key],
        }
        parent_id = j['parent_numeric_id'][chemont_key]
        chemont_d[new_key]['parent_id'] = int(parent_id) if parent_id != None else None

    return chemont_d