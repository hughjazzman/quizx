import csv
import os
import argparse
import pandas as pd
import numpy as np

headers = ["qs", "depth", "tcount", "min_weight", "max_weight", "nsamples", "seed", "terms", "time", "tcounts", "no_simp", "naive", "alpha"]

def parse_data(values: list[str]):
    data_dict = {}
    for key, value in zip(headers, values):
        if ',' in value:
            v = value.split(",")
            if '.' in value:
                v = [float(x) for x in v]
            else:
                v = [int(x) for x in v]
        elif '+' in value or '.' in value:
            v = float(value)
        else:
            v = int(value)
        data_dict[key.strip('"')] = v
    # max_tcount = max(max(data_dict["tcounts"]), 1)
    # data_dict["alpha"] = round(math.log2(data_dict["terms"]) / max_tcount, 5)
    return data_dict

def parse_directory(directory: str):
    dfs = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        curr = pd.read_csv(
            filepath, 
            header=None, 
            names=["qs", "depth", "tcount", "min_weight", 
                    "max_weight", "nsamples", "seed", "terms", 
                    "time", "tcounts", "no_simp", "naive", "alpha"]
        )
        dfs.append(curr)
    
    full_data = pd.concat(dfs, ignore_index=True)
    full_data["tcounts"] = full_data["tcounts"].str.split(",")
    if full_data["alpha"].dtype != np.float64:
        full_data["alpha"] = full_data["alpha"].str.split(",")
        full_data = full_data.explode(["tcounts", "alpha"])
    else:
        full_data = full_data.explode("tcounts")

    
    full_data = full_data.astype({"tcounts": int, "alpha": float})


    full_data = full_data[full_data['alpha'] >= 0]
    full_data = full_data[full_data['tcounts'] > 1]

    # full_data["tcounts"] = full_data["tcounts"].apply(lambda x: max(x, 1))
    full_data["log_terms"] = np.log10(full_data["terms"])
    full_data["log_time"] = np.log10(full_data["time"])

    data = full_data.loc[:, ["tcount","seed","terms", "log_terms", "time", "log_time", "alpha", "tcounts"]]
    data["time"] /= 1000    # convert milliseconds -> seconds

    return data

def main():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('-f', '--is_file', action='store_true', help='Flag indicating if the input is a file.', required=False, default=False)
    parser.add_argument('-d', '--directory', help='The directory containing the files.', required=False, default='data')
    parser.add_argument('-nq', '--num_qubits', type=int, help='The number of qubits.', required=False, default=50)
    parser.add_argument('-nc', '--num_ccz', type=int, help='The number of CCZ gates.', required=False, default=30)
    parser.add_argument('-s', '--seed', type=int, help='The seed value.', required=False, default=1077)
    parser.add_argument('-c', '--compare_directories', action='store_true', help='Flag indicating if directories should be compared.', required=False, default=False)
    parser.add_argument('-d2', '--second_directory', help='The second directory to compare.', required=False, default='data_og')
    args = parser.parse_args()

    is_file = args.is_file
    directory = args.directory
    num_qubits = args.num_qubits
    num_ccz = args.num_ccz
    seed = args.seed
    compare_directories = args.compare_directories
    second_directory = args.second_directory
    
    if is_file:
        filename = f"pauli_gadget_{num_qubits}_{num_ccz}_2_4_1_{seed}"
        filepath = os.path.join(directory, filename)
        with open(filepath) as f:
            r = csv.reader(f)
            data = list(r)
        data = parse_data(data[0])
        print(data)
    elif compare_directories:
        d = {}
        df1 = parse_directory(directory)
        df2 = parse_directory(second_directory)

        d = df1.join(df2, how='inner',lsuffix='1', rsuffix='2')

        d["diff_terms"] = d["terms1"] - d["terms2"]
        d["diff_alpha"] = d["alpha1"] - d["alpha2"]
        d["diff_alpha"] = d["diff_alpha"].abs()
        d = d[d["diff_alpha"] > 1e-5]
        d.sort_values(by=['diff_alpha'], ascending=False, inplace=True)  
        d = d[["tcount1", "seed1", "log_terms1", "alpha1", "alpha2", "diff_alpha"]]

        print(d.head())

    else:
        df = parse_directory(directory)
        d = df.loc[:, ["seed","terms", "log_terms", "alpha", "tcounts"]]
        print(d.to_string())
    


if __name__ == "__main__":
    main()