import csv
import os
import argparse

headers = ["qs", "depth", "tcount", "min_weight", "max_weight", "nsamples", "seed", "terms", "time", "tcounts", "no_simp", "naive"]

def parse_data(values):
    data_dict = {}
    for key, value in zip(headers, values):
        if ',' in value:
            v = value.split(",")
            v = [int(x) for x in v]
        elif '+' in value:
            v = float(value)
        else:
            v = int(value)
        data_dict[key.strip('"')] = v
    return data_dict

def main():
    parser = argparse.ArgumentParser(description='Process some files.')
    parser.add_argument('-f', '--is_file', action='store_true', help='Flag indicating if the input is a file.', required=False, default=False)
    parser.add_argument('-d', '--directory', help='The directory containing the files.', required=False, default='data_pg_ap')
    parser.add_argument('-nq', '--num_qubits', type=int, help='The number of qubits.', required=False, default=50)
    parser.add_argument('-nc', '--num_ccz', type=int, help='The number of CCZ gates.', required=False, default=30)
    parser.add_argument('-s', '--seed', type=int, help='The seed value.', required=False, default=1077)
    parser.add_argument('-c', '--compare_directories', action='store_true', help='Flag indicating if directories should be compared.', required=False, default=False)
    parser.add_argument('-d2', '--second_directory', help='The second directory to compare.', required=False, default='data_pg_og')
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
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            with open(filepath) as f:
                r = csv.reader(f)
                data = list(r)
            data = parse_data(data[0])
            d[filename] = [int(data["terms"])]
            
        for filename in os.listdir(second_directory):
            filepath = os.path.join(second_directory, filename)
            with open(filepath) as f:
                r = csv.reader(f)
                data = list(r)
            data = parse_data(data[0])
            if filename not in d:
                d[filename] = [None]
            d[filename].append(int(data["terms"]))
        for k, v in d.items():
            print(k, v, v[0] < v[1])

    else:
        d = []
        for filename in os.listdir(directory):
            filepath = os.path.join(directory, filename)
            with open(filepath) as f:
                r = csv.reader(f)
                data = list(r)
            p_data = [filename]
            p_data.extend(parse_data(data[0]))
            d.append(p_data)
        for p in d:
            print(p)
        

if __name__ == "__main__":
    main()