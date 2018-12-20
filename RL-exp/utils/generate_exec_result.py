import pickle
import argparse
import os
import re
import sys

"""
Generate execution result file from labels of each models

Usage:
    python3 generate_exec_result.py in_dir regex out_path

    in_dir should contains execution results(.pkl) of each model
    one pickle file for each model
    e.g.
        result/
            res_darknet.pkl
            res_openface.pkl
            ...

    regex is the regular expression to extract the 'modelname' substring
    e.g. above in_dir uses 'res_(.+).pkl' as the regex

Return:
exec_result_pkl: the execution result file(.pkl) of data
    data structure:
    {
        "data_id": [
            ("label", conf),    # label & confidence
            ...
        ]
    }
    
"""

def extract_modelname(file_name, regex):
    tmp = re.search(regex, file_name)
    if tmp:
        return tmp.group(1)
    return None

def read_darknet(label_tmp):
    res = []
    for (obj_id, conf, _) in label_tmp:
        res.append((obj_id, conf))
    return res

def read_facerecog(label_tmp):
    res = []
    for _ in label_tmp:
        res.append(("face", 1.0))
    return res

def read_places365(label_tmp):
    res = []
    for (conf, place_id) in label_tmp[:1]:
        res.append((place_id, conf))
    return res

if __name__ == "__main__":
    # argument parsing
    parser = argparse.ArgumentParser(description="Generate execution result.")
    parser.add_argument("in_dir", type=str, help="Input directory of execution results of models.")
    parser.add_argument("regex", type=str, help="The regular expression to extract the 'modelname' substring.")
    parser.add_argument("out_path", type=str, help="Output file path.")
    args = parser.parse_args()
    if not os.path.isdir(args.in_dir):
        parser.error("{} is not a valid directory.".format(args.in_dir))

    exec_result = {}
    for pkl_file_name in os.listdir(args.in_dir):
        modelname = extract_modelname(pkl_file_name, args.regex)
        if not modelname:
            print("Skip file {}".format(pkl_file_name))
            continue
        pkl_data = pickle.load(open(os.path.join(args.in_dir, pkl_file_name), "rb"))
        for data_id, label_tmp in pkl_data.items():
            if data_id not in exec_result:
                exec_result[data_id] = []
            # call local ad-hoc function to read pickle data
            exec_result[data_id] += locals()["read_"+modelname](label_tmp)
            #print(data_id, modelname)
    pickle.dump(exec_result, open(args.out_path, "wb"))
    print("{} saved.".format(args.out_path))
