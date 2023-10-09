
import re
import cudf
import time
import json
import cupy

REPEAT = 1000

def get_dataset(filename, column_names=['column 1', 'column 2'],
                rows=None):
    if rows != None:
        nrows = rows
    else:
        nrows = int(re.search('\d+|$', filename).group())
    return cudf.read_csv(filename, sep='\t', header=None,
                         names=column_names, nrows=nrows)


def loading_time_benchmark(dataset, rows=None):
    graph = get_dataset(dataset, rows=rows)
    arr_cupy = cudf.to_cupy()
    size = len(arr_cupy)
    time_start = time.time()
    for _ in range(REPEAT):
        re_constructed = cudf.from_cupy(arr_cupy)
    time_end = time.time()
    return time_end - time_start, size

def main():
    datasets = {
        "ego-Facebook": "../data/data_88234.txt",
        "wiki-Vote": "../data/data_103689.txt",
        "luxembourg_osm": "../data/data_119666.txt",
        "fe_sphere": "../data/data_49152.txt",
        "fe_body": "../data/data_163734.txt",
        "cti": "../data/data_48232.txt",
        "fe_ocean": "../data/data_409593.txt",
        "wing": "../data/data_121544.txt",
        "loc-Brightkite": "../data/data_214078.txt",
        "delaunay_n16": "../data/data_196575.txt",
        "usroads": "../data/data_165435.txt",
        "CA-HepTh": "../data/data_51971.txt",
        "SF.cedge": "../data/data_223001.txt",
        "p2p-Gnutella31": "../data/data_147892.txt",
        "p2p-Gnutella09": "../data/data_26013.txt",
        "p2p-Gnutella04": "../data/data_39994.txt",
        "cal.cedge": "../data/data_21693.txt",
        "TG.cedge": "../data/data_23874.txt",
        "OL.cedge": "../data/data_7035.txt",
    }
    results = {}
    for dataset_name, dataset_path in datasets.items():
        loading_time, size = loading_time_benchmark(dataset_path)
        results[dataset_name] = {
            "loading_time": loading_time,
            "tuple/s: ": size * REPEAT / loading_time
        } 
    print(json.dumps(results, indent=4))

if __name__ == "__main__":
    main()
