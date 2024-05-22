import re
import cudf
import time
import json


def display_time(time_start, time_end, message):
    time_took = time_end - time_start
    print(f"Debug: {message}: {time_took:.6f}s")


def get_join(relation_1, relation_2, column_names=['column 1', 'column 2']):
    return relation_1.merge(relation_2, on=column_names[0],
                            how="inner",
                            suffixes=('_relation_1', '_relation_2'))


def get_projection(result, column_names=['column 1', 'column 2']):
    temp = result.drop([column_names[0]], axis=1).drop_duplicates()
    temp.columns = column_names
    return temp


def get_union(relation_1, relation_2):
    return cudf.concat([relation_1, relation_2],
                       ignore_index=True).drop_duplicates()


def get_dataset(filename, column_names=['column 1', 'column 2'],
                rows=None):
    if rows != None:
        nrows = rows
    else:
        nrows = int(re.search('\d+|$', filename).group())
    return cudf.read_csv(filename, sep='\t', header=None,
                         names=column_names, nrows=nrows)


def get_transitive_closure(dataset):
    COLUMN_NAMES = ['column 1', 'column 2']
    rows = int(re.search('\d+|$', dataset).group())
    start_time_outer = time.perf_counter()
    relation_1 = get_dataset(dataset, COLUMN_NAMES, rows)
    relation_2 = relation_1.copy()
    relation_2.columns = COLUMN_NAMES[::-1]
    temp_result = relation_1
    i = 0
    while True:
        temp_projection = get_projection(get_join(relation_2, relation_1,
                                                  COLUMN_NAMES), COLUMN_NAMES)
        x = len(temp_projection)
        previous_result_size = len(temp_result)
        temp_result = get_union(temp_result, temp_projection)
        current_result_size = len(temp_result)
        if previous_result_size == current_result_size:
            i += 1
            break
        del relation_2
        relation_2 = temp_projection
        relation_2.columns = COLUMN_NAMES[::-1]
        i += 1
        del temp_projection
        # print(f"i: {i}, projection size: {x}, rows: {current_result_size}")
    end_time_outer = time.perf_counter()
    time_took = end_time_outer - start_time_outer
    time_took = f"{time_took:.6f}"
    # print(temp_result)
    return rows, len(temp_result), i, time_took


def generate_benchmark(iterative=True, datasets=None):
    result = []
    if iterative:
        print("| Number of rows | TC size | Iterations | Time (s) |")
        print("| --- | --- | --- | --- |")
        increment = 1000
        n = 990
        count = 0
        while n < 11000:
            try:
                dataset = f"../data/data_{n}.txt"
                n = int(re.search('\d+|$', dataset).group())
                record = get_transitive_closure(dataset)
                result.append(record)
                print(
                    f"| {record[0]} | {record[1]} | {record[2]} | {record[3]:.6f} |")
                n += increment
            except Exception as ex:
                print(str(ex))
                break
            count += 1
    if datasets:
        print("| Dataset | Number of rows | TC size | Iterations | Time (s) |")
        print("| --- | --- | --- | --- | --- |")
        for key, dataset in datasets.items():
            try:
                record = get_transitive_closure(dataset)
                record = list(record)
                record.insert(0, key)
                result.append(record)
                message = " | ".join([str(s) for s in record])
                message = "| " + message + " |"
                print(message)
            except Exception as ex:
                print(str(ex))
                break
    print("\n")
    with open('transitive_closure.json', 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    generate_benchmark(iterative=False, datasets={
        "ego-Facebook": "../data/data_88234.txt",
        "wiki-Vote": "../data/data_103689.txt",
        "luxembourg_osm": "../data/data_119666.txt",
        "fe_sphere": "../data/data_49152.txt",
        # "fe_body": "../data/data_163734.txt",
        "cti": "../data/data_48232.txt",
        # "fe_ocean": "../data/data_409593.txt",
        "wing": "../data/data_121544.txt",
        # "loc-Brightkite": "../data/data_214078.txt",
        "delaunay_n16": "../data/data_196575.txt",
        # "usroads": "../data/data_165435.txt",
        "CA-HepTh": "../data/data_51971.txt",
        "SF.cedge": "../data/data_223001.txt",
        # "p2p-Gnutella31": "../data/data_147892.txt",
        "p2p-Gnutella09": "../data/data_26013.txt",
        "p2p-Gnutella04": "../data/data_39994.txt",
        "cal.cedge": "../data/data_21693.txt",
        "TG.cedge": "../data/data_23874.txt",
        "OL.cedge": "../data/data_7035.txt",
    })


