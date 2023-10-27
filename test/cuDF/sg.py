import re
import cudf
import time
import json

REPEAT = 3

def display_time(time_start, time_end, message):
    time_took = time_end - time_start
    print(f"Debug: {message}: {time_took:.6f}s")


def get_join(relation_1, relation_2, column_names=['column 1', 'column 2']):
    return relation_1.merge(relation_2, on=column_names[0],
                            how="inner",
                            suffixes=('_relation_1', '_relation_2'))


def get_projection(result, column_names=['column 1', 'column 2'], remove_same_val=False):
    temp = result.drop([column_names[0]], axis=1).drop_duplicates()
    temp.columns = column_names
    if remove_same_val:
        temp = temp.loc[(temp[column_names[0]] != temp[column_names[1]])]
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


def get_sg(dataset):
    COLUMN_NAMES = ['column 1', 'column 2']
    rows = int(re.search('\d+|$', dataset).group())
    start_time_outer = time.perf_counter()
    relation_1 = get_dataset(dataset, COLUMN_NAMES, rows)
    relation_2 = relation_1.copy()
    # sg(x, y): - edge(p, x), edge(p, y), x != y.
    temp_result = get_projection(get_join(relation_1, relation_2,
                                                  COLUMN_NAMES), COLUMN_NAMES, remove_same_val=True)
    i = 0
    relation_2 = temp_result
    while True:
        # tmp(b, x): - edge(a, x), sg(a, b).
        temp_projection = get_projection(get_join(relation_1, relation_2,
                                                  COLUMN_NAMES), COLUMN_NAMES)
        temp_projection.columns = COLUMN_NAMES[::-1]
        # sg(x, y): - tmp(b, x), edge(b, y).
        temp_projection_2 = get_projection(get_join(temp_projection, relation_1,
                                                  COLUMN_NAMES), COLUMN_NAMES)
        relation_2 = temp_projection_2
        previous_result_size = len(temp_result)
        temp_result = get_union(temp_result, relation_2)
        current_result_size = len(temp_result)
        if previous_result_size == current_result_size:
            i += 1
            break
        i += 1
        del temp_projection
        del temp_projection_2
    end_time_outer = time.perf_counter()
    time_took = end_time_outer - start_time_outer
    time_took = f"{time_took:.6f}"
    return rows, len(temp_result), int(i), time_took


def generate_benchmark(datasets=None):
    result = []
    print("| Dataset | Number of rows | SG size | Iterations | Time (s) |")
    print("| --- | --- | --- | --- | --- |")
    for key, dataset in datasets.items():
        time_took = []
        record = None
        try:
            # Omit the warm up round timing
            warm_up = get_sg(dataset)
            for i in range(REPEAT):
                try:
                    record = get_sg(dataset)
                    time_took.append(float(record[3]))
                except Exception as ex:
                    print(str(ex))
            record = list(record)
            record[3] = f"{(sum(time_took) / REPEAT):.6f}"
            record.insert(0, key)
            result.append(record)
            message = " | ".join([str(s) for s in record])
            message = "| " + message + " |"
            print(message)
        except Exception as ex:
            print(f"Error in {key}. Message: {str(ex)}")
    print("\n")
    with open('sg.json', 'w') as f:
        json.dump(result, f)


if __name__ == "__main__":
    generate_benchmark(datasets={
        "hipc": "../../data/data_5.txt",
        "fe_body": "../../data/data_163734.txt",
        "loc-Brightkite": "../../data/data_214078.txt",
        "fe_sphere": "../../data/data_49152.txt",
        "CA-HepTh": "../../data/data_51971.txt",
        # "SF.cedge": "../../data/data_223001.txt",
        "ego-Facebook": "../../data/data_88234.txt",
        "wiki-Vote": "../../data/data_103689.txt",
        "luxembourg_osm": "../../data/data_119666.txt",
        "cti": "../../data/data_48232.txt",
        "fe_ocean": "../../data/data_409593.txt",
        "wing": "../../data/data_121544.txt",
        "delaunay_n16": "../../data/data_196575.txt",
        "usroads": "../../data/data_165435.txt",
        "p2p-Gnutella31": "../../data/data_147892.txt",
        "p2p-Gnutella09": "../../data/data_26013.txt",
        "p2p-Gnutella04": "../../data/data_39994.txt",
        "cal.cedge": "../../data/data_21693.txt",
        "TG.cedge": "../../data/data_23874.txt",
        "OL.cedge": "../../data/data_7035.txt",
    })
