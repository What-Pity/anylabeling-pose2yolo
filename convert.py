from pathlib import Path
import json
from argparse import ArgumentParser
import numpy as np
import logging
import pandas as pd
from tqdm import tqdm


def get_args():
    parser = ArgumentParser(
        "Convert JSON annotation created by labelme to YOLO format")
    parser.add_argument("--input", type=str,
                        help="Path to input JSON file", default='./json')
    parser.add_argument("--output", type=str,
                        help="Path to output file", default='./yolo')
    parser.add_argument("--cfg_label", type=str,
                        default="./labels.txt", help="Path to label file of labelme")
    args = parser.parse_args()
    return args


def pre_process(args):
    if not Path(args.input).exists():
        logging.error("Input file does not exist")
        exit(1)
    Path(args.output).mkdir(parents=True, exist_ok=True)

    if not Path(args.cfg_label).exists():
        logging.error("Label file does not exist")
        exit(1)

    try:
        next(Path(args.input).glob("*.json"))
    except StopIteration:
        logging.error("No JSON file found in input directory")
        exit(1)


def validate_dataframe(data, indexs, columns):
    # 检查data.loc[indexs,columns]是否存在，不做任何事
    # 若存在，检查其元素是否为NAN，若不是，发出警告
    idx = data.index
    col = data.columns
    if not isinstance(indexs, list):
        indexs = [indexs]
    if not isinstance(columns, list):
        columns = [columns]

    for index in indexs:
        if index not in idx:
            logging.debug(f"Index {index} not found in dataframe")
            return
    for column in columns:
        if column not in col:
            logging.debug(f"Column {column} not found in dataframe")
            return
        else:
            if not data.loc[indexs, columns].isnull().all().all():
                logging.warning(
                    f"Conflict data found in <group_id {index} - attribute {columns}>, previous data <{data.loc[indexs, columns]}> will be overwritten")


def import_data(data_in, label_map):
    columns = ['x1', 'y1', 'x2', 'y2', 'W', 'H']
    for i in range(len(label_map)):
        columns.append("px"+str(i))
        columns.append("py"+str(i))
        columns.append("occluded"+str(i))
    data = pd.DataFrame(columns=columns)
    log_debug_df(data)
    for shape in data_in['shapes']:
        anno_type = shape['label']
        group_id = shape['group_id']
        if group_id is None:
            logging.warning(
                f"No group_id found in shape, please check annotiation file")
            group_id = -1
        if shape['shape_type'] == 'rectangle':
            points = shape['points']
            x1, y1, x2, y2 = points[0][0], points[0][1], points[1][0], points[1][1]
            validate_dataframe(data, group_id, ['x1', 'y1', 'x2', 'y2'])
            data.loc[group_id, ['x1', 'y1', 'x2', 'y2']] = [x1, y1, x2, y2]
        elif shape['shape_type'] == 'point':
            points = shape['points']
            x, y = points[0][0], points[0][1]
            occluded = any(shape["flags"].values())  # flags有任何一个为真，则认为被遮挡
            try:
                points_label = label_map.index(anno_type)
            except ValueError:
                logging.error(
                    f"Unexpected label detected: {anno_type}, please check annotaion file")
                exit(1)
            validate_dataframe(data, group_id,
                               ['px'+str(points_label), 'py'+str(points_label), 'occluded'+str(points_label)])

            data.loc[group_id,
                     ['px'+str(points_label), 'py'+str(points_label), 'occluded'+str(points_label)]] = [x, y, occluded]
        else:
            logging.warning(
                f"Unexpected shape type detected: {shape['shape_type']}")
            continue
    w, h = data_in['imageWidth'], data_in['imageHeight']
    data['W'], data['H'] = w, h
    return data


def log_debug_df(dataframe):
    logging.debug(f"\n{dataframe}")


def process_data(data_in):
    data_out = pd.DataFrame(
        columns=["classIndex", "x", "y", "width", "height"])
    data_out['x'] = (data_in['x1']+data_in["x2"])/2/data_in['W']
    data_out['y'] = (data_in['y1']+data_in["y2"])/2/data_in['H']
    data_out['width'] = abs((data_in['x2']-data_in["x1"])/data_in['W'])
    data_out['height'] = abs((data_in['y2']-data_in["y1"])/data_in['H'])
    data_out['classIndex'] = 0
    log_debug_df(data_out)
    keypoint = data_in.iloc[:, 6:].fillna(0)
    log_debug_df(keypoint)
    occulded_name = [occ for occ in keypoint.columns if 'occluded' in occ]
    keypoint[occulded_name] = keypoint[occulded_name].map(
        lambda x: 1 if x is True else (2 if x is False else x))
    keypoint = keypoint.rename(columns=lambda x: x.replace(
        "occluded", "visibility") if "occluded" in x else x)
    log_debug_df(keypoint)
    coordinatex_names = [coorx for coorx in keypoint.columns if "px" in coorx]
    coordinatey_names = [coory for coory in keypoint.columns if "py" in coory]
    for coordinatex_name in coordinatex_names:
        keypoint[coordinatex_name] = keypoint[coordinatex_name]/data_in['W']
    for coordinatey_name in coordinatey_names:
        keypoint[coordinatey_name] = keypoint[coordinatey_name]/data_in['H']
    log_debug_df(keypoint)
    data_out = pd.concat([data_out, keypoint], axis=1)
    log_debug_df(data_out)
    return data_out


logging.basicConfig(level=logging.INFO)
args = get_args()
pre_process(args)
with open(args.cfg_label, 'r') as f:
    labels = f.read().splitlines()[1:]

for json_path in tqdm(Path(args.input).glob("*.json")):
    logging.info(f"Processing {str(json_path.absolute())}")
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    raw_data = import_data(json_data, labels)
    log_debug_df(raw_data)
    data = process_data(raw_data)
    data.to_csv(Path(args.output, json_path.stem+'.txt'),
                sep=' ', index=False, header=False)
