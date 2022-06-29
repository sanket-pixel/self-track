import os


def get_sequences(tracking_path="", dataset=""):
    sequence_ids = os.listdir(tracking_path)
    sequence_list = []
    for seq_id in sequence_ids:
        if dataset == "MOT17":
            if "FRCNN" in seq_id:
                sequence_list.append(seq_id)
        elif dataset == "MOT20":
            sequence_list.append(seq_id)
    return sorted(sequence_list)