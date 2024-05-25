import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from util import determine_pair_direction


def data_process():
    origin_df = pd.read_csv("events.csv")
    match_id_list = origin_df["id_odsp"].unique()

    event_count_dict = pd.DataFrame(columns=list(range(1, 12)))
    for match_id in tqdm(match_id_list):
        match_df = origin_df[(origin_df["id_odsp"] == match_id) & (origin_df["side"] == 1)]
        temp_dict = Counter(list(match_df["event_type"]))
        event_count_dict = event_count_dict._append(temp_dict, ignore_index=True)

        match_df = origin_df[(origin_df["id_odsp"] == match_id) & (origin_df["side"] == 2)]
        temp_dict = Counter(list(match_df["event_type"]))
        event_count_dict = event_count_dict._append(temp_dict, ignore_index=True)

    event_count_dict.to_csv("foot_ball_event_table.csv", columns=None)



if __name__ == '__main__':
    """
    Event type:
    0	Announcement 
    1	Attempt
    2	Corner
    3	Foul
    4	Yellow card
    5	Second yellow card
    6	Red card
    7	Substitution
    8	Free kick won
    9	Offside
    10	Hand ball
    11	Penalty conceded
    """
    # data_process()
    event_table = pd.read_csv("foot_ball_event_table.csv").fillna(0)
    event_table = np.array(event_table).astype(int)

    causal_pair_list = [[3, 4], [3, 5], [3, 6], [4, 5], [5, 6], [4, 7]]

    for causal_pait in causal_pair_list:
        x = event_table[:, causal_pait[0]]
        y = event_table[:, causal_pait[1]]

        print(determine_pair_direction(x, y, max_order=3, alpha=0.1, threshold=0))

