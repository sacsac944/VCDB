"""
This module handles creation of dataframe for video copy sets from the annotations folder
"""

import os
import glob
import pandas as pd


class DataLoader:

    def __init__(self, dataset_path: str):
        self.vcdb_dataset_folder = dataset_path

    def create_video_copy_df(self) -> pd.DataFrame:
        videos_annotations_dir_path = os.path.join(self.vcdb_dataset_folder, "annotation")
        annotation_files = glob.glob(videos_annotations_dir_path + "/*")

        df = pd.read_csv(annotation_files[0])
        df.columns = ["Video_A", "Video_B", "Copy_Start_Video_A", "Copy_End_Video_A", "Copy_Start_Video_B", "Copy_End_Video_B"]

        for file in annotation_files[1:]:
            df_temp = pd.read_csv(file)
            df_temp.columns = ["Video_A", "Video_B", "Copy_Start_Video_A", "Copy_End_Video_A", "Copy_Start_Video_B",
                          "Copy_End_Video_B"]
            df = pd.concat([df, df_temp])

        return df


