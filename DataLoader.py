"""
This module handles creation of dataframe for video copy sets from the annotations folder
"""

import os
import glob
import random
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd

from pre_process_videos import preprocess_frame_inception_v3, preprocess_frame_resnet_50, preprocess_frame_vgg_19
from static import ImageNetwork

import tensorflow as tf


class DataLoader:

    def __init__(self, dataset_path: str):
        self.vcdb_dataset_folder = dataset_path
        self.video_category_copy_pairs = self.create_video_category_copy_pairs()

    def create_annotation_df(self) -> pd.DataFrame:
        annotation_files = self.get_video_categories()

        df = pd.read_csv(annotation_files[0])
        df.columns = ["Video_A", "Video_B", "Copy_Start_Video_A", "Copy_End_Video_A", "Copy_Start_Video_B", "Copy_End_Video_B"]

        for file in annotation_files[1:]:
            df_temp = pd.read_csv(file)
            df_temp.columns = ["Video_A", "Video_B", "Copy_Start_Video_A", "Copy_End_Video_A", "Copy_Start_Video_B",
                          "Copy_End_Video_B"]
            df = pd.concat([df, df_temp])

        return df

    def create_video_category_copy_pairs(self):
        annotation_files = self.get_video_categories()
        video_category_copy_pairs = {}

        for file in annotation_files:
            video_category_folder = file.replace("annotation", "core_dataset").replace(".txt", "")
            df = pd.read_csv(file, header=None)
            df.columns = ["Video_A", "Video_B", "Copy_Start_Video_A", "Copy_End_Video_A", "Copy_Start_Video_B",
                          "Copy_End_Video_B"]
            df = df[["Video_A", "Video_B"]]
            df["Video_A"] = df["Video_A"].apply(lambda x: os.path.join(video_category_folder, x))
            df["Video_B"] = df["Video_B"].apply(lambda x: os.path.join(video_category_folder, x))
            video_pairs = list(df.itertuples(index=False, name=None))
            video_category_copy_pairs[video_category_folder] = video_pairs

        return video_category_copy_pairs

    def get_video_categories(self):
        videos_annotations_dir_path = os.path.join(self.vcdb_dataset_folder, "annotation")
        annotation_files = glob.glob(videos_annotations_dir_path + "/*")
        return annotation_files

    @staticmethod
    def read_video_frames(video_file_path: str, num_frames_per_sec: int, pre_process_network: str, max_frames: int):
        cap = cv2.VideoCapture(video_file_path)

        if not cap.isOpened():
            if ".mp4" in video_file_path:
                video_file_path = video_file_path.replace(".mp4", ".flv")
                cap = cv2.VideoCapture(video_file_path)
            elif ".flv" in video_file_path:
                video_file_path = video_file_path.replace(".flv", ".mp4")
                cap = cv2.VideoCapture(video_file_path)
            else:
                raise RuntimeError(f"Video {video_file_path} not readable.")

        fps = round(cap.get(cv2.CAP_PROP_FPS))
        if fps == 0:
            print(f"fps of video {video_file_path} is {fps}. Setting default to 30.")
            fps = 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        video_duration_secs = round(total_frames / fps)

        num_frames_to_sample = num_frames_per_sec * video_duration_secs
        if num_frames_to_sample > max_frames:
            num_frames_to_sample = max_frames
        intervals = np.linspace(0, total_frames - 1, num_frames_to_sample, dtype=int)

        frame_index = [np.random.randint(intervals[i], intervals[i+1]+1) for i in range(0, len(intervals)-1)]

        if pre_process_network == ImageNetwork.INCEPTION_V3.value:
            pre_process_fn = preprocess_frame_inception_v3

        elif pre_process_network == ImageNetwork.RESNET_50.value:
            pre_process_fn = preprocess_frame_resnet_50

        elif pre_process_network == ImageNetwork.VGG_19.value:
            pre_process_fn = preprocess_frame_vgg_19

        else:
            raise ValueError(f"{pre_process_network} is not supported. Choose either "
                             f"{ImageNetwork.INCEPTION_V3.value} or {ImageNetwork.RESNET_50.value}")

        frame_array = []
        for idx, frame_idx in enumerate(frame_index):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                processed_frame = pre_process_fn(frame)
                frame_array.append(processed_frame)
            else:
                print(f"Unable to read frame #{frame_idx} of video '{video_file_path}'. Skipping and Continuing.!!")
                continue
        return np.array(frame_array, dtype=float)

    def get_copy_video_pairs_for_category(self, video_category_folder: str):
        num_videos = len(self.video_category_copy_pairs[video_category_folder])
        rand_index = np.random.randint(0, num_videos)
        return self.video_category_copy_pairs[video_category_folder][rand_index]

    def create_data_generator(self, batch_size: int, pre_process_network: str, num_frames_per_sec: int = 1, max_frames: int = 100):
        video_categories = self.get_video_categories()
        video_category_folders = [file.replace("annotation", "core_dataset").replace(".txt", "") for file in video_categories]
        num_video_categories = len(video_category_folders)
        if batch_size > num_video_categories:
            raise RuntimeError(f"Batch Size: {batch_size} can not be greater than number of video categories: {num_video_categories}")

        while True:
            for i in range(0, num_video_categories, batch_size):
                start = i
                end = i + batch_size
                if end > num_video_categories:
                    end = num_video_categories
                video_folders = video_category_folders[start:end]

                videos_to_read = [self.get_copy_video_pairs_for_category(folder) for folder in video_folders]
                videos_to_read_1 = np.array([video_pairs[0] for video_pairs in videos_to_read])
                videos_to_read_2 = np.array([video_pairs[1] for video_pairs in videos_to_read])

                vectorized_load_fn = np.vectorize(self.read_video_frames, excluded=["num_frames_per_sec", "pre_process_network", "max_frames"], otypes=[np.ndarray])

                video_batch_1 = vectorized_load_fn(videos_to_read_1, num_frames_per_sec=num_frames_per_sec, pre_process_network=pre_process_network, max_frames=max_frames)
                video_batch_2 = vectorized_load_fn(videos_to_read_2, num_frames_per_sec=num_frames_per_sec, pre_process_network=pre_process_network, max_frames=max_frames)

                video_batch_1 = tf.ragged.stack([tf.convert_to_tensor(arr) for arr in video_batch_1], axis=0)
                video_batch_2 = tf.ragged.stack([tf.convert_to_tensor(arr) for arr in video_batch_2], axis=0)

                yield video_batch_1, video_batch_2

    def create_test_video_pairs(self, pos_per_folder: int = 5):
        video_categories = self.get_video_categories()
        pos_pairs_list = []
        for file in video_categories:
            df = pd.read_csv(file, header=None)
            df.columns = ["Video_A", "Video_B", "Copy_Start_Video_A", "Copy_End_Video_A", "Copy_Start_Video_B", "Copy_End_Video_B"]
            df_sample = df.sample(n=pos_per_folder)
            folder = file.replace("annotation", "core_dataset").replace(".txt", "")
            df_sample["Video_A"] = df_sample["Video_A"].apply(lambda x: os.path.join(folder, x))
            df_sample["Video_B"] = df_sample["Video_B"].apply(lambda x: os.path.join(folder, x))

            pos_pairs_list += list(zip(df_sample["Video_A"], df_sample["Video_B"]))

        video_category_folders = [file.replace("annotation", "core_dataset").replace(".txt", "") for file in video_categories]

        candidate_negs = []
        for video_folder in video_category_folders:
            video_files = glob.glob(video_folder + "/*")
            video_file = random.sample(video_files, 1)
            candidate_negs.append(video_file[0])

        neg_pairs_list = [(a, b) for idx, a in enumerate(candidate_negs) for b in candidate_negs[idx + 1:]]

        return pos_pairs_list, neg_pairs_list

