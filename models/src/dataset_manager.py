# src/dataset_manager.py

import torch
from torch.utils.data import Dataset
import json
import os
from .config import Config  # 레이블 매핑 등이 정의된 설정 파일


class DatasetManager(Dataset):
    def __init__(self, path: str, tokenizer, max_length: int = 128):
        """
        데이터셋 관리 클래스.

        :param path: 데이터셋 JSON 파일 경로.
        :param tokenizer: Hugging Face 토크나이저.
        :param max_length: 토큰화 시 최대 시퀀스 길이.
        """
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = Config.LABEL2ID
        self.id2label = Config.ID2LABEL
        self.num_labels = len(self.label2id)

        # 데이터 로드 및 전처리
        self.encodings, self.labels = self.load_and_preprocess()

    def load_and_preprocess(self):
        # 데이터 로드
        data = self.load_data(self.path)

        # 데이터 전처리
        texts = [item['발화'] for item in data]
        labels = [self.label2id.get(item['label'], -1) for item in data]

        # 레이블 매핑 검증
        if -1 in labels:
            missing_labels = set(item['label'] for item, label in zip(data, labels) if label == -1)
            raise ValueError(f"매핑되지 않은 레이블이 있습니다: {missing_labels}")

        # 토큰화
        encodings = self.tokenizer(
            texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return encodings, labels

    def load_data(self, path: str) -> list:
        if not os.path.exists(path):
            raise FileNotFoundError(f"데이터를 찾을 수 없음: {path}")

        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return data

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class DatasetLoader:
    def __init__(self, train_path: str, val_path: str, tokenizer, max_length: int = 128):
        self.train_path = train_path
        self.val_path = val_path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def get_datasets(self) -> dict:
        datasets = {}
        if self.train_path:
            print("훈련 데이터 로드 중...")
            train_dataset = DatasetManager(self.train_path, self.tokenizer, self.max_length)
            datasets['train'] = train_dataset
            print("훈련 데이터 전처리 완료.")

        if self.val_path:
            print("검증 데이터 로드 중...")
            val_dataset = DatasetManager(self.val_path, self.tokenizer, self.max_length)
            datasets['val'] = val_dataset
            print("검증 데이터 전처리 완료.")

        print("데이터셋 로드 및 전처리 완료.")
        return datasets