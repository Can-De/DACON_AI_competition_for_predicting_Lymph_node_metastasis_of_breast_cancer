import os
# from PIL import Image
import cv2
import torch
# import torchvision
import albumentations as A 
from albumentations.pytorch.transforms import ToTensorV2  # 이미지 tenser화
from tqdm.auto import tqdm
import json
# import pandas as pd

# from base line in DACON
transform = A.Compose(
    [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Rotate(limit=90, border_mode=cv2.BORDER_CONSTANT, p=0.3),
        A.Resize(128, 128),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            always_apply=False,
            p=1.0,
        ),
        ToTensorV2(),
    ]
)
# task: 이미지분할 증식
transform_test = A.Compose(
    [
        A.Resize(128, 128),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            max_pixel_value=255.0,
            always_apply=False,
            p=1.0,
        ),
        ToTensorV2(),
    ]
)

class BreastDataset(torch.utils.data.Dataset):
    """Pytorch dataset api for loading patches and preprocessed clinical data of breast."""

    def __init__(
        self,
        with_label, # 훈련용인지 테스트용인지 구분해서 dataset 생성하기 위함(True or False)
        json_path,
        data_dir_path,
        clinical_data_path=None,
        is_preloading=True,
        transform=None,
    ):
        self.with_label = with_label
        self.data_dir_path = data_dir_path
        self.is_preloading = is_preloading
        self.transform = transform

        if clinical_data_path is not None:
            self.clinical_data_df = clinical_data_path.set_index("ID")
        else:
            self.clinical_data_df = None

        with open(json_path) as f:
            print(f"load data from {json_path}")
            self.json_data = json.load(f)
        if self.is_preloading:
            self.bag_tensor_list = self.preload_bag_data()

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        if self.with_label: # 훈련 세트만 label이 존재함 (테스트는 label 없음)
            label = int(self.json_data[index]["label"])
        else: pass
        patient_id = self.json_data[index]["ID"]
        patch_paths = self.json_data[index]["patch_paths"]

        data = {}
        if self.is_preloading:
            data["bag_tensor"] = self.bag_tensor_list[index]
        else:
            data["bag_tensor"] = self.load_bag_tensor(
                [os.path.join(self.data_dir_path, p_path) for p_path in patch_paths]
            )

        if self.clinical_data_df is not None:
            data["clinical_data"] = self.clinical_data_df.loc[patient_id].to_numpy()

        if self.with_label:
            data["label"] = label
        else: pass
        data["patient_id"] = patient_id
        data["patch_paths"] = patch_paths

        return data

    # 이미지 텐서화

    def preload_bag_data(self):
        """Preload data into memory"""
        bag_tensor_list = []
        from tqdm.auto import tqdm
        for item in tqdm(self.json_data, ncols=120, desc="Preloading bag data"):
            patch_paths = [
                os.path.join(self.data_dir_path, p_path)
                for p_path in item["patch_paths"]
            ]
            bag_tensor = self.load_bag_tensor(patch_paths)  # [N, C, H, W]
            bag_tensor_list.append(bag_tensor)

        return bag_tensor_list

    def load_bag_tensor(self, patch_paths):
        """Load a bag data as tensor with shape [N, C, H, W]"""

        patch_tensor_list = []
        for p_path in patch_paths:
            # patch = Image.open(p_path).convert("RGB")
            # patch_tensor = self.transform(patch)
            image = cv2.imread(p_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            patch_tensor = transform(image=image)["image"]  # [C, H, W]
            patch_tensor = torch.unsqueeze(patch_tensor, dim=0)  # [1, C, H, W]
            patch_tensor_list.append(patch_tensor)

        bag_tensor = torch.cat(patch_tensor_list, dim=0)  # [N, C, H, W]

        return bag_tensor
    
# 데이터로더
def init_dataloader_(base_dir,json_path_train,json_path_val,df_train_clinical_data):

    train_dataset = BreastDataset(
        with_label=True,
        json_path=json_path_train,
        data_dir_path=base_dir,
        clinical_data_path=df_train_clinical_data,
        is_preloading=True,
        transform=transform,
    )
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )

    val_dataset = BreastDataset(
        with_label=True,
        json_path=json_path_val,
        data_dir_path=base_dir,
        clinical_data_path=df_train_clinical_data,
        is_preloading=True,
        transform=transform_test,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=0
    )

    return train_loader, val_loader