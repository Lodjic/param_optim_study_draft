# Author : Loïc Thiriet

import json
import random
from pathlib import Path
from typing import Literal

import albumentations as A
import cv2
import numpy as np
import polars as pl
import torch
from albumentations.pytorch.transforms import ToTensorV2
from torch.utils.data import DataLoader, Dataset

###################### Custom Dataset class ######################

POLARS_GTS_SCHEMA = {
    "id": pl.UInt32,
    "sub_id": pl.UInt8,
    "img_name": pl.Utf8,  # pl.Utf8 because Colab does not have pl.String
    "bbox_xmin": pl.Int16,
    "bbox_ymin": pl.Int16,
    "bbox_xmax": pl.Int16,
    "bbox_ymax": pl.Int16,
    "label": pl.Int32,
}


class CustomDataset(Dataset):
    """
    Customized Dataset class herited from Dataset class from PyTorch that needs all images
    in a subdirctory '../imgs' and some arguments descibred there after. A CustomDataset
    needs to be intiaited for each of the train, validation and test sets with their
    corresponding list of image names.
    The Dataset class (from PyTorch) requires at least a init, len and getitem function.

    Args:
        data_dir (str): directory where the data is located
        file_names_list (list): list of the images names of the current dataset
        img_name_to_class (dict): dictionnary that map the images name to their class id
    """

    def __init__(self, data_dir: Path, images_extension: str = ".png"):
        # Images directory
        self.imgs_dir = data_dir / "imgs"
        # Initiate a list with path of the image
        self.imgs_path_list = [Path(img_path) for img_path in list(self.imgs_dir.glob(f"*{images_extension}"))]
        # Load the ground truth pre-computed csv
        # with open(data_dir / "annotations/gts.json", "r") as geojsons_file:
        #     self.gts = json.load(geojsons_file)
        self.gts = pl.read_csv(data_dir / "annotations/gts.csv", has_header=True).cast(POLARS_GTS_SCHEMA, strict=True)

    def __len__(self):
        """Returns the number of images in the CustomDataset.

        Returns:
            int: number of images in the CustomDataset
        """
        return len(self.imgs_path_list)

    def __getitem__(self, idx):
        """Returns the image at index idx as a torch tensor reiszed to the input size of the network.

        Args:
            idx (int): index of the item wanted

        Returns:
            torch.tensor: image
            int: class id of the image
        """
        img_path = self.imgs_path_list[idx]
        img_targets = self.gts.filter(pl.col("img_name") == img_path.name)

        target = {}
        target["bboxes"] = img_targets[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].to_numpy()
        target["labels"] = img_targets[["label"]].to_numpy()

        # Because there are some images that have 4 channels and could not  make it work with cv2
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # because cv2 opens images in BGR and not RGB
        # Resize and scale the image
        transform = A.Compose(
            [
                A.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), max_pixel_value=255
                ),  # Normalized and transformed into float32
                ToTensorV2(),  # transforms to torch.tensor + permute axis dimensions : (height, width, channels) -> (channels, height, width)
            ]
        )
        img = transform(image=img)["image"]
        bboxes = torch.tensor(data=target["bboxes"])
        labels = torch.tensor(data=target["labels"])

        return img, {"boxes": bboxes, "labels": labels}, img_path


class CustomDatasetWithAugmentation(Dataset):
    """
    Customized Dataset class herited from Dataset class from PyTorch that needs all images
    in a subdirctory '../imgs' and some arguments descibred there after. A CustomDataset
    needs to be intiaited for each of the train, validation and test sets with their
    corresponding list of image names.
    The Dataset class (from PyTorch) requires at least a init, len and getitem function.

    Args:
        data_dir (str): directory where the data is located
        file_names_list (list): list of the images names of the current dataset
        img_name_to_class (dict): dictionnary that map the images name to their class id
    """

    def __init__(self, data_dir: Path, images_extension: str = ".png"):
        # Images directory
        self.imgs_dir = data_dir / "imgs"
        # Initiate a list with path of the image
        self.imgs_path_list = [Path(img_path) for img_path in list(self.imgs_dir.glob(f"*{images_extension}"))]
        # Load the ground truth pre-computed csv
        # with open(data_dir / "annotations/gts.json", "r") as geojsons_file:
        #     self.gts = json.load(geojsons_file)
        self.gts = pl.read_csv(data_dir / "annotations/gts.csv", has_header=True).cast(POLARS_GTS_SCHEMA, strict=True)

    def __len__(self):
        """Returns the number of images in the CustomDataset.

        Returns:
            int: number of images in the CustomDataset
        """
        return len(self.imgs_path_list)

    def __getitem__(self, idx):
        """Returns the image at index idx as a torch tensor reiszed to the input size of the network.

        Args:
            idx (int): index of the item wanted

        Returns:
            torch.tensor: image
            int: class id of the image
        """
        img_path = self.imgs_path_list[idx]
        img_targets = self.gts.filter(pl.col("img_name") == img_path.name)

        target = {}
        target["bboxes"] = img_targets[["bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax"]].to_numpy()
        target["labels"] = img_targets["label"].to_numpy()

        # Because there are some images that have 4 channels and could not  make it work with cv2
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # because cv2 opens images in BGR and not RGB

        # Transforms the original image
        TRANSFORMATIONS = A.Compose(
            [
                # A.Rotate(limit=(-180, 180), interpolation=1, border_mode=0, value=0, p=0.9),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.augmentations.transforms.CLAHE(p=0.3),
                A.augmentations.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2, p=0.3),
                A.RandomBrightnessContrast(p=0.3),
                A.augmentations.transforms.Posterize(p=0.2),
                A.augmentations.transforms.ToGray(p=0.1),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", min_visibility=0.2, label_fields=["labels"]),
        )
        transformed_data = TRANSFORMATIONS(image=img, bboxes=target["bboxes"], labels=target["labels"])
        img = transformed_data["image"]
        target["bboxes"] = np.array(transformed_data["bboxes"])
        target["labels"] = np.array(transformed_data["labels"])[:, np.newaxis]

        # Normalize all images the same way
        classical_transform = A.Compose(
            [
                A.Normalize(
                    (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), max_pixel_value=255
                ),  # Normalized and transformed into float32
                ToTensorV2(),  # transforms to torch.tensor + permute axis dimensions : (H,W,C) -> (C,H,W)
            ]
        )

        img = classical_transform(image=img)["image"]
        bboxes = torch.tensor(data=target["bboxes"])
        labels = torch.tensor(data=target["labels"])

        return img, {"boxes": bboxes, "labels": labels}, img_path


#################### Function to automatically generate train, val and test Datasets & DataLoaders ####################


def generate_datasets(
    general_data_dir: Path,
    task_type: str,
    dataset_fct: Literal["val", "test"],
    images_extension: str = ".png",
):
    # Create a dict of training and validation datasets
    if task_type == "train":
        datasets = {"train": CustomDatasetWithAugmentation(general_data_dir / "train", images_extension)}
        datasets["val"] = CustomDataset(general_data_dir / "val", images_extension)
    # Creates 1 dataset for test or evaluation
    elif task_type == "predict":
        datasets = CustomDataset(general_data_dir / dataset_fct, images_extension)
    else:
        ValueError(f"The task_type='{task_type}' is not implemented yet for dataset generation.")
    return datasets


def collate_fn(batch):
    imgs = torch.stack([elt[0] for elt in batch])
    targets = [elt[1] for elt in batch]
    imgs_path = [elt[2] for elt in batch]
    return imgs, targets, imgs_path


def generate_dataloaders(
    root_data_dir: Path,
    task_type: Literal["train", "predict"],
    dataset_fct: Literal["val", "test"],
    images_extension: str = ".png",
    batch_size: int = 8,
    manual_seed: int | float | None = None,
):
    datasets = generate_datasets(root_data_dir, task_type, dataset_fct, images_extension)

    if manual_seed:
        # Seed worker function for reproducibility
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Seed generator for reproducibility
        generator = torch.Generator()
        generator.manual_seed(manual_seed)

    # Creates a dict of training and validation dataloaders
    if task_type == "train":
        dataloaders = {
            x: DataLoader(
                datasets[x],
                batch_size=batch_size,
                shuffle=True,
                collate_fn=collate_fn,
                pin_memory=True,
                worker_init_fn=seed_worker if manual_seed else None,
                generator=generator if manual_seed else None,
            )
            for x in list(datasets.keys())
        }
    # Creates 1 dataloader for test or evaluation
    elif task_type == "predict":
        dataloaders = DataLoader(datasets, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, pin_memory=True)
    else:
        raise ValueError(f"Parameter task_type='{task_type}' not implemented yet for dataloaders generation.")

    return dataloaders


#################### Function to send inputs and targets to device ####################


def send_inputs_and_targets_to_device(
    inputs: torch.Tensor, targets: list[dict[str : torch.Tensor]] | list, device: torch.device
) -> tuple[torch.Tensor, list[dict[str : torch.Tensor]]]:
    # Sends inputs to device
    inputs = inputs.to(device, non_blocking=True)

    # Sends lables to device
    for i, input_label in enumerate(targets):
        for label_key in input_label.keys():
            targets[i][label_key] = targets[i][label_key].to(device, non_blocking=True)

    return inputs, targets
