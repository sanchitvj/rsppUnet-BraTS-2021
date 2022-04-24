import os
from tqdm import tqdm
from skimage.transform import resize  # not using, highly inefficient
import nibabel as nib
import numpy as np
from torch.utils.data import Dataset


class BratsDatasetEval(Dataset):
    def __init__(
        self,
        args,
        data,
    ):
        self.data = data
        self.data_types = [
            "_flair.nii.gz",
            "_t1.nii.gz",
            "_t1ce.nii.gz",
            "_t2.nii.gz",
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        root_path = self.data[idx]
        dir_name = root_path.split("/")[-1]
        patient_id = dir_name
        # load all modalities
        images = []
        for data_type in self.data_types:
            img_name = dir_name + data_type
            img_path = os.path.join(root_path, img_name)
            if data_type == "_t1.nii.gz":
                seg_path = img_path
            img = self.load_img(img_path)  # .transpose(2, 0, 1)
            img = self.normalize(img)
            images.append(img)

        img = np.stack(images)
        img = np.moveaxis(img, (0, 1, 2, 3), (0, 3, 2, 1))
        img = img.astype(np.float32)

        return {"image": img, "id": patient_id, "seg_path": seg_path}

    def load_img(self, file_path):
        img = nib.load(file_path)
        data = np.asarray(img.dataobj)
        return data

    def normalize(self, data: np.ndarray, low_perc=1, high_perc=99):
        """Main pre-processing function used for the challenge (seems to work the best).
        Remove outliers voxels first, then min-max scale.
        Warnings
        --------
        This will not do it channel wise!!
        """
        non_zeros = data > 0
        low, high = np.percentile(data[non_zeros], [low_perc, high_perc])
        image = np.clip(data, low, high)

        data_min = np.min(data)
        return (data - data_min) / (np.max(data) - data_min)


def get_dataset(args):
    train_dir = []
    for filename in os.listdir(args.data_path):
        f = os.path.join(args.data_path, filename)
        train_dir.append(f)

    eval_dataset = BratsDatasetEval(args, train_dir)

    return eval_dataset
