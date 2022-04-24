import argparse, datetime, os, time, sys

import torch

sys.path.append("..")
from model.unet import rsppUnet
from eval_loader import get_dataset
from eval_utils import generate_segmentations, seed_torch

parser = argparse.ArgumentParser(description="Brats Evaluation")
parser.add_argument("--batch_size", default=1, help="batch size, default=1")
parser.add_argument("--seed", default=12536, help="seeds")
parser.add_argument("--model", default=False, help="set false while evaluating")
parser.add_argument("--activation", default="leakyrelu", help="activation")
parser.add_argument(
    "--normalization", default="group_normalization", help="normalization"
)
parser.add_argument("--num_groups", default=16, help="number of groups")
parser.add_argument("--wt_std", default=False, help="weight standardization")
parser.add_argument("--ckpt_path", help="checkpoint path")
parser.add_argument("--data_path", help="/path/to/validation_data/")
# parser.add_argument('--device', default="cuda:0", type=str,
#                     help='Set the CUDA_VISIBLE_DEVICES env var from this string')
parser.add_argument("--com", default="none", help="comment")


def evaluate(args):
    start = time.time()
    seed_torch(args.seed)
    print("GPU Name: ", torch.cuda.get_device_name(0))
    args.device = "cuda:0"
    #     current_exp_time = datetime.now().strftime("%Y%m%d_%T").replace(":", "")
    current_time = datetime.datetime.now()
    subdir = (
        str(current_time.day)
        + "-"
        + str(current_time.month)
        + "-"
        + str(current_time.year)
        + "__"
    )
    time_name = (
        str(current_time.hour)
        + "_"
        + str(current_time.minute)
        + "_"
        + str(current_time.second)
    )
    current_exp_time = subdir + time_name
    save_folder = f"/nfs/Workspace/brats_brain_segmentation/src/val_maps/{current_exp_time}_{args.com}"
    os.makedirs(save_folder, exist_ok=True)

    model_config = {
        "input_shape": (args.batch_size, 4, [155, 240, 240]),  # [240, 240, 155]
        "output_channel": 3,
        "n_labels": 3,
    }

    model = rsppUnet(model_config, args)
    model = model.to(args.device)
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt["state_dict"])
    # model.cuda()
    dataset = get_dataset(args)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, num_workers=8
    )

    os.makedirs(f"{save_folder}/visuals", exist_ok=True)

    generate_segmentations(
        loader, model, args.device, save_folder, verbose=False, visualize=False
    )
    print(f"Evaluation done in: {(time.time()-start)/60} mins")


if __name__ == "__main__":
    arguments = parser.parse_args()
    evaluate(arguments)
