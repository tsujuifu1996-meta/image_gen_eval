import argparse, glob, json, os, time, tqdm

import numpy as np, torch

from PIL import Image


class Data(torch.utils.data.Dataset):
    def __init__(self, dir):
        super().__init__()

        self.lst = sorted(glob.glob(f"{dir}/*"))

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        folder = self.lst[idx]

        try:
            txt = open(f"{folder}/text.txt", "r").read().strip()
            img = Image.open(f"{folder}/image.png").convert("RGB")
            return folder, txt, img
        except:
            print(f"== no text/image in {folder} ==")
            return None

    @staticmethod
    def collate(batch):  # always use batch_size=1
        return batch[0]


def all_gather(item):
    lst = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather_object(lst, item)
    return lst


def main(args):
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(backend="nccl")
    torch.distributed.barrier()

    args.eval_dir = os.path.normpath(args.eval_dir)
    args.dump_dir = os.path.normpath(args.dump_dir)
    if args.filename == "":
        args.filename = (
            f"{time.strftime('%Y%m%d%H%M%S', time.localtime())}_{args.metric}"
        )
    if int(os.environ["LOCAL_RANK"]) == 0:
        print(f"== args: {args} ==")
        os.makedirs(args.dump_dir, exist_ok=True)
    torch.distributed.barrier()

    data = Data(args.eval_dir)
    sampler = torch.utils.data.distributed.DistributedSampler(data, shuffle=False)
    loader = torch.utils.data.DataLoader(
        data, collate_fn=Data.collate, batch_size=1, num_workers=4, sampler=sampler
    )
    if int(os.environ["LOCAL_RANK"]) == 0:
        loader = tqdm.tqdm(loader, ascii=True)
    torch.distributed.barrier()

    if args.metric == "clip_score":
        from metric_t2i.clip_score import CLIP_Score as Metric
    elif args.metric == "aesthetic":
        from metric_t2i.aesthetic import Aesthetic as Metric
    elif args.metric == "pick_score":
        pass
    elif args.metric == "hps_v2":
        pass
    elif args.metric == "vqa_rating":
        pass
    else:
        raise NotImplementedError(f"== {args.metric} is not supported ==")
    metric = Metric(args.benchmark)
    torch.distributed.barrier()

    raw_res = {}
    for item in loader:
        if item is None:
            continue
        folder, txt, img = item
        raw_res.update({folder: metric({"text": txt, "image": img})})
    torch.distributed.barrier()

    raw_res = all_gather(raw_res)
    raw_res = {f: v for sub in raw_res for f, v in sub.items()}
    res = np.mean([v for _, v in raw_res.items()]).item()
    torch.distributed.barrier()

    if int(os.environ["LOCAL_RANK"]) == 0:
        json.dump(
            {"args": vars(args), "result": res, "raw_result": raw_res},
            open(f"{args.dump_dir}/{args.filename}.json", "w"),
            indent=4,
        )
        print(f"== {args.eval_dir}: {args.metric}={res} ==")
    torch.distributed.barrier()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_dir", type=str, required=True)
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=["clip_score", "aesthetic", "pick_score", "hps_v2", "vqa_rating"],
    )
    parser.add_argument("--dump_dir", type=str, required=True)
    parser.add_argument("--benchmark", type=str, default="")
    parser.add_argument("--filename", type=str, default="")
    args = parser.parse_args()

    main(args)
