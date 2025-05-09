# https://github.com/christophschuhmann/improved-aesthetic-predictor/blob/main/simple_inference.py

import numpy as np, torch, transformers


class MLP(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.layers = torch.nn.Sequential(
            *[
                torch.nn.Linear(input_dim, 1024),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(1024, 128),
                torch.nn.Dropout(0.2),
                torch.nn.Linear(128, 64),
                torch.nn.Dropout(0.1),
                torch.nn.Linear(64, 16),
                torch.nn.Linear(16, 1),
            ]
        )

    def forward(self, x):
        return self.layers(x)

    @staticmethod
    def normalized(x, axis=-1, order=2):
        l2 = np.atleast_1d(np.linalg.norm(x, order, axis))
        l2[l2 == 0] = 1
        return x / np.expand_dims(l2, axis)


class Aesthetic:
    def __init__(self, _):
        clip_path = "/mnt/wsfuse/tsujuifu1996/_huggingface_dump/models--openai--clip-vit-large-patch14"
        self.model = (
            transformers.CLIPModel.from_pretrained(clip_path, local_files_only=True)
            .eval()
            .to("cuda")
        )
        self.pproc = transformers.AutoProcessor.from_pretrained(
            clip_path, local_files_only=True
        )

        aesthetic_path = "/mnt/wsfuse/tsujuifu1996/_external_checkpoint_dump/aesthetic_sac_logos_ava1_l14_linear_mse.pt"
        self.mlp = MLP(768).eval().to("cuda")
        _ = self.mlp.load_state_dict(torch.load(aesthetic_path, map_location="cpu"))

    def __call__(self, item):
        txt, img = "", item["image"]
        inp = self.pproc(
            text=[txt],
            images=img,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        pix_val = inp["pixel_values"].to("cuda")

        with torch.no_grad():
            f_img = self.model.get_image_features(pixel_values=pix_val.to("cuda"))
            f_img = MLP.normalized(f_img.cpu().numpy())
            score = self.mlp(torch.from_numpy(f_img).float().to("cuda"))[0].item()

        return score
