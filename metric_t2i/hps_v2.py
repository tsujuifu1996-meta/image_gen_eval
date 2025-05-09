import open_clip, torch, transformers


class HPS_v2:
    def __init__(self, _):
        open_clip_path = "/mnt/wsfuse/tsujuifu1996/_external_checkpoint_dump/hps_v2.pt"
        self.model, _, self.pproc = open_clip.create_model_and_transforms("ViT-H-14")
        self.model.load_state_dict(torch.load(open_clip_path, map_location="cpu"))
        _ = self.model.eval().to("cuda")
        self.tokenizer = open_clip.get_tokenizer("ViT-H-14")

    def __call__(self, item):
        txt, img = item["text"], item["image"]
        txt, img = self.tokenizer(txt)[0], self.pproc(img)
        txt, img = txt.unsqueeze(dim=0).to("cuda"), img.unsqueeze(dim=0).to("cuda")

        with torch.no_grad():
            f_txt = self.model.encode_text(txt)
            f_img = self.model.encode_image(img)
            score = (
                torch.nn.functional.cosine_similarity(f_txt, f_img, dim=-1)
                .clamp(min=0.0)[0]
                .item()
            )

        return score
