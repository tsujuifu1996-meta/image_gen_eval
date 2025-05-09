import torch, transformers


class Pick_Score:
    def __init__(self, _):
        pick_score_path = "/mnt/wsfuse/tsujuifu1996/_huggingface_dump/models--yuvalkirstain--pickscore_v1"
        self.model = (
            transformers.AutoModel.from_pretrained(
                pick_score_path, local_files_only=True
            )
            .eval()
            .to("cuda")
        )
        clip_path = "/mnt/wsfuse/tsujuifu1996/_huggingface_dump/models--laion--clip-vit-h-14-laion2b-s32b-b79k"
        self.pproc = transformers.AutoProcessor.from_pretrained(
            clip_path, local_files_only=True
        )

    def __call__(self, item):
        txt, img = item["text"], item["image"]
        inp = self.pproc(
            text=[txt],
            images=img,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inp_ids, attn_mask, pix_val = (
            inp["input_ids"].to("cuda"),
            inp["attention_mask"].to("cuda"),
            inp["pixel_values"].to("cuda"),
        )

        with torch.no_grad():
            f_txt = self.model.get_text_features(
                input_ids=inp_ids, attention_mask=attn_mask
            )
            f_img = self.model.get_image_features(pixel_values=pix_val)
            score = (
                torch.nn.functional.cosine_similarity(f_txt, f_img, dim=-1)
                .clamp(min=0.0)[0]
                .item()
            )

        return score
