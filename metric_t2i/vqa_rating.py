import torch, transformers


class VQA_Rating:
    def __init__(self, _):
        llama_path = "/mnt/wsfuse/tsujuifu1996/_huggingface_dump/models--meta-llama--llama-3.2-11b-vision-instruct"
        self.model = (
            transformers.MllamaForConditionalGeneration.from_pretrained(
                llama_path, torch_dtype=torch.bfloat16
            )
            .eval()
            .to("cuda")
        )
        self.pproc = transformers.AutoProcessor.from_pretrained(llama_path)

    def __call__(self, item):
        txt, img = item["text"], item["image"]
        txt = f"""You are an expert at image annotation for AI training. You are given an image and the following text prompt:
        {txt}

        On a scale of 1-5, score "does the image match the prompt?".

        Guidelines:
            • The ranking of each image given the same text input
            is important. If you believe the current scoring criteria
            cannot reflect your ranking preference, pick scores that
            are consistent with your ranking. Ties are allowed.
            • To evaluate the generated image, there are two aspects:
            image quality and text-image match. Here we only
            care about text-image match, which is referred to as
            “faithfulness”.
            • There are several kinds of elements in the text: object,
            attribute, relation, and context. Measure the consistency
            by counting how many elements are missed/misrepresented in the generated image.
            • For some elements, e.g. “train conductor’s hat", if you
            can see there is a hat but not a train conductor’s hat,
            consider half of the element is missed/misrepresented
            in the generated image.
            • Objects are the most important elements. If an object
            is missing, then consider all related attributes, activity,
            and attributes missing.
            • When you cannot tell what the object/attribute/activity/context is, consider the element missing. (e.g., can’t
            tell if an object is a microwave)
            Given the above guideline, suppose the text input contains
            n elements, and x elements are missed or misrepresented.
            n and x are all counted by the annotators. The reference
            scoring guideline is as follows:
            • 5: The image perfectly matches the prompt.
            • 4: x ≤ 2 and x ≤ n/3. A few elements are missed/misrepresented.
            • 3: More than n/3 (or 2) but less than n/2 elements are missed/misrepresented.
            • 2: x > n/2. More than half of the elements are
            missed/misrepresented.
            • 1: None of the major objects are correctly presented in
            the image.

        Only return the score. Do not include any explanation or additional words.
        """
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": txt},
                    {"type": "image"},
                ],
            }
        ]
        inp = self.pproc.apply_chat_template(message, add_generation_prompt=True)
        inp = self.pproc(img, inp, add_special_tokens=False, return_tensors="pt").to(
            "cuda"
        )

        out = self.model.generate(**inp, max_new_tokens=16)
        out = (
            self.pproc.decode(out[0])
            .split("<|eot_id|><|start_header_id|>assistant<|end_header_id|>")[-1]
            .strip()
        )
        try:
            score = int("".join([c for c in out if c.isdigit()]))
        except:
            score = 3

        return score
