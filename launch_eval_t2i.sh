torchrun --nproc_per_node=8 eval_t2i.py \
    --eval_dir="/mnt/wsfuse/4b_mot_u16g4_512p/victorialin/l4_mot_diff/chinch_4.5b_mot_u16g4_diffusion_dcae_unfreezing_backbone_n32/l4_mot_diff_chinch_4_5b_mot_u16g4_diffusion_dcae_unfreezing_b-bp64zs/eval/0148000/eval/t2i_cfg8/00000000" \
    --metric="vqa_rating" \
    --dump_dir="/home/tsujuifu1996/_result" \
    --filename="e_vqa_rating.json"
