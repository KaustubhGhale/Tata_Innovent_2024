from box import Box

config = {
    "num_devices": 1,
    "batch_size": 2,
    "num_workers": 4,
    "num_epochs": 10,
    "eval_interval": 2,
    "out_dir": "out/training",
    "opt": {
        "learning_rate": 8e-4,
        "weight_decay": 1e-4,
        "decay_factor": 10,
        "steps": [60000, 86666],
        "warmup_steps": 250,
    },
    "model": {
        "type": 'vit_h',
        "checkpoint": "sam_vit_h_4b8939.pth",
        "freeze": {
            "image_encoder": True,
            "prompt_encoder": True,
            "mask_decoder": False,
        },
    },
    "dataset": {
        "train": {
            "root_dir": r"E:\Random Python Scripts\Tata HaxS\sam_l\sam.v2i.coco-segmentation\train",
            "annotation_file": r"E:\Random Python Scripts\Tata HaxS\sam_l\sam.v2i.coco-segmentation\train\_annotations.coco.json"
        },
        "val": {
            "root_dir": r"E:\Random Python Scripts\Tata HaxS\sam_l\sam.v2i.coco-segmentation\valid",
            "annotation_file": r"E:\Random Python Scripts\Tata HaxS\sam_l\sam.v2i.coco-segmentation\valid\_annotations.coco.json"
        }
    }
}

cfg = Box(config)
