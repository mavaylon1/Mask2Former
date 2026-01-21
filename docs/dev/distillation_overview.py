"""
Distillation Overview
==============

This is a developer-focused guide for changes in Mask2Former to support distillation.
"""

#############
# The Configs
#############
# Overview
# --------
# For better or for worse, there is a system of yaml configs that define parameters across a range of datasets and tasks.
# It can be tricky to keep track of what is where, especially when there is a detectron2 component that requires a user to 
# venture to a whole other repository. Let's go over some useful changes and some requirements that went into the distillation config so 
# that users can make their own configs if needed.

# Weights
# -------
# You will have to go through the process of downloading the weights. If you want to store them in a volume, you need to update the path
# in the task specific configs.
```
WEIGHTS: "swin_tiny_patch4_window7_224.pkl"
to
WEIGHTS: "/home/jovyan/mask2former/pretrained_weights/swin/swin_tiny_patch4_window7_224.pkl"
```

# Introducing new Config Keys
# ---------------------------
# In order to use new keys in the config, they first need to be registered. Within `mask2fromer/config.py` you can declare a new key group with `CN()` and then populate with keys as follows:

```
cfg.KD = CN()
cfg.KD.FILE = ""
cfg.KD.TEMP = 2.0
cfg.KD.WEIGHT = .01
```

##############
# Data Mappers
##############
# To understand why we needed to create a custom data mapper class, let's first understand the full data loading infrastructure at a high level.

# i) Data on the filesystem
#    Images/Annotations (jpg/png/etc)

# ii) Dataset registration
#    In Detectron2, you register data using `DatasetCatalog`. This is a dictionary with the name of the data as the key and the function that returns a list of dicts (data records).
```
def register_ade20k_subset(n=50):
    base = DatasetCatalog.get("ade20k_sem_seg_train")
    return base[:n]

DatasetCatalog.register(
    "ade20k_sem_seg_train_subset",
    lambda: register_ade20k_subset(50),
)
```
#    The function needs to return a list of dictionaries of the entire dataset. The dictionary is similar to COCO's style where it contains the location
#    and certain metadata:
```
{
  "file_name": "/data/imgs/0001.jpg",
  "height": 1024,
  "width": 768,
  "annotations": [...]
}
```
# If the data is too large to index in memory, then that would need special attention that is not covered here.

# The mapper class is used to bridge where the data is on the file system and the loader. Refer to `build_train_loader` for more details.
# We brought in a new mapper not because we brought in a new dataset, but rather we have to contend with the cached teacher probabilites for each image.
# Rather than making edits to the existing native mapper for ade20k, it is safer to make a new one that will also resolve reading and retrieving from the teacher hdf5 file. Refer to `MaskFormerSemanticDatasetKDMapper`.

#############################
# Cache teacher probabilities
#############################
# We implemented a cache in the train_net to store the predictions for each data point. Notice that we are storing the semantic output probabilities, not the mask_cls_logits or mask_pred_logits.

###################
# Distillation Loss
###################
# The loss is defined in `mask2former/maskformer_model.py`. 


