# Bone age assessment

This implementation is base on the paper on the `Bone age assessment based on deep neural networks with annotation-free cascaded critical bone region extraction` by `Zhangyong Li et al`. The paper is available [here](https://www.frontiersin.org/articles/10.3389/frai.2023.1142895/full).

The implementation is done using the `Pytorch` framework. The model is trained on the `RSNA Bone Age` dataset. The dataset is available [here](https://www.kaggle.com/kmader/rsna-bone-age).

The dataset is expected in the folder `data/` in the root directory of the project. The dataset should be in the following format:

```
data/
    - boneage-test-dataset/
        - 1.png
        - 2.png
        - ...
    - boneage-train-dataset/
        - 1.png
        - 2.png
        - ...
```

# Goal

This is the first approach to create the algorithm for the bone age assessment. 
In the long run, I want to apply this to dolphin pectoral fins to assess their age.