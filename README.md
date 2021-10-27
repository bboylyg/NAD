# Neural Attention Distillation

This is an implementation demo of the ICLR 2021 paper **[Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks](https://openreview.net/pdf?id=9l0K4OM-oXE)** in PyTorch.

![Python 3.6](https://img.shields.io/badge/python-3.6-DodgerBlue.svg?style=plastic)
![Pytorch 1.10](https://img.shields.io/badge/pytorch-1.2.0-DodgerBlue.svg?style=plastic)
![CUDA 10.0](https://img.shields.io/badge/cuda-10.0-DodgerBlue.svg?style=plastic)
![License CC BY-NC](https://img.shields.io/badge/license-CC_BY--NC-DodgerBlue.svg?style=plastic)

## NAD: Quick start with pretrained model
We have already uploaded the `all2one` pretrained backdoor student model(i.e. gridTrigger WRN-16-1, target label 5) and the clean teacher model(i.e. WRN-16-1) in the path of `./weight/s_net` and `./weight/t_net` respectively. 

For evaluating the performance of  NAD, you can easily run command:

```bash
$ python main.py 
```
where the default parameters are shown in `config.py`.

The trained model will be saved at the path `weight/erasing_net/<s_name>.tar`

Please carefully read the `main.py` and `configs.py`, then change the parameters for your experiment.

### Erasing Results on BadNets  
- The setting of data augmentation for Finetuning and NAD in this table:
```
tf_train = transforms.Compose([
      transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
```   

| Dataset  | Baseline ACC | Baseline ASR | Finetuning ACC | Finetuning ASR | NAD ACC | NAD ASR |  
| -------- | ------------ | ------------ | ------- | ------- | ------- |------- | 
| CIFAR-10 | 85.65        | 100.0        | 82.32   | 18.13 | 82.12   | **3.57**  |  

---

## Training your own backdoored model
We have provided a `DatasetBD` Class in `data_loader.py` for generating training set of different backdoor attacks. 

For implementing backdoor attack(e.g. GridTrigger attack), you can run the below command:

```bash
$ python train_badnet.py
```
This command will train the backdoored model and print clean accuracies and attack rate. You can also select the other backdoor triggers reported in the paper. 

Please carefully read the `train_badnet.py` and `configs.py`, then change the parameters for your experiment.  

## How to get teacher model?  
we obtained the teacher model by finetuning all layers of the backdoored model using 5% clean data with data augmentation techniques. In our paper, we only finetuning the backdoored model for 5~10 epochs. Please check more details of our experimental settings in section 4.1 and Appendix A; The finetuning code is easy to get by just setting all the param `beta = 0`, which means the distillation loss to be zero in the training process.  

## Other source of backdoor attacks
#### Attack

**CL:** Clean-label backdoor attacks

- [Paper](https://people.csail.mit.edu/madry/lab/cleanlabel.pdf)
- [pytorch implementation](https://github.com/hkunzhe/label_consistent_attacks_pytorch)

**SIG:** A New Backdoor Attack in CNNS by Training Set Corruption Without Label Poisoning

- [Paper](https://ieeexplore.ieee.org/document/8802997/footnotes)

```python
## reference code
def plant_sin_trigger(img, delta=20, f=6, debug=False):
    """
    Implement paper:
    > Barni, M., Kallas, K., & Tondi, B. (2019).
    > A new Backdoor Attack in CNNs by training set corruption without label poisoning.
    > arXiv preprint arXiv:1902.11237
    superimposed sinusoidal backdoor signal with default parameters
    """
    alpha = 0.2
    img = np.float32(img)
    pattern = np.zeros_like(img)
    m = pattern.shape[1]
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                pattern[i, j] = delta * np.sin(2 * np.pi * j * f / m)

    img = alpha * np.uint32(img) + (1 - alpha) * pattern
    img = np.uint8(np.clip(img, 0, 255))

    #     if debug:
    #         cv2.imshow('planted image', img)
    #         cv2.waitKey()

    return img
```

**Refool**: Reflection Backdoor: A Natural Backdoor Attack on Deep Neural Networks

- [Paper](https://arxiv.org/abs/2007.02343)
- [Code](https://github.com/DreamtaleCore/Refool)
- [Project](http://liuyunfei.xyz/Projs/Refool/index.html)

#### Defense

**MCR**: Bridging Mode Connectivity in Loss Landscapes and Adversarial Robustness

- [Paper](https://arxiv.org/abs/2005.00060)
- [Pytorch implementation](https://github.com/IBM/model-sanitization)

**Fine-tuning & Fine-Pruning**: Defending Against Backdooring Attacks on Deep Neural Networks

- [Paper](https://link.springer.com/chapter/10.1007/978-3-030-00470-5_13)
- [Pytorch implementation1](https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master/defenses)
- [Pytorch implementation2](https://github.com/adityarajagopal/pytorch_pruning_finetune)

**Neural Cleanse**: Identifying and Mitigating Backdoor Attacks in Neural Networks

- [Paper](https://people.cs.uchicago.edu/~ravenben/publications/pdf/backdoor-sp19.pdf)
- [Tensorflow implementation](https://github.com/Abhishikta-codes/neural_cleanse)
- [Pytorch implementation1](https://github.com/lijiachun123/TrojAi)
- [Pytorch implementation2](https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master/defenses)

**STRIP**: A Defence Against Trojan Attacks on Deep Neural Networks

- [Paper](https://arxiv.org/pdf/1911.10312.pdf)
- [Pytorch implementation1](https://github.com/garrisongys/STRIP)
- [Pytorch implementation2](https://github.com/VinAIResearch/input-aware-backdoor-attack-release/tree/master/defenses)

#### Library

`Note`: TrojanZoo provides a universal pytorch platform to conduct security researches (especially backdoor attacks/defenses) of image classification in deep learning.

Backdoors 101 — is a PyTorch framework for state-of-the-art backdoor defenses and attacks on deep learning models. 

- [trojanzoo](https://github.com/ain-soph/trojanzoo)
- [backdoors101](https://github.com/ebagdasa/backdoors101)

## References

If you find this code is useful for your research, please cite our paper
```
@inproceedings{li2021neural,
  title={Neural Attention Distillation: Erasing Backdoor Triggers from Deep Neural Networks},
  author={Li, Yige and Lyu, Xixiang and Koren, Nodens and Lyu, Lingjuan and Li, Bo and Ma, Xingjun},
  booktitle={ICLR},
  year={2021}
}
```

## Contacts

If you have any questions, leave a message below with GitHub.

