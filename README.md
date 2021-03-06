#COVID19T
This repo contains the implmentation of the paper- 'MIA-COV19D: A transformer-based framework for COVID19 classification in chest CTs'

<img src="images/framework.png" width="600">


### Notices
This is a part of work in the COV19D Competition
https://mlearn.lincoln.ac.uk/mia-cov19d/
### Citation
If you find this code is useful for your research, please consider citing:
```
@article{,
  title={MIA-COV19D: A transformer-based framework for COVID19 classification in chest CTs},
  author={Lei Zhang, Yan Wen},
  year={2021}
}
```

## Setup
### Prerequisites
- Annocoda
- python 3.8.10
- pytorch 1.8.1
- torchvision 0.9.1
- SimpleITK 2.0.2
- batchgenerators
- tensorboardX
- timm=0.4.9
```
refer to the requirement.txt for more specification 
```

## Trained-model
The model is available at [link](https://drive.google.com/file/d/1BcyRDM3g9CZTkIa3cS8e21RqDEnJzGVr/view?usp=sharing)


## COVNet
<img src="images/demo.jpg" width="200"> <img src="images/demo1.jpg" width="200"> <img src="images/demo2.jpg" width="200">

### Acknowledgement 
This implementation is highly relied on the repos <br>[Swin-Transformer](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md) <br>
[COVNet](https://github.com/bkong999/COVNet)
<br>
Thanks for all these great work
