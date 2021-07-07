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
- scipy
- opencv
- matplotlib
- scikit-image
- pydicom
- sklearn

```
pip install git+https://github.com/MELSunny/lungmask
```

```
refer to the requirement.txt for more specification 
```

## Prepare the dataset

Unzip the train val test RAR files to **DECOMPRESS_DIRECTORY**

The files and folders in **DECOMPRESS_DIRECTORY** are like this arch:  
* DECOMPRESS_DIRECTORY  
    * train  
        * covid  
            * ct_scan_0  
            * etc  
        * non-covid 
            * ct_scan_0  
            * etc  
    * val  
        * covid  
            * ct_scan_0  
            * etc  
        * non-covid 
            * ct_scan_0  
            * etc  
    * test
        * 0a4e7be0-3f5f-4f0e-90be-2774b101a47b
        * etc  

Set **DECOMPRESS_DIRECTORY** and **save_path** in dataloader.py and run dataloader.py  
You can get the progress and log in save_path/run.log

## Trained-model
The model is available at [link](https://drive.google.com/file/d/1BcyRDM3g9CZTkIa3cS8e21RqDEnJzGVr/view?usp=sharing)


## COVNet
<img src="images/demo.jpg" width="200"> <img src="images/demo1.jpg" width="200"> <img src="images/demo2.jpg" width="200">

### Acknowledgement 
This implementation is highly relied on the repos <br>[Swin-Transformer](https://github.com/microsoft/Swin-Transformer/blob/main/get_started.md) <br>
[COVNet](https://github.com/bkong999/COVNet)
<br>
[lungmask](https://github.com/JoHof/lungmask)  
Thanks for all these great work
