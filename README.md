# Cascade of Encoder-Decoder CNNs with Learned Coordinates Regressor for Robust Facial Landmarks Detection

We provide C++ code in order to replicate the experiments in our paper
https://doi.org/10.1016/j.patrec.2019.10.012

If you use this code for your own research, you must reference our PRL paper:

```
Cascade of Encoder-Decoder CNNs with Learned Coordinates Regressor for Robust Facial Landmarks Detection
Roberto Valle, José M. Buenaposada and Luis Baumela.
Pattern Recognition Letters, PRL 2019.
```

#### Requisites
- faces_framework https://github.com/bobetocalo/faces_framework

#### Installation
This repository must be located inside the following directory:
```
faces_framework
    └── alignment
        └── bobetocalo_prl19
```
You need to have a C++ compiler (supporting C++11):
```
> mkdir release
> cd release
> cmake ..
> make -j$(nproc)
> cd ..
```
#### Usage
Use the --database option to load the proper trained model.
```
> ./release/face_alignment_bobetocalo_prl19_test --database 300w_public
```
