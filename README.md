# CNN-DL-MRI-water-fat-separation

This code solves the optimization problem of water/fat separation in MRI by reconstructing water, fat, field, and R2* from input complex GRE data with 6 echoes using No-Training Deep Neural Network (NTD) method. For details see:
http://dx.doi.org/10.1002/mrm.28546




<p align="center">
  <img src="https://github.com/RaminJafari/DL-MRI-Water-Fat-Separation/blob/master/network.png" width="400" height="800" />
</p>




Download input_data.mat:
https://wcm.box.com/s/b1fiikn3shqy44v3eewfia8unit9ybse

To run the code, download all the files into a folder, make sure all the libraries listed in **main.py** exist and run:

```CUDA_VISIBLE_DEVICES=0 python main.py input_data.mat output_data.mat```

To plot results in MATLAB run **run_results.m**

