# CNN-DL-MRI-water-fat-separation

This code solves the optimization problem of water/fat separation in MRI by reconstrucitng water, fat, field, and R2* from input complex GRE data with 6 echoes using No-Training Deep Neural Networks (NTD) method. For details please see:
https://arxiv.org/abs/2004.07923




<p align="center">
  <img src="https://github.com/RaminJafari/DL-MRI-Water-Fat-Separation/blob/master/network.png" width="400" height="800" />
</p>




Download input_data.mat:
https://wcm.box.com/s/b1fiikn3shqy44v3eewfia8unit9ybse

To run the code, please download all the files into a folder, make sure all the libraries listed in **main.py** exist and run:

```CUDA_VISIBLE_DEVICES=5 python main.py input_data.mat output_data.mat```

To plot results in MATLAB run **run_results.m**

