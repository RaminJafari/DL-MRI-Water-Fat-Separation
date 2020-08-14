# CNN-DL-MRI-water-fat-separation

This code will solve the MRI water/fat separation problem by reconstrucitng water, fat, field, and R2* from  input complex GRE data with 6 echoes using No-Training DNN (NTD) mehtod. For details please see:
https://arxiv.org/abs/2004.07923







```CUDA_VISIBLE_DEVICES=5 python main.py input_data.mat output_data.mat```

To run the code, please download all the files into a folder, make sure all the libraries listed in main.py exist and run:


<p align="center">
  <img src="https://github.com/RaminJafari/DL-MRI-Water-Fat-Separation/blob/master/network.png" width="400" height="800" />
</p>
