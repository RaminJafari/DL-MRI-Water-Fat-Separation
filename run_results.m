
clear
close all
s=load('G:\2019\DL\publication_results\paper\1st_revision\github\output_data_github4_clean.mat');

s.test_pd = permute(s.test_pd,[2 3 1 4]);
s.test_gt = permute(s.test_gt,[2 3 1 4]);

water_abs_pd = s.test_pd(:,:,:,1)*s.w_std_r+s.w_mean_r;
fat_abs_pd = s.test_pd(:,:,:,2)*s.f_std_r+s.f_mean_r;
water_phs_pd = s.test_pd(:,:,:,4)*s.w_std_i+s.w_mean_i;
fat_phs_pd = s.test_pd(:,:,:,5)*s.f_std_i+s.f_mean_i;
field_pd = s.test_pd(:,:,:,6)*s.frq_std+s.frq_mean;
r2s_pd = s.test_pd(:,:,:,3)*s.r2_std+s.r2_mean;

water_abs_gt = s.test_gt(:,:,:,1)*s.w_std_r+s.w_mean_r;
fat_abs_gt = s.test_gt(:,:,:,2)*s.f_std_r+s.f_mean_r;
water_phs_gt = s.test_gt(:,:,:,3)*s.w_std_i+s.w_mean_i;
fat_phs_gt = s.test_gt(:,:,:,4)*s.f_std_i+s.f_mean_i;
field_gt = s.test_gt(:,:,:,5)*s.frq_std+s.frq_mean;
r2s_gt = s.test_gt(:,:,:,6)*s.r2_std+s.r2_mean;

mask = permute(s.mask,[2 3 1]);
pd = cat(4,water_abs_pd.*mask,fat_abs_pd.*mask,water_phs_pd.*mask,fat_phs_pd.*mask,field_pd.*mask,r2s_pd.*mask);
gt = cat(4,water_abs_gt.*mask,fat_abs_gt.*mask,water_phs_gt.*mask,fat_phs_gt.*mask,field_gt.*mask,r2s_gt.*mask);

slc = 22;
figure;
subplot(221)
imshow([gt(:,:,slc,1) pd(:,:,slc,1)])
title('Water (Reference vs NTD)')
subplot(222)
imshow([gt(:,:,slc,2) pd(:,:,slc,2)])
title('Fat (Reference vs NTD)')
subplot(223)
imshow([gt(:,:,slc,5) pd(:,:,slc,5)])
caxis([-50 100])
title('Field (Reference vs NTD)')
subplot(224)
imshow([gt(:,:,slc,6) pd(:,:,slc,6)])
caxis([0 200])
title('R2* (Reference vs NTD)')