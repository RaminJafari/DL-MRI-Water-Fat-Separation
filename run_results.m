
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

close all;
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
%%
% vis([permute(s.test_gt(slcs:slce,st:ed,:,:),[2 3 1 4]).*(Mask(st:ed,:,slcs:slce))....
%     permute(s.test_pd(slcs:slce,st:ed,:,:),[2 3 1 4]).*(Mask(st:ed,:,slcs:slce))])
%%
close all;
clear all;
matdir_src = 'G:\2019\DL\publication_results\paper\1st_revision\40cases_notraining'; %source directory with *.mat files
%matdir_tgt = 'G:\2019\Mark\data_2019\all_mask_test'; %target directory to write new results

pd_all=zeros(35,256,256,6);
gt_all=zeros(35,256,256,6);

filelist = dir(matdir_src);
for i=3:length(filelist)
 ld_dir = [matdir_src '\' filelist(i).name] 
 load([matdir_src '\' filelist(i).name]);

    %s1=load('7_result_halfweight')
    %s1=load('result_halfweight_allcases_unsuper')


    %%good one: result_halfweight_allcases_unsuper_1st.mat
    
    %s1=load('33_result_halfweight.mat')
    %s1=load('saved_weights_halfwtest5_all_case_unsupereight_unsuper-1000.hdf5.mat')
   % s1=load('result_halfweight_allcases_unsuper_1st')
    %s1=load('result_halfweight_allcases_super')
    
    %s1=load('30result.mat')
    %s1=load('30result_halfweight.mat')
    %s1=load('result_halfweight_allcases_unsuper')
    %s1=load('test5_all_case_unsuper_1st.mat')

tt = test_pd;
test_gt(:,:,:,1)= test_gt(:,:,:,1)*w_std_r+w_mean_r;
test_gt(:,:,:,2)= test_gt(:,:,:,2)*f_std_r+f_mean_r;
tt(:,:,:,1)= test_pd(:,:,:,1)*w_std_r+w_mean_r;
tt(:,:,:,2)= test_pd(:,:,:,2)*f_std_r+f_mean_r;

test_gt(:,:,:,5)= test_gt(:,:,:,5)*frq_std+frq_mean;
tt(:,:,:,5)= test_pd(:,:,:,6)*frq_std+frq_mean;
test_gt(:,:,:,6)= test_gt(:,:,:,6)*r2_std+r2_mean;
tt(:,:,:,6)= test_pd(:,:,:,3)*r2_std+r2_mean;

st=50;
ed=220;

pd_all = cat(1,pd_all,tt);
gt_all = cat(1,gt_all,test_gt);


end
% vis([permute(test_gt(:,st:ed,:,:),[2 3 1 4])....
%     permute(tt(:,st:ed,:,:),[2 3 1 4])])

%save -v7.3 40cases_notraining_results.mat 
%%
vis([permute(gt_all(:,st:ed,:,:),[2 3 1 4])....
    permute(pd_all(:,st:ed,:,:),[2 3 1 4])])
%%
 vis(cat(4,permute(gt_all(:,:,:,1),[2 3 1 4]),permute(pd_all(:,:,:,1),[2 3 1 4]),...
     permute(gt_all(:,:,:,2),[2 3 1 4]),permute(pd_all(:,:,:,2),[2 3 1 4]),...
     permute(gt_all(:,:,:,5),[2 3 1 4]),permute(pd_all(:,:,:,5),[2 3 1 4]),...
     permute(gt_all(:,:,:,6),[2 3 1 4]),permute(pd_all(:,:,:,6),[2 3 1 4])))

%%


load roi_data_new_40cases.mat
%load roi_data_h1_espcase.mat

gt=permute(gt_all(:,:,:,:),[2 3 1 4]);
l1=permute(pd_all(:,:,:,:),[2 3 1 4]);
l2=permute(pd_all(:,:,:,:),[2 3 1 4]);
l3=permute(pd_all(:,:,:,:),[2 3 1 4]);

pdff_gt = 100*(abs(gt(:,:,:,2))./(abs(gt(:,:,:,2))+abs(gt(:,:,:,1))));
pdff_l1 = 100*(abs(l1(:,:,:,2))./(abs(l1(:,:,:,2))+abs(l1(:,:,:,1))));
pdff_l2 = 100*(abs(l2(:,:,:,2))./(abs(l2(:,:,:,2))+abs(l2(:,:,:,1))));
pdff_l3 = 100*(abs(l3(:,:,:,2))./(abs(l3(:,:,:,2))+abs(l3(:,:,:,1))));

r2s_gt = abs(gt(:,:,:,6));
r2s_l1 = abs(l1(:,:,:,6));
r2s_l2 = abs(l2(:,:,:,6));
r2s_l3 = abs(l3(:,:,:,6));


fld_gt = (gt(:,:,:,5));
fld_l1 = (l1(:,:,:,5));
fld_l2 = (l2(:,:,:,5));
fld_l3 = (l3(:,:,:,5));


for i=1:size(roilist,2)
     Mask = roilist(i).mask; 
     Mask_3d = logical(zeros(size(pdff_gt)));
     roi_slice = roilist(i).slice;
     Mask_3d(:,:,roi_slice)=(Mask);
     
     pdff_gt_roi(i) = mean(pdff_gt(Mask_3d));
     pdff_l1_roi(i) = mean(pdff_l1(Mask_3d));
     pdff_l2_roi(i) = mean(pdff_l2(Mask_3d));
     pdff_l3_roi(i) = mean(pdff_l3(Mask_3d));
     
     r2s_gt_roi(i) = mean(r2s_gt(Mask_3d));
     r2s_l1_roi(i) = mean(r2s_l1(Mask_3d));
     r2s_l2_roi(i) = mean(r2s_l2(Mask_3d));
     r2s_l3_roi(i) = mean(r2s_l3(Mask_3d));
     
     fld_gt_roi(i) = mean(fld_gt(Mask_3d));
     fld_l1_roi(i) = mean(fld_l1(Mask_3d));
     fld_l2_roi(i) = mean(fld_l2(Mask_3d));
     fld_l3_roi(i) = mean(fld_l3(Mask_3d));

end
%%

%save -v7.3 notraining_40cases_forplot.mat
%load  notraining_40cases_forplot.mat

close all
tit = ''; % figure title
gnames = { }; % names of groups in data {dimension 1 and 2}
corrinfo = {'r2','eq'}; % stats to display of correlation scatter plot
BAinfo = {'',''}; % stats to display on Bland-ALtman plot
limits = 'auto'; % how to set the axes limits

if 1 % colors for the data sets may be set as:
	colors = 'kk';      % character codes
else
	colors = [0 0 0;... % or RGB triplets
		      0 0 0];
end
 markerSize = 10;
 fontSize = 15;
% Generate figure with symbols

%label = {'Reference-PDFF','PDFF','%'}; % Names of data sets
%[cr, fig, statsStruct] = BlandAltman([pdff_gt_roi]',[pdff_l1_roi]',label,tit,gnames,'corrInfo',corrinfo,'baInfo',BAinfo,'axesLimits',limits,'colors',colors,'markersize',markerSize,'showFitCI',' on');

%label = {'Reference-Field','Field','Hz'}; % Names of data sets
%[cr, fig, statsStruct] = BlandAltman([fld_gt_roi]',[fld_l1_roi]',label,tit,gnames,'corrInfo',corrinfo,'baInfo',BAinfo,'axesLimits',limits,'colors',colors,'markersize',markerSize,'showFitCI',' on');

label = {'Reference-R2*','R2*','Hz'}; % Names of data sets
[cr, fig, statsStruct] = BlandAltman([r2s_gt_roi]',[r2s_l1_roi]',label,tit,gnames,'corrInfo',corrinfo,'baInfo',BAinfo,'axesLimits',limits,'colors',colors,'markersize',markerSize,'showFitCI',' on');


%hold on 

 %[r2s_gt_roi'-r2s_l1_roi' [1:size(r2s_l1_roi,2)]']
%[cr, fig, statsStruct] = BlandAltman([pdff_gt_roi]',[pdff_l1_roi]'+0.1,label,tit,gnames,'corrInfo',corrinfo,'baInfo',BAinfo,'axesLimits',limits,'colors',colors,'markersize',markerSize,'showFitCI',' on');

% %%
% close all
% figure;
% subplot(131)
% mdl=fitlm([pdff_gt_roi]',[pdff_l1_roi]','linear');
% p1=plot(mdl,'Marker','o','MarkerSize',5,'color','red')
% p1(end-1,1).Visible='off'
% p1(end,1).Visible='off'
% p1(2).Color=[1 0 0];
% legend('off')
% txt1 = ['C1    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% hold on 
% mdl=fitlm([pdff_gt_roi]',[pdff_l2_roi]','linear');
% p2=plot(mdl,'Marker','o','MarkerSize',5,'color','blue')
% p2(end-1,1).Visible='off'
% p2(end,1).Visible='off'
% p2(2).Color=[0 0 1];
% legend('off')
% txt2 = ['C2    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% %legend(txt1,txt2,txt3)
% 
% hold on 
% mdl=fitlm([pdff_gt_roi]',[pdff_l3_roi]','linear');
% p3=plot(mdl,'Marker','o','MarkerSize',5,'color','green')
% p3(end-1,1).Visible='off'
% p3(end,1).Visible='off'
% p3(2).Color=[0 1 0];
% legend('off')
% txt3 = ['WF-DNN,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% %legend(txt1,txt2,txt3)
% 
% 
% legend([p1(1) p2(1) p3(1)],{txt1,txt2,txt3},'Location','northwest')
% 
% 
% xlabel('Reference-PDFF(%)')
% ylabel('PDFF(%)')
% title('')
%  %ylim([0 90])
%  %xlim([0 90])
%  
%  ylim([0 100])
%  xlim([0 100])
% pbaspect([1 1 1])
% set(gca,'FontSize',16)
% 
% 
% subplot(133)
% mdl=fitlm([r2s_gt_roi]',[r2s_l1_roi]','linear');
% p1=plot(mdl,'Marker','o','MarkerSize',5,'color','red')
% p1(end-1,1).Visible='off'
% p1(end,1).Visible='off'
% p1(2).Color=[1 0 0];
% legend('off')
% txt1 = ['C1    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% hold on  
% mdl=fitlm([r2s_gt_roi]',[r2s_l2_roi]','linear');
% p2=plot(mdl,'Marker','o','MarkerSize',5,'color','blue');
% p2(end-1,1).Visible='off'
% p2(end,1).Visible='off'
% p2(2).Color=[0 0 1];
% legend('off')
% txt2 = ['C2    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% 
% hold on  
% mdl=fitlm([r2s_gt_roi]',[r2s_l3_roi]','linear');
% p3=plot(mdl,'Marker','o','MarkerSize',5,'color','green');
% p3(end-1,1).Visible='off'
% p3(end,1).Visible='off'
% p3(2).Color=[0 1 0];
% legend('off')
% txt3 = ['WFF-DNN   ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% legend([p1(1) p2(1) p3(1)],{txt1,txt2,txt3},'Location','northwest')
% xlabel('Reference-R2*(Hz)')
% ylabel('R2*(Hz)')
% title('')
% pbaspect([1 1 1])
% set(gca,'FontSize',16)
% % xlim([10 100])
% % ylim([10 100])
% 
% xlim([0 150])
% ylim([0 150])
% 
% 
% 
% 
% 
% subplot(132)
% mdl=fitlm([fld_gt_roi]',[fld_l1_roi]','linear');
% p1=plot(mdl,'Marker','o','MarkerSize',5,'color','red')
% p1(end-1,1).Visible='off'
% p1(end,1).Visible='off'
% p1(2).Color=[1 0 0];
% legend('off')
% txt1 = ['C1    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% hold on  
% mdl=fitlm([fld_gt_roi]',[fld_l2_roi]','linear');
% p2=plot(mdl,'Marker','o','MarkerSize',5,'color','blue')
% p2(end-1,1).Visible='off'
% p2(end,1).Visible='off'
% p2(2).Color=[0 0 1];
% legend('off')
% txt2 = ['C2    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% 
% hold on  
% mdl=fitlm([fld_gt_roi]',[fld_l3_roi]','linear');
% p3=plot(mdl,'Marker','o','MarkerSize',5,'color','green')
% p3(end-1,1).Visible='off'
% p3(end,1).Visible='off'
% p3(2).Color=[0 1 0];
% legend('off')
% txt3 = ['WFF-DNN   ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% 
% 
% legend([p1(1) p2(1) p3(1)],{txt1,txt2,txt3},'Location','northwest')
% xlabel('Reference-Field(Hz)')
% ylabel('Field(Hz)')
% title('')
% pbaspect([1 1 1])
% % xlim([-60 80])
% % ylim([-60 80])
% xlim([10 90])
% ylim([10 90])
% set(gca,'FontSize',16)
% 
% st=50;
% ed=220;
% 
% %%
% clear
% filename='G:\2019\DL\esp_test_data\esp23'
% %filename='G:\B_new_2\BU_new\Liver_Iron_Quantification\exams\human\reproducibility_healthy_subjects\Gaiying\ge_mr1\scan1'
% [iField,voxel_size,matrix_size,CF,delta_TE,TE,B0_dir]=Read_DICOM(filename);
% iField = permute(iField,[2 1 3 4]);
% Mask1= genMaskRamin(iField, voxel_size,0.7);
% filename='G:\2019\DL\esp_test_data\esp_change_big_BW_change'
% %filename='G:\B_new_2\BU_new\Liver_Iron_Quantification\exams\human\reproducibility_healthy_subjects\Gaiying\ge_mr1\scan1'
% [iField,voxel_size,matrix_size,CF,delta_TE,TE,B0_dir]=Read_DICOM(filename);
% iField = permute(iField,[2 1 3 4]);
% Mask2= genMaskRamin(iField, voxel_size,0.7);
% filename='G:\B_new_2\BU_new\Liver_Iron_Quantification\exams\human\sheth_patient\p11_iron_06292018\gre'
% %filename='G:\B_new_2\BU_new\Liver_Iron_Quantification\exams\human\reproducibility_healthy_subjects\Gaiying\ge_mr1\scan1'
% [iField,voxel_size,matrix_size,CF,delta_TE,TE,B0_dir]=Read_DICOM(filename);
% iField = permute(iField,[2 1 3 4]);
% Mask3= genMaskRamin(iField, voxel_size,0.7);
% Mask = cat(3,Mask1,Mask2,Mask3);
% save Masks.mat Mask
% %%
% clear
% 
% %load final_single_2000_normless_input_db
% %s1=load('final_single_l2_normless_input_2000_lrlow.mat');
% %s2=load('final_single_2000_normless_input.mat');
% 
% %  s1=load('final_single_l2_2000_normless_input_dut.mat');
% %  s2=load('final_single_2000_normless_input_dut.mat');
% 
%  %s1=load('final_single_l2_2000_normless_input_dut.mat');
%  %s2=load('paper_final_single_l2_norm_input_2000_lrlow.mat')
%  
%  %s1=load('paper_final_single_l2_normless_input_2000_lrlow');
%  %s2=load('paper_final_single_l2_norm_input_2000_lrlow');
% 
%  %s2=load('final_single_2000_normless_input_dut.mat');
%  
%   load Masks.mat
%   s=load('final_single_2000_normless_input_dut')
%   %saved-final_single_2000_normless_input-2000-24.42.hdf5.mat
%   
%   
%   s1=load('final_single_l2_normless_input_2000_lowlr-2000-355.15.hdf5.mat');
%   s2=load('saved-final_single_2000_normless_input-2000-24.42.hdf5.mat');
%   
%  
%   s3=load('final_single_l2_normless_input_2000_realimg-2000-479.06.hdf5.mat') 
%   s4 = load('saved-final_single_2000_realimg-2000-24.35.hdf5.mat')
% 
% s1.test_gt(:,:,:,1)= s1.test_gt(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s1.test_gt(:,:,:,2)= s1.test_gt(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s1.test_pd(:,:,:,1)= s1.test_pd(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s1.test_pd(:,:,:,2)= s1.test_pd(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s2.test_pd(:,:,:,1)= s2.test_pd(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s2.test_pd(:,:,:,2)= s2.test_pd(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s3.test_pd(:,:,:,1)= s3.test_pd(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s3.test_pd(:,:,:,2)= s3.test_pd(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s4.test_pd(:,:,:,1)= s4.test_pd(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s4.test_pd(:,:,:,2)= s4.test_pd(:,:,:,2)*s.f_std_r+s.f_mean_r;
% 
% s1.test_gt(:,:,:,5)= s1.test_gt(:,:,:,5)*s.frq_std+s.frq_mean;
% s1.test_gt(:,:,:,6)= s1.test_gt(:,:,:,6)*s.r2_std+s.r2_mean;
% s1.test_pd(:,:,:,5)= s1.test_pd(:,:,:,5)*s.frq_std+s.frq_mean;
% s1.test_pd(:,:,:,6)= s1.test_pd(:,:,:,6)*s.r2_std+s.r2_mean;
% s2.test_pd(:,:,:,5)= s2.test_pd(:,:,:,5)*s.frq_std+s.frq_mean;
% s2.test_pd(:,:,:,6)= s2.test_pd(:,:,:,6)*s.r2_std+s.r2_mean;
% s3.test_pd(:,:,:,5)= s3.test_pd(:,:,:,5)*s.frq_std+s.frq_mean;
% s3.test_pd(:,:,:,6)= s3.test_pd(:,:,:,6)*s.r2_std+s.r2_mean;
% s4.test_pd(:,:,:,5)= s4.test_pd(:,:,:,5)*s.frq_std+s.frq_mean;
% s4.test_pd(:,:,:,6)= s4.test_pd(:,:,:,6)*s.r2_std+s.r2_mean;
% 
% st=50;
% ed=220;
% 
% vis([permute(s1.test_gt(:,st:ed,:,:),[2 3 1 4]).*Mask(st:ed,:,:)....
%     permute(s1.test_pd(:,st:ed,:,:),[2 3 1 4]).*Mask(st:ed,:,:)...
%     permute(s2.test_pd(:,st:ed,:,:),[2 3 1 4]).*Mask(st:ed,:,:)])
% 
% 
% vis([permute(s1.test_gt(:,st:ed,:,:),[2 3 1 4]).*Mask(st:ed,:,:)....
%     permute(s1.test_pd(:,st:ed,:,:),[2 3 1 4]).*Mask(st:ed,:,:)...
%     permute(s2.test_pd(:,st:ed,:,:),[2 3 1 4]).*Mask(st:ed,:,:)...
%     permute(s3.test_pd(:,st:ed,:,:),[2 3 1 4]).*Mask(st:ed,:,:)...
%     permute(s4.test_pd(:,st:ed,:,:),[2 3 1 4]).*Mask(st:ed,:,:)])
% 
% 
% vis([iField(st:ed,:,:,:).*Mask1(st:ed,:,:) iField(st:ed,:,:,:).*Mask1(st:ed,:,:)...
%     iField(st:ed,:,:,:).*Mask1(st:ed,:,:) iField(st:ed,:,:,:).*Mask1(st:ed,:,:) iField(st:ed,:,:,:).*Mask1(st:ed,:,:)])
% %%
% st=50;
% ed=220;
% 
% vis([permute(s1.test_gt(:,st:ed,:,:),[2 3 1 4])...
%     permute(s1.test_pd(:,st:ed,:,:),[2 3 1 4])...
%     permute(s2.test_pd(:,st:ed,:,:),[2 3 1 4])])
% %%
% 
%  vis(cat(4,permute(s1.test_gt(:,:,:,1),[2 3 1 4]),permute(s1.test_pd(:,:,:,1),[2 3 1 4]),permute(s2.test_pd(:,:,:,1),[2 3 1 4]),...
%      permute(s1.test_gt(:,:,:,2),[2 3 1 4]),permute(s1.test_pd(:,:,:,2),[2 3 1 4]),permute(s2.test_pd(:,:,:,2),[2 3 1 4]),...
%      permute(s1.test_gt(:,:,:,5),[2 3 1 4]),permute(s1.test_pd(:,:,:,5),[2 3 1 4]),permute(s2.test_pd(:,:,:,5),[2 3 1 4]),...
%      permute(s1.test_gt(:,:,:,6),[2 3 1 4]),permute(s1.test_pd(:,:,:,6),[2 3 1 4]),permute(s2.test_pd(:,:,:,6),[2 3 1 4])))
% 
% %%
%  vis(cat(4,permute(s1.test_gt(:,:,:,:),[2 3 1 4])))
% %%
% clear
% close all
% 
% %   s1=load('paper_final_single_l2_normless_input_2000_lrlow');
% %   s2=load('final_single_2000_normless_input_dut.mat');
%  s=load('final_single_2000_normless_input_dut')
%  s1=load('final_single_l2_normless_input_2000_lowlr-2000-355.15.hdf5.mat');
%  s2=load('saved-final_single_2000_normless_input-2000-24.42.hdf5.mat');
%  s3=load('onecasehighiron_final_single_2000_normless_input_001.mat');
% 
% s1.test_gt(:,:,:,1)= s1.test_gt(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s1.test_gt(:,:,:,2)= s1.test_gt(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s1.test_pd(:,:,:,1)= s1.test_pd(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s1.test_pd(:,:,:,2)= s1.test_pd(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s2.test_pd(:,:,:,1)= s2.test_pd(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s2.test_pd(:,:,:,2)= s2.test_pd(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s3.test_pd(:,:,:,1)= s3.test_pd(:,:,:,1)*s3.w_std_r+s3.w_mean_r;
% s3.test_pd(:,:,:,2)= s3.test_pd(:,:,:,2)*s3.f_std_r+s3.f_mean_r;
% 
% s1.test_gt(:,:,:,5)= s1.test_gt(:,:,:,5)*s.frq_std+s.frq_mean;
% s1.test_gt(:,:,:,6)= s1.test_gt(:,:,:,6)*s.r2_std+s.r2_mean;
% s1.test_pd(:,:,:,5)= s1.test_pd(:,:,:,5)*s.frq_std+s.frq_mean;
% s1.test_pd(:,:,:,6)= s1.test_pd(:,:,:,6)*s.r2_std+s.r2_mean;
% s2.test_pd(:,:,:,5)= s2.test_pd(:,:,:,5)*s.frq_std+s.frq_mean;
% s2.test_pd(:,:,:,6)= s2.test_pd(:,:,:,6)*s.r2_std+s.r2_mean;
% s3.test_pd(:,:,:,5)= s3.test_pd(:,:,:,5)*s3.frq_std+s3.frq_mean;
% s3.test_pd(:,:,:,6)= s3.test_pd(:,:,:,6)*s3.r2_std+s3.r2_mean;
% 
% 
% %high iron test=train plotting
% s3p.test_pd(1:70,:,:,1)= 0*s2.test_pd(1:70,:,:,1);
% s3p.test_pd(1:70,:,:,2)= 0*s2.test_pd(1:70,:,:,2);
% s3p.test_pd(1:70,:,:,5)= 0*s2.test_pd(1:70,:,:,5);
% s3p.test_pd(1:70,:,:,6)= 0*s2.test_pd(1:70,:,:,6);
% 
% s3p.test_pd(71:71+28,:,:,1)= s3.test_pd(:,:,:,1);
% s3p.test_pd(71:71+28,:,:,2)= s3.test_pd(:,:,:,2);
% s3p.test_pd(71:71+28,:,:,5)= s3.test_pd(:,:,:,5);
% s3p.test_pd(71:71+28,:,:,6)= s3.test_pd(:,:,:,6);
% 
% s3p.test_pd(100,:,:,1)= 0*s2.test_pd(100,:,:,1);
% s3p.test_pd(100,:,:,2)= 0*s2.test_pd(100,:,:,2);
% s3p.test_pd(100,:,:,5)= 0*s2.test_pd(100,:,:,5);
% s3p.test_pd(100,:,:,6)= 0*s2.test_pd(100,:,:,6);
% 
% 
% %load roi_data_17.mat
% %load roi_data_all.mat
% %load roi_data_healthy.mat
% %load roi_data_patient.mat
% %load roi_data_patient2.mat
% 
% %load roi_data_h1_dut.mat
% load roi_data_p1_dut.mat
% %load roi_data_h1_espcase.mat
% 
% gt=permute(s1.test_gt(:,:,:,:),[2 3 1 4]);
% l1=permute(s1.test_pd(:,:,:,:),[2 3 1 4]);
% l2=permute(s2.test_pd(:,:,:,:),[2 3 1 4]);
% l3=permute(s3p.test_pd(:,:,:,:),[2 3 1 4]);
% 
% pdff_gt = 100*(abs(gt(:,:,:,2))./(abs(gt(:,:,:,2))+abs(gt(:,:,:,1))));
% pdff_l1 = 100*(abs(l1(:,:,:,2))./(abs(l1(:,:,:,2))+abs(l1(:,:,:,1))));
% pdff_l2 = 100*(abs(l2(:,:,:,2))./(abs(l2(:,:,:,2))+abs(l2(:,:,:,1))));
% pdff_l3 = 100*(abs(l3(:,:,:,2))./(abs(l3(:,:,:,2))+abs(l3(:,:,:,1))));
% 
% r2s_gt = abs(gt(:,:,:,6));
% r2s_l1 = abs(l1(:,:,:,6));
% r2s_l2 = abs(l2(:,:,:,6));
% r2s_l3 = abs(l3(:,:,:,6));
% 
% 
% fld_gt = (gt(:,:,:,5));
% fld_l1 = (l1(:,:,:,5));
% fld_l2 = (l2(:,:,:,5));
% fld_l3 = (l3(:,:,:,5));
% 
% 
% for i=1:size(roilist,2)
%      Mask = roilist(i).mask; 
%      Mask_3d = logical(zeros(size(pdff_gt)));
%      roi_slice = roilist(i).slice;
%      Mask_3d(:,:,roi_slice)=(Mask);
%      
%      pdff_gt_roi(i) = mean(pdff_gt(Mask_3d));
%      pdff_l1_roi(i) = mean(pdff_l1(Mask_3d));
%      pdff_l2_roi(i) = mean(pdff_l2(Mask_3d));
%      pdff_l3_roi(i) = mean(pdff_l3(Mask_3d));
%      
%      r2s_gt_roi(i) = mean(r2s_gt(Mask_3d));
%      r2s_l1_roi(i) = mean(r2s_l1(Mask_3d));
%      r2s_l2_roi(i) = mean(r2s_l2(Mask_3d));
%      r2s_l3_roi(i) = mean(r2s_l3(Mask_3d));
%      
%      fld_gt_roi(i) = mean(fld_gt(Mask_3d));
%      fld_l1_roi(i) = mean(fld_l1(Mask_3d));
%      fld_l2_roi(i) = mean(fld_l2(Mask_3d));
%      fld_l3_roi(i) = mean(fld_l3(Mask_3d));
% 
% end
% 
% close all
% figure;
% subplot(131)
% mdl=fitlm([pdff_gt_roi]',[pdff_l1_roi]','linear');
% p1=plot(mdl,'Marker','o','MarkerSize',5,'color','red')
% p1(end-1,1).Visible='off'
% p1(end,1).Visible='off'
% p1(2).Color=[1 0 0];
% legend('off')
% txt1 = ['C1    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% hold on 
% mdl=fitlm([pdff_gt_roi]',[pdff_l2_roi]','linear');
% p2=plot(mdl,'Marker','o','MarkerSize',5,'color','blue')
% p2(end-1,1).Visible='off'
% p2(end,1).Visible='off'
% p2(2).Color=[0 0 1];
% legend('off')
% txt2 = ['C2    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% %legend(txt1,txt2,txt3)
% 
% hold on 
% mdl=fitlm([pdff_gt_roi]',[pdff_l3_roi]','linear');
% p3=plot(mdl,'Marker','o','MarkerSize',5,'color','green')
% p3(end-1,1).Visible='off'
% p3(end,1).Visible='off'
% p3(2).Color=[0 1 0];
% legend('off')
% txt3 = ['WF-DNN,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% %legend(txt1,txt2,txt3)
% 
% 
% legend([p1(1) p2(1) p3(1)],{txt1,txt2,txt3},'Location','northwest')
% 
% 
% xlabel('Reference-PDFF(%)')
% ylabel('PDFF(%)')
% title('')
%  %ylim([0 90])
%  %xlim([0 90])
%  
%  ylim([0 100])
%  xlim([0 100])
% pbaspect([1 1 1])
% set(gca,'FontSize',16)
% 
% 
% subplot(133)
% mdl=fitlm([r2s_gt_roi]',[r2s_l1_roi]','linear');
% p1=plot(mdl,'Marker','o','MarkerSize',5,'color','red')
% p1(end-1,1).Visible='off'
% p1(end,1).Visible='off'
% p1(2).Color=[1 0 0];
% legend('off')
% txt1 = ['C1    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% hold on  
% mdl=fitlm([r2s_gt_roi]',[r2s_l2_roi]','linear');
% p2=plot(mdl,'Marker','o','MarkerSize',5,'color','blue');
% p2(end-1,1).Visible='off'
% p2(end,1).Visible='off'
% p2(2).Color=[0 0 1];
% legend('off')
% txt2 = ['C2    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% 
% hold on  
% mdl=fitlm([r2s_gt_roi]',[r2s_l3_roi]','linear');
% p3=plot(mdl,'Marker','o','MarkerSize',5,'color','green');
% p3(end-1,1).Visible='off'
% p3(end,1).Visible='off'
% p3(2).Color=[0 1 0];
% legend('off')
% txt3 = ['WFF-DNN   ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% legend([p1(1) p2(1) p3(1)],{txt1,txt2,txt3},'Location','northwest')
% xlabel('Reference-R2*(Hz)')
% ylabel('R2*(Hz)')
% title('')
% pbaspect([1 1 1])
% set(gca,'FontSize',16)
% % xlim([10 100])
% % ylim([10 100])
% 
% xlim([0 150])
% ylim([0 150])
% 
% 
% 
% 
% 
% subplot(132)
% mdl=fitlm([fld_gt_roi]',[fld_l1_roi]','linear');
% p1=plot(mdl,'Marker','o','MarkerSize',5,'color','red')
% p1(end-1,1).Visible='off'
% p1(end,1).Visible='off'
% p1(2).Color=[1 0 0];
% legend('off')
% txt1 = ['C1    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% hold on  
% mdl=fitlm([fld_gt_roi]',[fld_l2_roi]','linear');
% p2=plot(mdl,'Marker','o','MarkerSize',5,'color','blue')
% p2(end-1,1).Visible='off'
% p2(end,1).Visible='off'
% p2(2).Color=[0 0 1];
% legend('off')
% txt2 = ['C2    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% 
% hold on  
% mdl=fitlm([fld_gt_roi]',[fld_l3_roi]','linear');
% p3=plot(mdl,'Marker','o','MarkerSize',5,'color','green')
% p3(end-1,1).Visible='off'
% p3(end,1).Visible='off'
% p3(2).Color=[0 1 0];
% legend('off')
% txt3 = ['WFF-DNN   ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% 
% 
% legend([p1(1) p2(1) p3(1)],{txt1,txt2,txt3},'Location','northwest')
% xlabel('Reference-Field(Hz)')
% ylabel('Field(Hz)')
% title('')
% pbaspect([1 1 1])
% % xlim([-60 80])
% % ylim([-60 80])
% xlim([10 90])
% ylim([10 90])
% set(gca,'FontSize',16)
% 
% st=50;
% ed=220;
% %%
% %%
% clear
% close all
% 
% %   s1=load('paper_final_single_l2_normless_input_2000_lrlow');
% %   s2=load('final_single_2000_normless_input_dut.mat');
%  s=load('final_single_2000_normless_input_dut')
%  s1=load('final_single_l2_normless_input_2000_lowlr-2000-355.15.hdf5.mat');
%  s2=load('saved-final_single_2000_normless_input-2000-24.42.hdf5.mat');
%  s3=load('onecase_final_single_2000_normless_input_075.mat');
% 
% s1.test_gt(:,:,:,1)= s1.test_gt(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s1.test_gt(:,:,:,2)= s1.test_gt(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s1.test_pd(:,:,:,1)= s1.test_pd(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s1.test_pd(:,:,:,2)= s1.test_pd(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s2.test_pd(:,:,:,1)= s2.test_pd(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s2.test_pd(:,:,:,2)= s2.test_pd(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s3.test_pd(:,:,:,1)= s3.test_pd(:,:,:,1)*s3.w_std_r+s3.w_mean_r;
% s3.test_pd(:,:,:,2)= s3.test_pd(:,:,:,2)*s3.f_std_r+s3.f_mean_r;
% 
% s1.test_gt(:,:,:,5)= s1.test_gt(:,:,:,5)*s.frq_std+s.frq_mean;
% s1.test_gt(:,:,:,6)= s1.test_gt(:,:,:,6)*s.r2_std+s.r2_mean;
% s1.test_pd(:,:,:,5)= s1.test_pd(:,:,:,5)*s.frq_std+s.frq_mean;
% s1.test_pd(:,:,:,6)= s1.test_pd(:,:,:,6)*s.r2_std+s.r2_mean;
% s2.test_pd(:,:,:,5)= s2.test_pd(:,:,:,5)*s.frq_std+s.frq_mean;
% s2.test_pd(:,:,:,6)= s2.test_pd(:,:,:,6)*s.r2_std+s.r2_mean;
% s3.test_pd(:,:,:,5)= s3.test_pd(:,:,:,5)*s3.frq_std+s3.frq_mean;
% s3.test_pd(:,:,:,6)= s3.test_pd(:,:,:,6)*s3.r2_std+s3.r2_mean;
% 
% 
% %normal volunteer test=train plotting
% % s3p.test_pd(1:70,:,:,1)= 0*s2.test_pd(1:70,:,:,1);
% % s3p.test_pd(1:70,:,:,2)= 0*s2.test_pd(1:70,:,:,2);
% % s3p.test_pd(1:70,:,:,5)= 0*s2.test_pd(1:70,:,:,5);
% % s3p.test_pd(1:70,:,:,6)= 0*s2.test_pd(1:70,:,:,6);
% 
% s3p.test_pd(1:32,:,:,1)= s3.test_pd(:,:,:,1);
% s3p.test_pd(1:32,:,:,2)= s3.test_pd(:,:,:,2);
% s3p.test_pd(1:32,:,:,5)= s3.test_pd(:,:,:,5);
% s3p.test_pd(1:32,:,:,6)= s3.test_pd(:,:,:,6);
% 
% s3p.test_pd(33:100,:,:,1)= 0*s2.test_pd(33:100,:,:,1);
% s3p.test_pd(33:100,:,:,2)= 0*s2.test_pd(33:100,:,:,2);
% s3p.test_pd(33:100,:,:,5)= 0*s2.test_pd(33:100,:,:,5);
% s3p.test_pd(33:100,:,:,6)= 0*s2.test_pd(33:100,:,:,6);
% 
% 
% %load roi_data_17.mat
% %load roi_data_all.mat
% %load roi_data_healthy.mat
% %load roi_data_patient.mat
% %load roi_data_patient2.mat
% 
% load roi_data_h1_dut.mat
% %load roi_data_p1_dut.mat
% %load roi_data_h1_espcase.mat
% 
% gt=permute(s1.test_gt(:,:,:,:),[2 3 1 4]);
% l1=permute(s1.test_pd(:,:,:,:),[2 3 1 4]);
% l2=permute(s2.test_pd(:,:,:,:),[2 3 1 4]);
% l3=permute(s3p.test_pd(:,:,:,:),[2 3 1 4]);
% 
% pdff_gt = 100*(abs(gt(:,:,:,2))./(abs(gt(:,:,:,2))+abs(gt(:,:,:,1))));
% pdff_l1 = 100*(abs(l1(:,:,:,2))./(abs(l1(:,:,:,2))+abs(l1(:,:,:,1))));
% pdff_l2 = 100*(abs(l2(:,:,:,2))./(abs(l2(:,:,:,2))+abs(l2(:,:,:,1))));
% pdff_l3 = 100*(abs(l3(:,:,:,2))./(abs(l3(:,:,:,2))+abs(l3(:,:,:,1))));
% 
% r2s_gt = abs(gt(:,:,:,6));
% r2s_l1 = abs(l1(:,:,:,6));
% r2s_l2 = abs(l2(:,:,:,6));
% r2s_l3 = abs(l3(:,:,:,6));
% 
% 
% fld_gt = (gt(:,:,:,5));
% fld_l1 = (l1(:,:,:,5));
% fld_l2 = (l2(:,:,:,5));
% fld_l3 = (l3(:,:,:,5));
% 
% 
% for i=1:size(roilist,2)
%      Mask = roilist(i).mask; 
%      Mask_3d = logical(zeros(size(pdff_gt)));
%      roi_slice = roilist(i).slice;
%      Mask_3d(:,:,roi_slice)=(Mask);
%      
%      pdff_gt_roi(i) = mean(pdff_gt(Mask_3d));
%      pdff_l1_roi(i) = mean(pdff_l1(Mask_3d));
%      pdff_l2_roi(i) = mean(pdff_l2(Mask_3d));
%      pdff_l3_roi(i) = mean(pdff_l3(Mask_3d));
%      
%      r2s_gt_roi(i) = mean(r2s_gt(Mask_3d));
%      r2s_l1_roi(i) = mean(r2s_l1(Mask_3d));
%      r2s_l2_roi(i) = mean(r2s_l2(Mask_3d));
%      r2s_l3_roi(i) = mean(r2s_l3(Mask_3d));
%      
%      fld_gt_roi(i) = mean(fld_gt(Mask_3d));
%      fld_l1_roi(i) = mean(fld_l1(Mask_3d));
%      fld_l2_roi(i) = mean(fld_l2(Mask_3d));
%      fld_l3_roi(i) = mean(fld_l3(Mask_3d));
% 
% end
% 
% close all
% figure;
% subplot(131)
% mdl=fitlm([pdff_gt_roi]',[pdff_l1_roi]','linear');
% p1=plot(mdl,'Marker','o','MarkerSize',5,'color','red')
% p1(end-1,1).Visible='off'
% p1(end,1).Visible='off'
% p1(2).Color=[1 0 0];
% legend('off')
% txt1 = ['C1    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% hold on 
% mdl=fitlm([pdff_gt_roi]',[pdff_l2_roi]','linear');
% p2=plot(mdl,'Marker','o','MarkerSize',5,'color','blue')
% p2(end-1,1).Visible='off'
% p2(end,1).Visible='off'
% p2(2).Color=[0 0 1];
% legend('off')
% txt2 = ['C2    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% %legend(txt1,txt2,txt3)
% 
% hold on 
% mdl=fitlm([pdff_gt_roi]',[pdff_l3_roi]','linear');
% p3=plot(mdl,'Marker','o','MarkerSize',5,'color','green')
% p3(end-1,1).Visible='off'
% p3(end,1).Visible='off'
% p3(2).Color=[0 1 0];
% legend('off')
% txt3 = ['WF-DNN,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% %legend(txt1,txt2,txt3)
% 
% 
% legend([p1(1) p2(1) p3(1)],{txt1,txt2,txt3},'Location','northwest')
% 
% 
% xlabel('Reference-PDFF(%)')
% ylabel('PDFF(%)')
% title('')
%  %ylim([0 90])
%  %xlim([0 90])
%  
%  ylim([0 100])
%  xlim([0 100])
% pbaspect([1 1 1])
% set(gca,'FontSize',16)
% 
% 
% subplot(133)
% mdl=fitlm([r2s_gt_roi]',[r2s_l1_roi]','linear');
% p1=plot(mdl,'Marker','o','MarkerSize',5,'color','red')
% p1(end-1,1).Visible='off'
% p1(end,1).Visible='off'
% p1(2).Color=[1 0 0];
% legend('off')
% txt1 = ['C1    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% hold on  
% mdl=fitlm([r2s_gt_roi]',[r2s_l2_roi]','linear');
% p2=plot(mdl,'Marker','o','MarkerSize',5,'color','blue');
% p2(end-1,1).Visible='off'
% p2(end,1).Visible='off'
% p2(2).Color=[0 0 1];
% legend('off')
% txt2 = ['C2    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% 
% hold on  
% mdl=fitlm([r2s_gt_roi]',[r2s_l3_roi]','linear');
% p3=plot(mdl,'Marker','o','MarkerSize',5,'color','green');
% p3(end-1,1).Visible='off'
% p3(end,1).Visible='off'
% p3(2).Color=[0 1 0];
% legend('off')
% txt3 = ['WFF-DNN   ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% legend([p1(1) p2(1) p3(1)],{txt1,txt2,txt3},'Location','northwest')
% xlabel('Reference-R2*(Hz)')
% ylabel('R2*(Hz)')
% title('')
% pbaspect([1 1 1])
% set(gca,'FontSize',16)
% % xlim([10 100])
% % ylim([10 100])
% 
% xlim([0 140])
% ylim([0 140])
% 
% 
% 
% 
% 
% subplot(132)
% mdl=fitlm([fld_gt_roi]',[fld_l1_roi]','linear');
% p1=plot(mdl,'Marker','o','MarkerSize',5,'color','red')
% p1(end-1,1).Visible='off'
% p1(end,1).Visible='off'
% p1(2).Color=[1 0 0];
% legend('off')
% txt1 = ['C1    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% hold on  
% mdl=fitlm([fld_gt_roi]',[fld_l2_roi]','linear');
% p2=plot(mdl,'Marker','o','MarkerSize',5,'color','blue')
% p2(end-1,1).Visible='off'
% p2(end,1).Visible='off'
% p2(2).Color=[0 0 1];
% legend('off')
% txt2 = ['C2    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% 
% hold on  
% mdl=fitlm([fld_gt_roi]',[fld_l3_roi]','linear');
% p3=plot(mdl,'Marker','o','MarkerSize',5,'color','green')
% p3(end-1,1).Visible='off'
% p3(end,1).Visible='off'
% p3(2).Color=[0 1 0];
% legend('off')
% txt3 = ['WFF-DNN   ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% 
% 
% legend([p1(1) p2(1) p3(1)],{txt1,txt2,txt3},'Location','northwest')
% xlabel('Reference-Field(Hz)')
% ylabel('Field(Hz)')
% title('')
% pbaspect([1 1 1])
% % xlim([-60 80])
% % ylim([-60 80])
% xlim([-10 25])
% ylim([-10 25])
% set(gca,'FontSize',16)
% 
% st=50;
% ed=220;
% 
% %%
% 
% %%
% clear
% close all
% %real and imaginaery results
% 
% %   s1=load('paper_final_single_l2_normless_input_2000_lrlow');
% %   s2=load('final_single_2000_normless_input_dut.mat');
% 
%   %saved-final_single_2000_normless_input-2000-24.42.hdf5.mat
%   
%   
%   %s1=load('final_single_l2_normless_input_2000_lowlr-2000-355.15.hdf5.mat');
%   %s2=load('saved-final_single_2000_normless_input-2000-24.42.hdf5.mat');
%   %s3=load('final_single_l2_normless_input_2000_realimg-2000-479.06.hdf5.mat') 
%   %s4 = load('saved-final_single_2000_realimg-2000-24.35.hdf5.mat')
% 
% 
% 
%  s=load('final_single_2000_normless_input_dut')
%  s1=load('final_single_l2_normless_input_2000_realimg-2000-479.06.hdf5.mat');
%  s2=load('saved-final_single_2000_realimg-2000-24.35.hdf5.mat');
%  s3=load('onecasen_final_single_2000_normless_input_075_realimg.mat');
% 
% s1.test_gt(:,:,:,1)= s1.test_gt(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s1.test_gt(:,:,:,2)= s1.test_gt(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s1.test_pd(:,:,:,1)= s1.test_pd(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s1.test_pd(:,:,:,2)= s1.test_pd(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s2.test_pd(:,:,:,1)= s2.test_pd(:,:,:,1)*s.w_std_r+s.w_mean_r;
% s2.test_pd(:,:,:,2)= s2.test_pd(:,:,:,2)*s.f_std_r+s.f_mean_r;
% s3.test_pd(:,:,:,1)= s3.test_pd(:,:,:,1)*s3.w_std_r+s3.w_mean_r;
% s3.test_pd(:,:,:,2)= s3.test_pd(:,:,:,2)*s3.f_std_r+s3.f_mean_r;
% 
% s1.test_gt(:,:,:,5)= s1.test_gt(:,:,:,5)*s.frq_std+s.frq_mean;
% s1.test_gt(:,:,:,6)= s1.test_gt(:,:,:,6)*s.r2_std+s.r2_mean;
% s1.test_pd(:,:,:,5)= s1.test_pd(:,:,:,5)*s.frq_std+s.frq_mean;
% s1.test_pd(:,:,:,6)= s1.test_pd(:,:,:,6)*s.r2_std+s.r2_mean;
% s2.test_pd(:,:,:,5)= s2.test_pd(:,:,:,5)*s.frq_std+s.frq_mean;
% s2.test_pd(:,:,:,6)= s2.test_pd(:,:,:,6)*s.r2_std+s.r2_mean;
% s3.test_pd(:,:,:,5)= s3.test_pd(:,:,:,5)*s3.frq_std+s3.frq_mean;
% s3.test_pd(:,:,:,6)= s3.test_pd(:,:,:,6)*s3.r2_std+s3.r2_mean;
% 
% 
% %normal volunteer test=train plotting
% % s3p.test_pd(1:70,:,:,1)= 0*s2.test_pd(1:70,:,:,1);
% % s3p.test_pd(1:70,:,:,2)= 0*s2.test_pd(1:70,:,:,2);
% % s3p.test_pd(1:70,:,:,5)= 0*s2.test_pd(1:70,:,:,5);
% % s3p.test_pd(1:70,:,:,6)= 0*s2.test_pd(1:70,:,:,6);
% 
% s3p.test_pd(1:32,:,:,1)= s3.test_pd(:,:,:,1);
% s3p.test_pd(1:32,:,:,2)= s3.test_pd(:,:,:,2);
% s3p.test_pd(1:32,:,:,5)= s3.test_pd(:,:,:,5);
% s3p.test_pd(1:32,:,:,6)= s3.test_pd(:,:,:,6);
% 
% s3p.test_pd(33:100,:,:,1)= 0*s2.test_pd(33:100,:,:,1);
% s3p.test_pd(33:100,:,:,2)= 0*s2.test_pd(33:100,:,:,2);
% s3p.test_pd(33:100,:,:,5)= 0*s2.test_pd(33:100,:,:,5);
% s3p.test_pd(33:100,:,:,6)= 0*s2.test_pd(33:100,:,:,6);
% 
% 
% %load roi_data_17.mat
% %load roi_data_all.mat
% %load roi_data_healthy.mat
% %load roi_data_patient.mat
% %load roi_data_patient2.mat
% 
% load roi_data_h1_dut.mat
% %load roi_data_p1_dut.mat
% %load roi_data_h1_espcase.mat
% 
% gt=permute(s1.test_gt(:,:,:,:),[2 3 1 4]);
% l1=permute(s1.test_pd(:,:,:,:),[2 3 1 4]);
% l2=permute(s2.test_pd(:,:,:,:),[2 3 1 4]);
% l3=permute(s3p.test_pd(:,:,:,:),[2 3 1 4]);
% 
% pdff_gt = 100*(abs(gt(:,:,:,2))./(abs(gt(:,:,:,2))+abs(gt(:,:,:,1))));
% pdff_l1 = 100*(abs(l1(:,:,:,2))./(abs(l1(:,:,:,2))+abs(l1(:,:,:,1))));
% pdff_l2 = 100*(abs(l2(:,:,:,2))./(abs(l2(:,:,:,2))+abs(l2(:,:,:,1))));
% pdff_l3 = 100*(abs(l3(:,:,:,2))./(abs(l3(:,:,:,2))+abs(l3(:,:,:,1))));
% 
% r2s_gt = abs(gt(:,:,:,6));
% r2s_l1 = abs(l1(:,:,:,6));
% r2s_l2 = abs(l2(:,:,:,6));
% r2s_l3 = abs(l3(:,:,:,6));
% 
% 
% fld_gt = (gt(:,:,:,5));
% fld_l1 = (l1(:,:,:,5));
% fld_l2 = (l2(:,:,:,5));
% fld_l3 = (l3(:,:,:,5));
% 
% 
% for i=1:size(roilist,2)
%      Mask = roilist(i).mask; 
%      Mask_3d = logical(zeros(size(pdff_gt)));
%      roi_slice = roilist(i).slice;
%      Mask_3d(:,:,roi_slice)=(Mask);
%      
%      pdff_gt_roi(i) = mean(pdff_gt(Mask_3d));
%      pdff_l1_roi(i) = mean(pdff_l1(Mask_3d));
%      pdff_l2_roi(i) = mean(pdff_l2(Mask_3d));
%      pdff_l3_roi(i) = mean(pdff_l3(Mask_3d));
%      
%      r2s_gt_roi(i) = mean(r2s_gt(Mask_3d));
%      r2s_l1_roi(i) = mean(r2s_l1(Mask_3d));
%      r2s_l2_roi(i) = mean(r2s_l2(Mask_3d));
%      r2s_l3_roi(i) = mean(r2s_l3(Mask_3d));
%      
%      fld_gt_roi(i) = mean(fld_gt(Mask_3d));
%      fld_l1_roi(i) = mean(fld_l1(Mask_3d));
%      fld_l2_roi(i) = mean(fld_l2(Mask_3d));
%      fld_l3_roi(i) = mean(fld_l3(Mask_3d));
% 
% end
% 
% close all
% figure;
% subplot(131)
% mdl=fitlm([pdff_gt_roi]',[pdff_l1_roi]','linear');
% p1=plot(mdl,'Marker','o','MarkerSize',5,'color','red')
% p1(end-1,1).Visible='off'
% p1(end,1).Visible='off'
% p1(2).Color=[1 0 0];
% legend('off')
% txt1 = ['STD    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% hold on 
% mdl=fitlm([pdff_gt_roi]',[pdff_l2_roi]','linear');
% p2=plot(mdl,'Marker','o','MarkerSize',5,'color','blue')
% p2(end-1,1).Visible='off'
% p2(end,1).Visible='off'
% p2(2).Color=[0 0 1];
% legend('off')
% txt2 = ['UTD    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% %legend(txt1,txt2,txt3)
% 
% hold on 
% mdl=fitlm([pdff_gt_roi]',[pdff_l3_roi]','linear');
% p3=plot(mdl,'Marker','o','MarkerSize',5,'color','green')
% p3(end-1,1).Visible='off'
% p3(end,1).Visible='off'
% p3(2).Color=[0 1 0];
% legend('off')
% txt3 = ['NTD,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% %legend(txt1,txt2,txt3)
% 
% 
% legend([p1(1) p2(1) p3(1)],{txt1,txt2,txt3},'Location','northwest')
% 
% 
% xlabel('Reference-PDFF(%)')
% ylabel('PDFF(%)')
% title('')
%  %ylim([0 90])
%  %xlim([0 90])
%  
%  ylim([0 100])
%  xlim([0 100])
% pbaspect([1 1 1])
% set(gca,'FontSize',16)
% 
% 
% subplot(133)
% mdl=fitlm([r2s_gt_roi]',[r2s_l1_roi]','linear');
% p1=plot(mdl,'Marker','o','MarkerSize',5,'color','red')
% p1(end-1,1).Visible='off'
% p1(end,1).Visible='off'
% p1(2).Color=[1 0 0];
% legend('off')
% txt1 = ['STD    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% hold on  
% mdl=fitlm([r2s_gt_roi]',[r2s_l2_roi]','linear');
% p2=plot(mdl,'Marker','o','MarkerSize',5,'color','blue');
% p2(end-1,1).Visible='off'
% p2(end,1).Visible='off'
% p2(2).Color=[0 0 1];
% legend('off')
% txt2 = ['UTD    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% 
% hold on  
% mdl=fitlm([r2s_gt_roi]',[r2s_l3_roi]','linear');
% p3=plot(mdl,'Marker','o','MarkerSize',5,'color','green');
% p3(end-1,1).Visible='off'
% p3(end,1).Visible='off'
% p3(2).Color=[0 1 0];
% legend('off')
% txt3 = ['NTD   ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% legend([p1(1) p2(1) p3(1)],{txt1,txt2,txt3},'Location','northwest')
% xlabel('Reference-R2*(Hz)')
% ylabel('R2*(Hz)')
% title('')
% pbaspect([1 1 1])
% set(gca,'FontSize',16)
% % xlim([10 100])
% % ylim([10 100])
% 
% xlim([0 140])
% ylim([0 140])
% 
% 
% 
% 
% 
% subplot(132)
% mdl=fitlm([fld_gt_roi]',[fld_l1_roi]','linear');
% p1=plot(mdl,'Marker','o','MarkerSize',5,'color','red')
% p1(end-1,1).Visible='off'
% p1(end,1).Visible='off'
% p1(2).Color=[1 0 0];
% legend('off')
% txt1 = ['STD    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% hold on  
% mdl=fitlm([fld_gt_roi]',[fld_l2_roi]','linear');
% p2=plot(mdl,'Marker','o','MarkerSize',5,'color','blue')
% p2(end-1,1).Visible='off'
% p2(end,1).Visible='off'
% p2(2).Color=[0 0 1];
% legend('off')
% txt2 = ['UTD    ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% 
% hold on  
% mdl=fitlm([fld_gt_roi]',[fld_l3_roi]','linear');
% p3=plot(mdl,'Marker','o','MarkerSize',5,'color','green')
% p3(end-1,1).Visible='off'
% p3(end,1).Visible='off'
% p3(2).Color=[0 1 0];
% legend('off')
% txt3 = ['NTD   ,R^2 = ' num2str(mdl.Rsquared.Ordinary) ',   ' 'y = ' num2str(mdl.Coefficients.Estimate(2)) 'x+' num2str(mdl.Coefficients.Estimate(1)) ''];
% 
% 
% 
% legend([p1(1) p2(1) p3(1)],{txt1,txt2,txt3},'Location','northwest')
% xlabel('Reference-Field(Hz)')
% ylabel('Field(Hz)')
% title('')
% pbaspect([1 1 1])
% % xlim([-60 80])
% % ylim([-60 80])
% xlim([-10 25])
% ylim([-10 25])
% set(gca,'FontSize',16)
% 
% st=50;
% ed=220;
% 
% %%
% load Masks.mat
% vis([permute(s1.test_gt(:,st:ed,:,:),[2 3 1 4]).*Mask(st:ed,:,:)....
%     permute(s1.test_pd(:,st:ed,:,:),[2 3 1 4]).*Mask(st:ed,:,:)...
%     permute(s2.test_pd(:,st:ed,:,:),[2 3 1 4]).*Mask(st:ed,:,:)...
%     permute(s3p.test_pd(:,st:ed,:,:),[2 3 1 4]).*Mask(st:ed,:,:)])
% 
% %%
% vis(permute(s3.test_pd(:,st:ed,:,:),[2 3 1 4]))
% 
% %%
%  vis(cat(4,permute(s1.test_gt(:,:,:,1),[2 3 1 4]),permute(s1.test_pd(:,:,:,1),[2 3 1 4]),permute(s2.test_pd(:,:,:,1),[2 3 1 4]),...
%      permute(s1.test_gt(:,:,:,2),[2 3 1 4]),permute(s1.test_pd(:,:,:,2),[2 3 1 4]),permute(s2.test_pd(:,:,:,2),[2 3 1 4]),...
%      permute(s1.test_gt(:,:,:,5),[2 3 1 4]),permute(s1.test_pd(:,:,:,5),[2 3 1 4]),permute(s2.test_pd(:,:,:,5),[2 3 1 4]),...
%      permute(s1.test_gt(:,:,:,6),[2 3 1 4]),permute(s1.test_pd(:,:,:,6),[2 3 1 4]),permute(s2.test_pd(:,:,:,6),[2 3 1 4])),'ROI',roilist)
% 
% 
% %%
% ref = single(s1.test_gt(:,:,20,2));
% tst1 = s1.test_pd(:,:,20,5);
% tst2 = s2.test_pd(:,:,20,5);
% 
% %%
% clear
% frq_mean=1.8652723753095015;
% frq_std=29.05647636650208;
% r2_mean=27.403429307867484;
% r2_std=57.37705642316145;
% 
% % 
% % 
% % s1=load('saved-magphase_saved-final_single_2000-400-33.43.hdf5.mat');
% % s2=load('saved-magphase_saved-final_single_2000-800-27.15.hdf5.mat');
% % s3=load('saved-magphase_saved-final_single_2000-1600-22.81.hdf5.mat');
% % s4=load('final_single_2000.mat');
% % 
% 
% s1=load('saved-magphase_final_single_l2_normless_input_2000_lowlr-400-369.67.hdf5.mat');
% s2=load('saved-magphase_final_single_l2_normless_input_2000_lowlr-800-356.34.hdf5.mat');
% s3=load('saved-magphase_final_single_l2_normless_input_2000_lowlr-1600-348.18.hdf5.mat');
% s4=load('final_single_l2_normless_input_2000_lrlow.mat');
% 
% %s3.test_pd = s3.test_pd
% 
% %s3=load('magphase_l1_norm300_magphs_normlessphs_midlayersigmoid.mat');
% 
% %s3=load('magphase_l1_norm300_magphs_normlessphs.mat');%workshop best
% %result
% 
% 
% 
% s1.test_gt(:,:,:,5)= s1.test_gt(:,:,:,5)*frq_std+frq_mean;
% s1.test_gt(:,:,:,6)= s1.test_gt(:,:,:,6)*r2_std+r2_mean;
% s1.test_pd(:,:,:,5)= s1.test_pd(:,:,:,5)*frq_std+frq_mean;
% s1.test_pd(:,:,:,6)= s1.test_pd(:,:,:,6)*r2_std+r2_mean;
% s2.test_pd(:,:,:,5)= s2.test_pd(:,:,:,5)*frq_std+frq_mean;
% s2.test_pd(:,:,:,6)= s2.test_pd(:,:,:,6)*r2_std+r2_mean;
% s3.test_pd(:,:,:,5)= s3.test_pd(:,:,:,5)*frq_std+frq_mean;
% s3.test_pd(:,:,:,6)= s3.test_pd(:,:,:,6)*r2_std+r2_mean;
% s4.test_pd(:,:,:,5)= s4.test_pd(:,:,:,5)*frq_std+frq_mean;
% s4.test_pd(:,:,:,6)= s4.test_pd(:,:,:,6)*r2_std+r2_mean;
% st=0;
% ed=0;
% sf=0
% vis([permute(s1.test_gt,[2 3 1 4])...
%     permute(s1.test_pd,[2 3 1 4])...
%     permute(s2.test_pd,[2 3 1 4])...
%     permute(s3.test_pd,[2 3 1 4])...
%     permute(s4.test_pd,[2 3 1 4])])
% 
% %%
% s1=load('final_single.mat');
% s2=load('final_single_l2_normless_input.mat');
% %BEST RESULTS
% % % % % % % clear
% % % % % % % frq_mean=1.8652723753095015;
% % % % % % % frq_std=29.05647636650208;
% % % % % % % r2_mean=27.403429307867484;
% % % % % % % r2_std=57.37705642316145;
% % % % % % % s1=load('magphase_l1_norm300_magphs_normphs_midlayersigma.mat');
% % % % % % % s2=load('magphase_l1_norm300_magphs_normphs_2network_acrlinonolin.mat');
% % % % % % % s3=load('magphase_l1_norm300_magphs_normlessphs_midlayersigmoid.mat');
% % % % % % % 
% % % % % % % 
% % % % % % % s1.test_gt(:,:,:,5)= s1.test_gt(:,:,:,5)*frq_std+frq_mean;
% % % % % % % s1.test_gt(:,:,:,6)= s1.test_gt(:,:,:,6)*r2_std+r2_mean;
% % % % % % % s1.test_pd(:,:,:,5)= s1.test_pd(:,:,:,5)*frq_std+frq_mean;
% % % % % % % s1.test_pd(:,:,:,6)= s1.test_pd(:,:,:,6)*r2_std+r2_mean;
% % % % % % % s2.test_pd(:,:,:,5)= s2.test_pd(:,:,:,5)*frq_std+frq_mean;
% % % % % % % s2.test_pd(:,:,:,6)= s2.test_pd(:,:,:,6)*r2_std+r2_mean;
% % % % % % % s3.test_pd(:,:,:,5)= s3.test_pd(:,:,:,5)*frq_std+frq_mean;
% % % % % % % s3.test_pd(:,:,:,6)= s3.test_pd(:,:,:,6)*r2_std+r2_mean;
% % % % % % % 
% % % % % % % st=0;
% % % % % % % ed=0;
% % % % % % % sf=0
% % % % % % % vis([permute(s1.test_gt,[2 3 1 4])...
% % % % % % %     permute(s1.test_pd,[2 3 1 4])...
% % % % % % %     permute(s2.test_pd,[2 3 1 4])...
% % % % % % %     permute(s3.test_pd,[2 3 1 4])])
% 
% 
% %%
% vis(cat(4,permute(test_gt_all(:,:,:,1),[2 3 1 4]).*Mask,permute(test_pd_all_mse(:,:,:,1),[2 3 1 4]).*Mask,permute(test_pd_all(:,:,:,1),[2 3 1 4]).*Mask,permute(test_pd_all_ml(:,:,:,1),[2 3 1 4]).*Mask...
%     ,permute(test_gt_all(:,:,:,2),[2 3 1 4]).*Mask,permute(test_pd_all_mse(:,:,:,2),[2 3 1 4]).*Mask,permute(test_pd_all(:,:,:,2),[2 3 1 4]).*Mask,permute(test_pd_all_ml(:,:,:,2),[2 3 1 4]).*Mask...
%     ,permute(test_gt_all(:,:,:,3),[2 3 1 4]).*Mask,permute(test_pd_all_mse(:,:,:,3),[2 3 1 4]).*Mask,permute(test_pd_all(:,:,:,3),[2 3 1 4]).*Mask,permute(test_pd_all_ml(:,:,:,3),[2 3 1 4]).*Mask...
%     ,permute(test_gt_all(:,:,:,4),[2 3 1 4]).*Mask,permute(test_pd_all_mse(:,:,:,4),[2 3 1 4]).*Mask,permute(test_pd_all(:,:,:,4),[2 3 1 4]).*Mask,permute(test_pd_all_ml(:,:,:,4),[2 3 1 4]).*Mask))
% 
