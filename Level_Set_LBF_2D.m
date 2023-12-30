% Implementation of the Local binary fitted model (LBF) as described in
% Soomro, Shafiullah, Asad Munir, and Kwang Nam Choi, PloS one 13.1 (2018): e0191827.
% This is the original LBF version 

% plt=1;
% I1 = imread('Test images\im1.png'); I1 = rgb2gray(I1);
% I2 = imread('Test images\im2.png'); I2 = rgb2gray(I2);
% I3 = imread('Test images\im3.png'); I3 = rgb2gray(I3);
% I4 = imread('Test images\im4.png'); I4 = rgb2gray(I4);
% I5 = imread('Test images\gourd.bmp');
% I6 = imread('Test images\twocells.bmp'); I6 = rgb2gray(I6);
% I7 = load('Test images\ASJ-1.mat'); I7 = I7.Original_cut; I7=I7(:,:,15); %15,35
% I8 = load('Test images\ASJ-2.mat'); I8 = I8.Original_cut; I8=max(I8,[],3);
% I9 = load('Test images\AIY-1.mat'); I9 = I9.Original_cut; I9=max(I9,[],3);
% I10 = imread('Test images\2_intensities.png'); I10 = rgb2gray(I10);
% I11 = imread('Test images/9_1_intensity.png'); I11 = rgb2gray(I11);
% I12 = load('Test images\circle11.mat'); I12 = I12.H;

%mac version of directory
% plt=1;
% I1 = imread('Test images/im1.png'); I1 = rgb2gray(I1);
% I2 = imread('Test images/im2.png'); I2 = rgb2gray(I2);
% I3 = imread('Test images/im3.png'); I3 = rgb2gray(I3);
% I4 = imread('Test images/im4.png'); I4 = rgb2gray(I4);
% I5 = imread('Test images/gourd.bmp');
% I6 = imread('Test images/twocells.bmp'); I6 = rgb2gray(I6);
% I7 = load('Test images/ASJ-1.mat'); I7 = I7.Original_cut; I7=I7(:,:,15); %15,35
% I8 = load('Test images/ASJ-2.mat'); I8 = I8.Original_cut; I8=max(I8,[],3);
% I9 = load('Test images/AIY-1.mat'); I9 = I9.Original_cut; I9=max(I9,[],3);
% I10 = imread('Test images/2_intensities.png'); I10 = rgb2gray(I10);
% I11 = imread('Test images/9_1_intensity.png'); I11 = rgb2gray(I11);
% I12 = load('Test images/circle11.mat'); I12 = I12.H;
% I13 = imread('Training_Images/im.png'); I13 = rgb2gray(I13);

% plt=1;
% I1 = imread('Worm_Training_Images\AIY2.png'); I1 = im2gray(I1);
% I2 = imread('Worm_Training_Images\AIY3.png'); I2 = im2gray(I2);
% I3 = imread('Worm_Training_Images\AIY4.png'); I3 = im2gray(I3);
% I4 = imread('Worm_Training_Images\AIY9.png'); I4 = im2gray(I4);
% I5 = imread('Worm_Training_Images\AIY10.png'); I5 = im2gray(I5);
% I6 = imread('Worm_Training_Images\AIY11.png'); I6 = im2gray(I6);
% I7 = imread('Worm_Training_Images\ASJ1.png'); I7 = im2gray(I7);
% I8 = imread('Worm_Training_Images\ASJ2.png'); I8 = im2gray(I8);
% I9 = imread('Worm_Training_Images\ASJ8.png'); I9 = im2gray(I9);
% I10 = imread('Worm_Training_Images\ASJ10.png'); I10 = im2gray(I10);
% I11 = imread('Worm_Training_Images\ASJ13.png'); I11 = im2gray(I11);
% I12 = imread('Worm_Training_Images\ASJ14.png'); I12 = im2gray(I12);

% plt=1;
% I1 = imread('Worm_Training_Images/AIY2.png'); I1 = im2gray(I1);
% I2 = imread('Worm_Training_Images/AIY3.png'); I2 = im2gray(I2);
% I3 = imread('Worm_Training_Images/AIY4.png'); I3 = im2gray(I3);
% I4 = imread('Worm_Training_Images/AIY9.png'); I4 = im2gray(I4);
% I5 = imread('Worm_Training_Images/AIY10.png'); I5 = im2gray(I5);
% I6 = imread('Worm_Training_Images/AIY11.png'); I6 = im2gray(I6);
% I7 = imread('Worm_Training_Images/ASJ1.png'); I7 = im2gray(I7);
% I8 = imread('Worm_Training_Images/ASJ2.png'); I8 = im2gray(I8);
% I9 = imread('Worm_Training_Images/ASJ8.png'); I9 = im2gray(I9);
% I10 = imread('Worm_Training_Images/ASJ10.png'); I10 = im2gray(I10);
% I11 = imread('Worm_Training_Images/ASJ13.png'); I11 = im2gray(I11);
% I12 = imread('Worm_Training_Images/ASJ14.png'); I12 = im2gray(I12);
% neuron='AIY2'

% [phi,E] = Level_Set_LBF_2D_v1('AIY2',plt);
% [phi,E] = Level_Set_LBF_2D_v1('AIY3',plt);
% [phi,E] = Level_Set_LBF_2D_v1('AIY4',plt);
% [phi,E] = Level_Set_LBF_2D_v1('AIY9',plt);
% [phi,E] = Level_Set_LBF_2D_v1('AIY10',plt);
% [phi,E] = Level_Set_LBF_2D_v1('AIY11',plt);
% [phi,E] = Level_Set_LBF_2D_v1('ASJ1',plt);
% [phi,E] = Level_Set_LBF_2D_v1('ASJ2',plt);
% [phi,E] = Level_Set_LBF_2D_v1('ASJ8',plt);
% [phi,E] = Level_Set_LBF_2D_v1('ASJ10',plt);
% [phi,E] = Level_Set_LBF_2D_v1('ASJ13',plt);
% [phi,E] = Level_Set_LBF_2D_v1('ASJ14',plt);


function [phi,E] = Level_Set_LBF_2D(neuron,plt)
% function [phi,E] = Level_Set_LBF_2D_v1(I,plt,params)
I = imread(['Worm_Training_Images/',neuron,'.png']); I = im2gray(I);
Igt = imread(['Worm_Training_Images/ground_truths/',neuron,'_gt.png']); Igt = im2gray(Igt);
Igt_axons = imread(['Worm_Training_Images/ground_truths_axons/',neuron,'_gt.png']); Igt_axons = im2gray(Igt_axons);


MaxIter=250;
TolE=10^-6;
eps1=eps;
beta=1;

epsilon=1;
sigma=20;
lambda1=1;
lambda2=1;
mu1=0.0;
mu2=0.0;
mu3=0.0078; %0.0078125;%0.0078;
mu4=0.0308; %0.030769;%0.0308;
mu5=0.0;
mu6=0.0;
mu7=0.0;
mu8=0.0;

% epsilon=params(1);
% sigma=params(2);
% lambda1=params(3);
% lambda2=params(4);
% mu1=params(5);
% mu2=params(6);
% mu3=params(7);
% mu4=params(8);
% mu5=params(9);
% mu6=params(10);
% mu7=params(11);
% mu8=params(12);

I=double(I); I=I./max(I(:));
H=@(x) 1/2+atan(x./epsilon)./pi;
Delta=@(x) (epsilon/pi)./(x.^2+epsilon^2);
% initialization
i=0; delE=inf;
edge_ind=true(size(I)); edge_ind(2:end-1,2:end-1)=false;
edge_value=-1;
phi = I - (max(I(:)) + min(I(:))) / 2;
%phi=-ones(size(I)); phi(50:end-50,50:end-50)=1;
%phi = I - 0.01;
phi(edge_ind)=edge_value;


f1=imgaussfilt(H(phi).*I,sigma,'padding','symmetric')./(imgaussfilt(H(phi),sigma,'padding','symmetric')+eps1)+eps1;
f2=imgaussfilt((1-H(phi)).*I,sigma,'padding','symmetric')./(imgaussfilt(1-H(phi),sigma,'padding','symmetric')+eps1)+eps1;
int1=imgaussfilt(I.^2.*H(phi),sigma,'padding','symmetric')-2.*f1.*imgaussfilt(I.*H(phi),sigma,'padding','symmetric')+f1.^2.*imgaussfilt(H(phi),sigma,'padding','symmetric');
int2=imgaussfilt(I.^2.*(1-H(phi)),sigma,'padding','symmetric')-2.*f2.*imgaussfilt(I.*(1-H(phi)),sigma,'padding','symmetric')+f2.^2.*imgaussfilt(1-H(phi),sigma,'padding','symmetric');
[GHx,GHy]=gradient(H(phi));
[Gphix,Gphiy]=gradient(phi);

Int1=sum(int1(:));
Int2=sum(int2(:));
C1=sum(GHx(:).^2+GHy(:).^2);
C2=sum((GHx(:).^2+GHy(:).^2).^0.5); % Length
C3=sum(H(phi(:))); % Area
C4=sum(Gphix(:).^2+Gphiy(:).^2);
C5=sum((Gphix(:).^2+Gphiy(:).^2).^0.5);
C6=(C4-2.*C5+numel(I))./2;
C7=sum(phi(:).^2);
C8=sum(abs(phi(:)));

Costs=nan(10,MaxIter+1);
Costs(:,1)=[Int1,Int2,C1,C2,C3,C4,C5,C6,C7,C8];
E=lambda1*Int1+lambda2*Int2+mu1*C1+mu2*C2+mu3*C3+mu4*C4+mu5*C5+mu6*C6+mu7*C7+mu8*C8;

if plt==1  
    figure(1)
    subplot(2,3,1)
    plot(0,E,'k.'), hold on
    ylabel('E'), xlabel('# steps'), axis square
    
    subplot(2,3,2)
    imshow(I,[0 2]), hold on
    imcontour(phi,[0,0],'r'); hold off

    subplot(2,3,3)
    imshow(phi>0), caxis([0 1])

    subplot(2,3,4)
    imshow(Igt), caxis([0 1])

    subplot(2,3,5)
    imshow(Igt_axons), caxis([0 1])
end

    
while i<MaxIter && delE>TolE
    i=i+1;   
    int1p=I.^2.*imgaussfilt(ones(size(I)),sigma,'padding','symmetric')-2.*I.*imgaussfilt(f1,sigma,'padding','symmetric')+imgaussfilt(f1.^2,sigma,'padding','symmetric');
    int2p=I.^2.*imgaussfilt(ones(size(I)),sigma,'padding','symmetric')-2.*I.*imgaussfilt(f2,sigma,'padding','symmetric')+imgaussfilt(f2.^2,sigma,'padding','symmetric');
    
    [Gphix,Gphiy]=gradient(phi);
    G_G_phi=divergence(Gphix./(Gphix.^2+Gphiy.^2+eps1).^0.5,Gphiy./(Gphix.^2+Gphiy.^2+eps1).^0.5);
    
    del_phi=-Delta(phi).*(lambda1.*int1p-lambda2.*int2p-2.*mu1.*del2(H(phi))-mu2.*G_G_phi+mu3)...
        +2.*mu4.*del2(phi)+mu5.*G_G_phi+mu6.*(del2(phi)-G_G_phi)-2.*mu7.*phi-mu8.*phi./(abs(phi)+eps1);
    
    phi_temp=phi+beta.*del_phi;
    phi_temp(edge_ind)=edge_value;
    
    f1=imgaussfilt(H(phi_temp).*I,sigma,'padding','symmetric')./(imgaussfilt(H(phi_temp),sigma,'padding','symmetric')+eps1)+eps1;
    f2=imgaussfilt((1-H(phi_temp)).*I,sigma,'padding','symmetric')./(imgaussfilt(1-H(phi_temp),sigma,'padding','symmetric')+eps1)+eps1;
    int1=imgaussfilt(I.^2.*H(phi_temp),sigma,'padding','symmetric')-2.*f1.*imgaussfilt(I.*H(phi_temp),sigma,'padding','symmetric')+f1.^2.*imgaussfilt(H(phi_temp),sigma,'padding','symmetric');
    int2=imgaussfilt(I.^2.*(1-H(phi_temp)),sigma,'padding','symmetric')-2.*f2.*imgaussfilt(I.*(1-H(phi_temp)),sigma,'padding','symmetric')+f2.^2.*imgaussfilt(1-H(phi_temp),sigma,'padding','symmetric');
    [GHx,GHy]=gradient(H(phi_temp));
    [Gphix,Gphiy]=gradient(phi_temp);

    Int1=sum(int1(:));
    Int2=sum(int2(:));
    C1=sum(GHx(:).^2+GHy(:).^2);
    C2=sum((GHx(:).^2+GHy(:).^2).^0.5); % Length
    C3=sum(H(phi_temp(:))); % Area
    C4=sum(Gphix(:).^2+Gphiy(:).^2);
    C5=sum((Gphix(:).^2+Gphiy(:).^2).^0.5);
    C6=(C4-2.*C5+numel(I))./2;
    C7=sum(phi_temp(:).^2);
    C8=sum(abs(phi_temp(:)));
    
    E_temp=lambda1*Int1+lambda2*Int2+mu1*C1+mu2*C2+mu3*C3+mu4*C4+mu5*C5+mu6*C6+mu7*C7+mu8*C8;
    
    if E_temp<=E
        phi=phi_temp;
        delE=E-E_temp;
        E=E_temp;
        beta=beta*1.5;
        Costs(:,i+1)=[Int1,Int2,C1,C2,C3,C4,C5,C6,C7,C8];
    else
        beta=beta/1.5;
    end
    
    if plt==1 && mod(i,1)==0
        figure(1)
        subplot(2,3,1)
        plot(i,E,'k.')

        subplot(2,3,2)
        imshow(I,[0 2]), hold on
        imcontour(phi,[0,0],'r'); hold off
               
        subplot(2,3,3)
        imshow(phi>0), caxis([0 1])

        subplot(2,3,4)
        p_errors=sum(Igt>0,[1,2])/numel(I);
        fp_errors=sum(phi>0 & Igt<=0,[1,2])/numel(I);
        fn_errors=sum(phi<=0 & Igt>0,[1,2])/numel(I);
        iou_errors=(p_errors-fn_errors)./(p_errors+fp_errors);

        %errors = (fp_errors.^2 + fn_errors.^2).^0.5;
        title({['FP = ',num2str(fp_errors)];['FN = ',num2str(fn_errors)];['IOU =',num2str(iou_errors)]})

        subplot(2,3,5)
        p_errors_axons=sum(Igt_axons>0,[1,2])/numel(I);
        fp_errors_axons=sum(phi>0 & Igt_axons<=0,[1,2])/numel(I);
        fn_errors_axons=sum(phi<=0 & Igt_axons>0,[1,2])/numel(I);
        iou_errors_axons=(p_errors_axons-fn_errors_axons)./(p_errors_axons+fp_errors_axons);
        %errors_axons = (fp_errors_axons.^2 + fn_errors_axons.^2).^0.5;
        title({['FP = ',num2str(fp_errors_axons)];['FN = ',num2str(fn_errors_axons)];['IOU =',num2str(iou_errors_axons)]})
    end
end
1
% if plt==1
%     figure
%     plot(1:MaxIter+1,Costs,'-')
%     ylabel('Costs'), xlabel('# steps'), axis square
%     legend({'Int1','Int2','C1','C2','C3','C4','C5','C6','C7','C8'})
% end
