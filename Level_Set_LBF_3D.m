% Implementation of the Local binary fitted model (LBF) as described in
% Soomro, Shafiullah, Asad Munir, and Kwang Nam Choi, PloS one 13.1 (2018): e0191827.

function phi = Level_Set_LBF_3D(image,plt)

% I0 = load(['3D_Tex_Images/',image,'.mat']); I0 = I0.matrix;
% if contains(image, '2_spheres')
%     Igt = load('3D_Tex_Images/ground_truths/2_spheres_ground_truth.mat');
%     Igt = Igt.matrix;
% elseif contains(image, '9_1')
%     Igt = load('3D_Tex_Images/ground_truths/9_1_spheres_ground_truth.mat');
%     Igt = Igt.matrix;
% elseif contains(image, 'im_3d')
%     Igt = load('3D_Tex_Images/ground_truths/im_3d_ground_truth.mat');
%     Igt = Igt.matrix;
% end
% 
% %I0=padarray(I0,[20 20 20],0,'both');
% %Igt=padarray(Igt,[20 20 20],0,'both');
% 
% MaxIter=250;
% TolE=10^-6;
% eps1=eps;
% voxel_size=[1,1,1];

I0 = load(['LS_Worm/',image,'.mat']); I0 = I0.Original;

%I0=padarray(I0,[20 20 20],0,'both');
%Igt=padarray(Igt,[20 20 20],0,'both');

MaxIter=100;
TolE=10^-6;
eps1=eps;
voxel_size=[1,1,3];

beta=1;
epsilon=1;
sigma=10./voxel_size;

lambda1=1;
lambda2=1;
mu1=0.0635; % Area'
mu2=0.0123;    % Area
mu3=0.0; % Volume
mu4=0.0;
mu5=0.0;
mu6=0.0;
mu7=0.0;
mu8=0.0;

I0=double(I0); 
I0=I0./max(I0(:));

I=I0;

H=@(x) 1/2+atan(x./epsilon)./pi;
Delta=@(x) (epsilon/pi)./(x.^2+epsilon^2);

i = 0;
delE = inf;
edge_ind=true(size(I)); edge_ind(2:end-1,2:end-1,2:end-1)=false;
edge_value=-1;
%phi = I - (max(I(:)) + min(I(:))) / 2;
phi = I - 0.01;
phi(edge_ind)=edge_value;

H_phi=H(phi);
G_I=imgaussfilt3(I,sigma);
G_I2=imgaussfilt3(I.^2,sigma);
G_H_phi=imgaussfilt3(H_phi,sigma)+eps;
G_H_phi_I=imgaussfilt3(H_phi.*I,sigma);
G_H_phi_I2=imgaussfilt3(H_phi.*I.^2,sigma);

f1=G_H_phi_I./G_H_phi;
f2=(G_I-G_H_phi_I)./(1-G_H_phi);
int1=G_H_phi_I2-2.*f1.*G_H_phi_I+f1.^2.*G_H_phi;
int2=G_I2-G_H_phi_I2-2.*f2.*(G_I-G_H_phi_I)+f2.^2.*(1-G_H_phi);

[GHx,GHy,GHz]=gradient(H_phi,1/voxel_size(1),1/voxel_size(2),1/voxel_size(3));
[Gphix,Gphiy,Gphiz]=gradient(phi,1/voxel_size(1),1/voxel_size(2),1/voxel_size(3));

Int1=sum(int1(:));
Int2=sum(int2(:));
C1=sum(GHx(:).^2+GHy(:).^2+GHz(:).^2);
C2=sum((GHx(:).^2+GHy(:).^2+GHz(:).^2).^0.5);
C3=sum(H_phi(:));
C4=sum(Gphix(:).^2+Gphiy(:).^2+Gphiz(:).^2);
C5=sum((Gphix(:).^2+Gphiy(:).^2+Gphiz(:).^2).^0.5);
C6=(C4-2.*C5+numel(I))./2;
C7=sum(phi(:).^2);
C8=sum(abs(phi(:)));

Costs=nan(10,MaxIter+1);
Costs(:,1)=[Int1,Int2,C1,C2,C3,C4,C5,C6,C7,C8];
E=lambda1*Int1+lambda2*Int2+mu1*C1+mu2*C2+mu3*C3+mu4*C4+mu5*C5+mu6*C6+mu7*C7+mu8*C8;

if plt==1
    fig_num=randi(1000);
    figure(fig_num)
    subplot(1,4,1)
    plot(0,log(E),'k.'), hold on
    ylabel('E'), xlabel('# steps'), axis square
    
    subplot(1,4,2)
    imshow(max(I,[],3),[0 1]), hold on
    fv = isosurface(phi,0);
    patch(fv,'FaceColor','red','EdgeColor','none');
    view(3)
    daspect(1./voxel_size)
    axis tight
    camlight
    camlight(-80,-10)
    lighting gouraud, hold off

    subplot(1,4,3)
    imshow(max(phi,[],3)>0,[0 1]), hold on

%     subplot(1,4,4)
%     imshow(max(Igt,[],3)), hold on
end
    
while i<MaxIter && delE>TolE
    i=i+1;   
    
%     p_errors=sum(Igt>0,[1,2,3])/numel(I);
%     fp_errors=sum(phi>0 & Igt<=0,[1,2,3])/numel(I);
%     fn_errors=sum(phi<=0 & Igt>0,[1,2,3])/numel(I);
%     iou_errors=(p_errors-fn_errors)./(p_errors+fp_errors);

    %disp([i,iou_errors])
    
    int1p=I.^2-2.*I.*imgaussfilt3(f1,sigma)+imgaussfilt3(f1.^2,sigma);
    int2p=I.^2-2.*I.*imgaussfilt3(f2,sigma)+imgaussfilt3(f2.^2,sigma);
    
    [Gphix,Gphiy,Gphiz]=gradient(phi,1/voxel_size(1),1/voxel_size(2),1/voxel_size(3));
    G_G_phi=divergence(Gphix./(Gphix.^2+Gphiy.^2+Gphiz.^2+eps1).^0.5,Gphiy./(Gphix.^2+Gphiy.^2+Gphiz.^2+eps1).^0.5,Gphiz./(Gphix.^2+Gphiy.^2+Gphiz.^2+eps1).^0.5);
    
    del_phi=-Delta(phi).*(lambda1.*int1p-lambda2.*int2p-2.*mu1.*del2(H(phi))-mu2.*G_G_phi+mu3)...
        +2.*mu4.*del2(phi)+mu5.*G_G_phi+mu6.*(del2(phi)-G_G_phi)-2.*mu7.*phi-mu8.*phi./(abs(phi)+eps1);

    phi_temp=phi+beta.*del_phi;
    phi_temp(edge_ind)=edge_value;
    
    H_phi_temp=H(phi_temp);
    G_H_phi_temp=imgaussfilt3(H_phi_temp,sigma)+eps;
    G_H_phi_I_temp=imgaussfilt3(H_phi_temp.*I,sigma);
    G_H_phi_I2_temp=imgaussfilt3(H_phi_temp.*I.^2,sigma);
    
    f1=G_H_phi_I_temp./G_H_phi_temp;
    f2=(G_I-G_H_phi_I_temp)./(1-G_H_phi_temp);
    int1=G_H_phi_I2_temp-2.*f1.*G_H_phi_I_temp+f1.^2.*G_H_phi_temp;
    int2=G_I2-G_H_phi_I2_temp-2.*f2.*(G_I-G_H_phi_I_temp)+f2.^2.*(1-G_H_phi_temp);
    
    [GHx,GHy,GHz]=gradient(H_phi_temp,1/voxel_size(1),1/voxel_size(2),1/voxel_size(3));
    [Gphix,Gphiy,Gphiz]=gradient(phi_temp,1/voxel_size(1),1/voxel_size(2),1/voxel_size(3));

    Int1=sum(int1(:));
    Int2=sum(int2(:));
    C1=sum(GHx(:).^2+GHy(:).^2+GHz(:).^2);
    C2=sum((GHx(:).^2+GHy(:).^2+GHz(:).^2).^0.5);
    C3=sum(H_phi_temp(:));
    C4=sum(Gphix(:).^2+Gphiy(:).^2+Gphiz(:).^2);
    C5=sum((Gphix(:).^2+Gphiy(:).^2+Gphiz(:).^2).^0.5);
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
    
    if plt==1 && mod(i,5)==0
        figure(fig_num)
        subplot(1,4,1)
        title('iter v. log(Energy)')
        plot(i,log(E),'k.')
        
        subplot(1,4,2)
        imshow(max(I0,[],3),[0 0.5]), hold on
        fv = isosurface(phi,0);
        patch(fv,'FaceColor','red','EdgeColor','none');
        view(3)
        daspect(1./voxel_size)
        axis tight
        camlight
        camlight(-80,-10)
        lighting gouraud, hold off

        subplot(1,4,3)
        title('Phi max projection')
        imshow(max(phi,[],3)>0,[0 1]), hold on

        subplot(1,4,4)
        %imshow(max(Igt,[],3)), hold on
        

        %errors = (fp_errors.^2 + fn_errors.^2).^0.5;
%         title({['FP = ',num2str(fp_errors)];['FN = ',num2str(fn_errors)];['IOU =',num2str(iou_errors)]})
    end
end

fig_num=randi(1000);
figure(fig_num)
imshow(max(I0,[],3),[0 0.5]), hold on
%fv = isosurface(phi,0);
fv = isosurface(imgaussfilt3(phi,1), 0);
patch(fv,'FaceColor','green','EdgeColor','none');
view(3)
daspect(1./voxel_size)
axis tight
camlight
camlight(-80,-10)
lighting gouraud, hold off

% fig_num=randi(1000);
% figure(fig_num)
% subplot(1,3,1)
% imshow(max(I0,[],3),[0 0.5]), hold on
% fv = isosurface(phi,0);
% patch(fv,'FaceColor','red','EdgeColor','none');
% title('2D: max(I0) and 3D: \phi > 0')
% view(3)
% daspect(1./voxel_size)
% axis tight
% camlight
% camlight(-80,-10)
% lighting gouraud, hold off
% 
% subplot(1,3,2)
% imshow(max(phi,[],3)>0,[0 1]), hold on
% title('\phi > 0')
% subplot(1,3,3)
% imshow(max(Igt,[],3)), hold on
% p_errors=sum(Igt>0,[1,2,3])/numel(I);
% fp_errors=sum(phi>0 & Igt<=0,[1,2,3])/numel(I);
% fn_errors=sum(phi<=0 & Igt>0,[1,2,3])/numel(I);
% iou_errors=(p_errors-fn_errors)./(p_errors+fp_errors);
% 
% %errors = (fp_errors.^2 + fn_errors.^2).^0.5;
% title({['FP = ',num2str(fp_errors)];['FN = ',num2str(fn_errors)];['IOU =',num2str(iou_errors)]})