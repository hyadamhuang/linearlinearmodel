function [M1]=findVesselLumen(I,M,num)
% manuscript: Deep Learning for quantification of basilar artery morphology
% using intracranial vessel wall MRI: a feasibility study
% by Chien-Hung Tsou, Hon-Man Liu, Adam Huang
% input:
%   I: cross sectional vessel image
%   M: vessel outer wall boundary mask
%   num: number of probing directions
% ouput:
%   M1: vessel lumen (inner wall) mask
% Author: Adam Huang, 2024/11/28
% adamhuan@gmail.com

% num=12;% 12 directions
[s1,s2]=size(I);
[row,col,ix]=polarCoord_discrete(num,s1,s2);
px=zeros(1,num);
py=zeros(1,num);
% fix disconnection problems for imperfect outer wall markings
B=(M>0);
se=strel('disk',2);
B=imdilate(B,se);
B=imfill(B,'holes');
B=imerode(B,se);
M(B)=1;
for j=1:num
    y=I(ix(:,j));
    z=M(ix(:,j));
    ii=sum(z);

    if ii>=5
        [BAT,MD,TTP]=BAT_LLM_ridge(1:ii,double(y(1:ii)));
        i=MD;
    else
        [mx,imx]=max(y(1:ii));
        [mn,imn]=min(y(1:imx));
        mid=(mx+mn)/2;
        for i=imn:imx
            if y(i)>=mid
                break;
            end
        end
    end
    if i>0
        py(j)=row(i,j);
        px(j)=col(i,j);
    end
end
disp(length(px))
M1=uint8(poly2mask(px,py,s1,s2));

function [row,col,ix] = polarCoord_discrete(n,s1,s2)
% create (discrete) polar coordinates indexing
% for fast radial sampling
% in n angular directions

m=floor(s1/2); % radius, assuming s1==s2
i0=m; % center of image
j0=m; % [s1/2,s1/2]
row=zeros(m,n); 
col=zeros(m,n);
dtheta=2*pi/n;
for j=1:n
    c=cos((j-1)*dtheta);
    s=sin((j-1)*dtheta);
    for i=1:m
        row(i,j)=i0+c*(i-1);
        col(i,j)=j0+s*(i-1);
    end
end
row=round(row); % discretized
col=round(col); % discretized
ix=sub2ind([s1,s2],row,col);

function [BAT,MD,TTP]=BAT_LLM_ridge(t,y)

% input:
%   t: time
%   y: contrast intensity
% ouput:
%   BAT: balus arrival time
%   TTP: time to peak
%   Cpre: baseline
%   CC: y-Cpre
%   CMX: max contrast
%   CNR: contrast intensity noise ratio (t:0~BAT)
% Author: Adam Huang, 2021/07/28, revised 2022/04/19
% adamhuan@gmail.com
%
% Prepared for the original paper: 
% Time to peak and full width at half maximum in MR perfusion: valuable 
% indicators for monitoring moyamoya patients after revasculariation
% Scientific Reports 11(1) (2021) 479
% by Adam Huang, Chung-Wei Lee, Hon-Man Liu
%
% LLM was adapted from:
% "An automatic approach for estimating bolus arrival time in dynamic 
% contrast MRI using piecewise continuous regression models"
% by LH Cheong, TS Koh, and Z Hou

len=length(y);
yy=y(max(1,len-5):end);
[~,dist]=max(yy);
len=len-length(yy)+dist;
TTP=t(len);
[c1,c2]=size(y);
if c2>c1
    y=y';
end
C=y(1:len);
ix=1;
err=realmax;% a big number
if flag>0
    figure,
    plot(t,y,'r');
end
for i=2:len-1
    X=ones(len,2);
    X(:,2)=0;
    for j=i+1:len
        X(j,2)=t(j)-t(i);
    end
%     B=inv(transpose(X)*X)*(transpose(X)*C); % find inv() explicitly
    B=(transpose(X)*X)\(transpose(X)*C); % '\' using gauss elimination
    D=C-X*B;
    e=norm(D);
    if e<err
        ix=i;
        err=e;
        if flag>0
            hold on,
            plot(t(1:len),X*B,'g');
            hold off;
        end
    end
end
BAT=t(ix);
md=(min(y(1:BAT))+y(TTP))/2;
for i=BAT:TTP
    if y(i)>=md
        MD=i;
        break;
    end
end