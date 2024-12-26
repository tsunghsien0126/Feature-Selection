%%  SFS
clear; close; clc;

s = readtable('wdbc.csv');

D = table2array(s(:,3:end));
label = string(s{:,2}); % acquire label (M:negative,B:positive)

for i = 1:569
    if (label(i)=='M') 
        D(i,31) = -1;
    else
        D(i,31) = 1;
    end
end

Pdata = D';
Pdata(:,Pdata(end,:) == -1) = [];

Ndata = D';
Ndata(:,Ndata(end,:) == 1) = [];

clearvars -except D Pdata Ndata;

% LDA
Ns=30;
P1 = Pdata(:,1:178);  P2 = Pdata(:,179:end);
N1 = Ndata(:,1:106);  N2 = Ndata(:,107:end); % 30 features & 1 label
 
for h=1:Ns
    if h==1 % all feature
        % perform 2-fold CV
        for k = 1:2
            if k == 1
                P = P1;N = N1;
                test = [P2 N2];
            else
                P = P2;N = N2;
                test = [P1 N1];  
            end
      
            Pp = length(P)/(length(P)+length(N));  % Positive Prior Probability
            Pn = length(N)/(length(P)+length(N));  % Negative Prior Probability


    % training 
            for j = 1:30
                clear mp mn covp covn cov w b;
                mp (h,1) = mean(P(j,:));
                mn (h,1) = mean(N(j,:));
                  
                covp = sum(((P(j,:)-mp)*((P(j,:)-mp)')))./(length(P)-1);
                covn = sum(((N(j,:)-mn)*((N(j,:)-mn)')))./(length(N)-1);
                cov = Pp.*covp + Pn.*covn;
        
                w = (((mp-mn)')/(cov))';
                b = -0.5*((((mp-mn)')/(cov))*(mp+mn))-log(Pn/Pp);
                
    % testing
                DB = (w').*test(j,:)+b;
                if k == 1
                    CR1(j) = 0;
                    for i = 1:length(DB)
                        if DB(i) *test(end,i)>0
                            CR1(j) = CR1(j)+1;
                        end
                    end
                CR1(j) = CR1(j)/length(DB);
                else
                    CR2(j) = 0;
                    for i = 1:length(DB)
                         if DB(i)*test(end,i) > 0
                            
                         CR2(j) = CR2(j)+1;
                            
                        end
                    end
                CR2(j) = CR2(j)/length(DB);
                end
            end
        end
        CR = (CR1+CR2)/2;
        [A,B] = max(CR);
        
        PSF1 = P1(B,:);
        NSF1 = N1(B,:);
        PSFm1 = mean(P1(B,:));
        NSFm1 = mean(N1(B,:));
    
        PSF2 = P2(B,:);
        NSF2 = N2(B,:);
        PSFm2 = mean(P2(B,:));
        NSFm2 = mean(N2(B,:));
    
        P1(B(h),:) = [];  P2(B(h),:) = [];
        N1(B(h),:) = [];  N2(B(h),:) = [];
    else 
        Nc = Ns+1-h;  
    
        for k = 1:2
            if k == 1
                P = P1;
                N = N1;
                test = [P2 N2];
            else
                P = P2;
                N = N2;
                test = [P1 N1];  
            end
        
            Pp = length(P)/(length(P)+length(N));  % Positive Prior Probability
            Pn = length(N)/(length(P)+length(N));  % Negative Prior Probability
        
            for j = 1:Nc
                clear mpc mpn cov w b
                mpc = mean(P(j,:));
                mnc = mean(N(j,:));
                if k == 1
                    mpc = [PSFm1 ; mpc];
                    mnc = [NSFm1 ; mnc];
                else
                    mpc = [PSFm2 ; mpc];
                    mnc = [NSFm2 ; mnc];
                end
            
                covp = zeros(h,h);  covn = zeros(h,h);
                for i = 1:length(P)
                    if k == 1
                        covp = covp + (([PSF1(:,i) ; P(j,i)]-mpc)*(([PSF1(:,i) ; P(j,i)]-mpc)'));
                    else
                        covp = covp + (([PSF2(:,i) ; P(j,i)]-mpc)*(([PSF2(:,i) ; P(j,i)]-mpc)'));
                    end
                end
                covp = covp /(length(P)-1);
    
                for i = 1:length(N)
                    if k == 1
                        covn = covn + (([NSF1(:,i) ; N(j,i)]-mnc)*(([NSF1(:,i) ; N(j,i)]-mnc)'));
                    else
                        covn = covn + (([NSF2(:,i) ; N(j,i)]-mnc)*(([NSF2(:,i) ; N(j,i)]-mnc)'));
                    end
                end
                covn = covn /(length(N)-1);
               
                cov = Pp.*covp + Pn.*covn;
            
    % calculating w, b
                w = (((mpc-mnc)')/(cov))';
                b = -0.5*((((mpc-mnc)')/(cov))*(mpc+mnc))-log(Pn/Pp);
                
    % testing
                if k == 1
                    DB = (w')*[[PSF2 NSF2] ; test(j,:)]+b;
                else
                    DB = (w')*[[PSF1 NSF1] ; test(j,:)]+b;
                end
    
                if k == 1
                    clear CR1;
                    CR1(j) = 0;
                    for i = 1:length(DB)
                        if DB(i)*test(end,i) > 0
                            CR1(j) = CR1(j)+1;
                        end
                    end
                CR1(j) = CR1(j)/length(DB);
                else
                    clear CR2;
                    CR2(j) = 0;
                    for i = 1:length(DB)
                         if DB(i)*test(end,i) > 0
                            CR2(j) = CR2(j)+1;
                        end
                    end
                CR2(j) = CR2(j)/length(DB);
                end
            end
        end
        clear CR
        CR = (CR1+CR2)./2;
        [A(h),B(h)] = max(CR);
    
        PSF1(h,:) = P1(B(h),:);
        NSF1(h,:) = N1(B(h),:);
        PSFm1(h,1) = mean(P1(B(h),:));
        NSFm1(h,1) = mean(N1(B(h),:));
    
        PSF2(h,:) = P2(B(h),:);
        NSF2(h,:) = N2(B(h),:);
        PSFm2(h,1) = mean(P2(B(h),:));
        NSFm2(h,1) = mean(N2(B(h),:));
    
        P1(B(h),:) = [];  P2(B(h),:) = [];
        N1(B(h),:) = [];  N2(B(h),:) = [];
    end
end
  
B = [B;B];
B(2,2) = B(2,2)+1;
B(2,3) = B(2,3)+1;

[FA,FB] = max(A);
for i = 1:FB
    if i == 1
        SF = [D(:,B(2,i))];
    else
        SF = [SF D(:,B(2,i))];
    end
end
SF = [B(2,1:FB) ; SF];

% plot
CR_SFS = A;
figure(1)
plot(1:30,CR_SFS.*100);
xlim([1 30]);
xlabel('Number of Features');
xticks(1:2:29);
ylabel('Classification Accuracy (%)');
ylim([80 100]);
grid on;
title('SFS Criterion');
set(gca,'fontname','arial','fontsize',10);

clearvars -except CR_SFS SF; clc;

%%  Fisher's Criterion

s = readtable('wdbc.csv');
D = table2array(s(:,3:end));
label = string(s{:,2});

for i = 1:569
    if label(i)=='M'
        D(i,31) = -1;
    else
        D(i,31) = 1;
    end
end

Pdata = D';
Pdata(:,Pdata(end,:) == -1) = [];

Ndata = D';
Ndata(:,Ndata(end,:) == 1) = [];

clearvars -except D Pdata Ndata CR_SFS SF;

Pp = length(Pdata)/(length(Pdata)+length(Ndata)); % Positive Prior Probability
Np = length(Ndata)/(length(Pdata)+length(Ndata)); % Negative Prior Probability

Pmclass=zeros(30,1); Nmclass=zeros(30,1);mall=zeros(30,1);
for i = 1:30
    Pmclass(i,1) = mean(Pdata(i,:));
    Nmclass(i,1) = mean(Ndata(i,:));
    mall(i,1) = mean(D(:,i));
end

Sw1 = zeros(30,1);
for i = 1:30
    for j = 1:length(Pdata)
        Sw1(i,1) = Sw1(i,1)+(Pdata(i,j)-Pmclass(i))*((Pdata(i,j)-Pmclass(i))');
    end
end

Sw2 = zeros(30,1);
for i = 1:30
    for j = 1:length(Ndata)
        Sw2(i,1) = Sw2(i,1)+(Ndata(i,j)-Nmclass(i))*((Ndata(i,j)-Nmclass(i))');
    end
end
Sw = (Pp.*Sw1)+(Np.*Sw2);

Sb=zeros(30,1);
for i = 1:30
    Sb(i,1) = (length(Pdata).*((Pmclass(i)-mall(i))*((Pmclass(i)-mall(i))'))) + (length(Ndata).*((Nmclass(i)-mall(i))*((Nmclass(i)-mall(i))')));
end

F(:,1) = Sb.*Sw;
F = [F ; min(F)-1];

Darrange = [D(:,:) ; F'];
[~,I] = sort(Darrange(end,:),'descend');
SortedD = Darrange(:,I);
clear Darrange I;
SortedD(end,:) = [];

clearvars -except D SortedD CR_SFS SF;

Pdata = SortedD;
Pdata(Pdata(:,end) == -1,:) = [];
Pdata = Pdata';

Ndata = SortedD;
Ndata(Ndata(:,end) == 1,:) = [];
Ndata = Ndata';
clearvars -except SortedD D Pdata Ndata CR_SFS SF;

P1 = Pdata(:,1:178);  P2 = Pdata(:,179:end);
N1 = Ndata(:,1:106);  N2 = Ndata(:,107:end);


%%%
for I = 1:30
% perform 2-fold CV
    for k = 1:2
        if k == 1
            P = P1;
            N = N1;
            test = [P2 N2];
        else
            P = P2;
            N = N2;
            test = [P1 N1];  
        end    
    
        Pp = length(P)/(length(P)+length(N));  % Positive Prior Probability
        Pn = length(N)/(length(P)+length(N));  % Negative Prior Probability
    

% training 
        mp(I,1) = mean(P(I,:));
        mn(I,1) = mean(N(I,:));

        covp = zeros(I,I);  covn = zeros(I,I);
        for i = 1:length(P)
            covp = covp + ((P(1:I,i)-mp)*((P(1:I,i)-mp)'));
        end
        covp = covp./(length(P)-1);
        for i = 1:length(N)
            covn = covn + ((N(1:I,i)-mn)*((N(1:I,i)-mn)'));
        end
        covn = covn./(length(N)-1);
        cov = Pp.*covp + Pn.*covn;

        
% calculating w, b
        clear w b DB;
        w = (((mp-mn)')/(cov))';
        b = -0.5.*((((mp-mn)')/(cov))*(mp+mn))-log(Pn/Pp);
            
% testing
        for j = 1:length(test)
            DB(j) = (w')*test(1:I,j)+b;
        end
        if k == 1
            CR1(I) = 0;
            for i = 1:length(DB)
                if DB(i) > 0
                    if test(end,i) == 1
                        CR1(I) = CR1(I)+1;
                    end
                elseif DB(i) < 0
                    if test(end,i) == -1
                       CR1(I) = CR1(I)+1;
                    end
                end
            end
        CR1(I) = CR1(I)/length(DB);
        else
        CR2(I) = 0;
            for i = 1:length(DB)
                if DB(i) > 0
                    if test(end,i) == 1
                        CR2(I) = CR2(I)+1;
                    end
                elseif DB(i) < 0
                    if test(end,i) == -1
                       CR2(I) = CR2(I)+1;
                    end
                end
            end
        CR2(I) = CR2(I)/length(DB);
        end        
    end
end
CR_Fisher = (CR1+CR2)/2;
figure(2)
plot(1:30,CR_Fisher.*100);
xlabel('Number of Top-rank Features');
xlim([1 30]);
xticks(1:2:29);
ylabel('Classification Accuracy (%)');
ylim([80 100]);
grid on;
title('Fishers Criterion');
set(gca,'fontname','arial','fontsize',10);

SFS_selected_feature = SF;
Fisher_selected_feature = D;
Fisher_selected_feature(:,end) = [];
clearvars -except CR_Fisher CR_SFS D SFS_selected_feature Fisher_selected_feature;
