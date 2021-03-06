clear all; close all; clc;
% Koch Gaitan Nemati HW2 Stoch
%% Part 1 radar Detection

snr = 10; 
not_present_prob = .8;
num_sims = 1e3;
A = 1;
var = A/snr;
format longG;

accuracys = zeros(1,num_sims); %get an average accuracy for MAP
for l = 1:num_sims
    x = zeros(2,num_sims);
    for i = 1:num_sims
        if rand > not_present_prob %is present
            x(1,i) = sqrt(var) * randn(1) + A;
            x(2,i) = 1;
        else %isnt present
            x(1,i) = sqrt(var) * randn(1);
            x(2,i) = 0;
        end
    end
    pdf_there = 1/(sqrt(2 * pi * var)) * exp(-(x(1,:) - A) .^ 2 / (2 * var));
    pdf_not_there = 1/(sqrt(2 * pi * var)) * exp(-x(1,:) .^2 / (2 * var));
    predictions = (pdf_there * (1 - not_present_prob) > pdf_not_there * not_present_prob);
    accuracys(l) = sum(predictions == x(2,:)) / num_sims; %error rate is 1 - accuracy
end
accuracy  = mean(accuracys);

%ROC Calculations
snrs = [.01 .1 1 5]; 
etas = linspace(-100,100,num_sims);
pds = zeros(length(snrs),length(etas)); %storage for P_d's 
pfas = zeros(length(snrs),length(etas)); %storage for P_fa's
for i = 1:length(snrs)
    snr = snrs(i);
    var = A / snr;
    x = zeros(2,num_sims);
    for j = 1:num_sims %generate the data at this snr
        if rand > not_present_prob %is present
            x(1,j) = sqrt(var) * randn(1) + A;
            x(2,j) = 1;
        else %isnt present
            x(1,j) = sqrt(var) * randn(1);
            x(2,j) = 0;
        end
    end
    pdf_there = 1/(sqrt(2 * pi * var)) * exp(-(x(1,:) - A) .^ 2 / (2 * var));
    pdf_not_there = 1/(sqrt(2 * pi * var)) * exp(-x(1,:) .^2 / (2 * var));
    for j = 1:length(etas) %neyman pearson test
        eta = etas(j);    
        predictions = (pdf_there ./ pdf_not_there > eta);
        pds(i,j) = sum((predictions + x(2,:)) == 2) / sum(x(2,:) == 1);
        pfas(i,j) = sum((predictions - x(2,:)) == 1) / sum(x(2,:) == 0);
    end
end

figure(); 
subplot(2,2,1); plot(pfas(1,:),pds(1,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, SNR = .01');
subplot(2,2,2);plot(pfas(2,:),pds(2,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, SNR = .1');
subplot(2,2,3); plot(pfas(3,:),pds(3,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, SNR = 1');
subplot(2,2,4); plot(pfas(4,:),pds(4,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, SNR = 5');

%assume eta = (.8 * 1) / (.2 * 10)
factor = 10;
eta = not_present_prob / ((1 - not_present_prob) * factor);
predictions = (pdf_there ./ pdf_not_there > eta);
pd = sum((predictions + x(2,:)) == 2) / sum(x(2,:) == 1);
pfa = sum((predictions - x(2,:)) == 1) / sum(x(2,:) == 0);
figure();
plot(pfas(4,:),pds(4,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, SNR = 5');
text(pfa,pd,'\leftarrow C_0_1 = 10 C_1_0','Color','red','FontSize',12);

%minimax decision is pd = 1 - pfa/10
pfas_mm = linspace(0,1,num_sims);
pds_mm = 1 - pfas_mm/10; 
[x,y] = intersections(pfas(4,:),pds(4,:),pfas_mm,pds_mm,1); %external function
text(x,y,'\leftarrow Mini-Max decision rule','Color','red','Fontsize',12);


%plot risk as a-prioris are varied at SNR = 5
P_0  = linspace(0,1,num_sims);
P_1 = ones(1,num_sims) - P_0;
snr = 10;
var = A / snr;
eta = P_0 ./ (factor * P_1);
pfa = 1 - normcdf((2 * var * log(eta) + A^2) / (2 * A * sqrt(var)));
pfd = normcdf(((2 * var * log(eta) + A^2) / (2 * A) - A) / sqrt(var));
risks = P_0 .* pfa + factor * P_1 .* pfd;

figure();
plot(P_0,risks); xlabel('P_0'); ylabel('Risk'); title('Risk as a Function of A-Prioris, SNR = 10');


% assume aprioris not known
% 'decision rule' : p_d = 1 - (p_f_a / 10) (figure 2 shows decision rule)
% risk = p_f_a at some eta which minimizes maximum risk
[biggest,i] = max(risks);
eta = eta(i);
pfa = 1 - normcdf((2 * var * log(eta) + A^2) / (2 * A * sqrt(var)));
hold on; plot(P_0,pfa * ones(1,num_sims)); 
legend('Neyman Pearson','MiniMax');


% assume not there -> Y = A + Z, Z is N(O,var_z)
% there -> Y = A + X, X is N(0,var_x), var_x < var_z

ratio_z_xs = [1.01 4 16 32];
x = zeros(2,num_sims);
A = 1;
var_x = 1;
var_z = 15 * var_x;
accuracys = zeros(1,num_sims);
for l = 1:num_sims
    for i = 1:num_sims
        if rand > not_present_prob %is present
            x(1,i) = sqrt(var_x) * randn(1) + A;
            x(2,i) = 1;
        else %isnt present
            x(1,i) = sqrt(var_z) * randn(1) + A;
            x(2,i) = 0;
        end
    end

    pdf_there = 1/(sqrt(2 * pi * var_x)) * exp(-(x(1,:) - A) .^ 2 / (2 * var_x));
    pdf_not_there = 1/(sqrt(2 * pi * var_z)) * exp(-(x(1,:) - A) .^2 / (2 * var_z));
    predictions = ((pdf_there * (1 - not_present_prob)) > (pdf_not_there * not_present_prob));
    accuracys(l) = sum(predictions == x(2,:)) / num_sims; 
end
accuracy = 1-mean(accuracys); %error rate is 1 - accuracy
%ROC Calculations
etas = linspace(-100,100,num_sims);
pds = zeros(length(ratio_z_xs),length(etas)); %storage for P_d's 
pfas = zeros(length(ratio_z_xs),length(etas)); %storage for P_fa's
for i = 1:length(ratio_z_xs)
    var_z = var_x * ratio_z_xs(i);
    x = zeros(2,num_sims);
    for j = 1:num_sims %generate the data at this snr
        if rand > not_present_prob %is present
            x(1,j) = sqrt(var_x) * randn(1) + A;
            x(2,j) = 1;
        else %isnt present
            x(1,j) = sqrt(var_z) * randn(1) + A;
            x(2,j) = 0;
        end
    end
    pdf_there = 1/(sqrt(2 * pi * var_x)) * exp(-(x(1,:) - A) .^ 2 / (2 * var_x));
    pdf_not_there = 1/(sqrt(2 * pi * var_z)) * exp(-(x(1,:) - A) .^2 / (2 * var_z));
    for j = 1:length(etas) %neyman pearson test
        eta = etas(j);    
        predictions = (pdf_there ./ pdf_not_there > eta);
        pds(i,j) = sum((predictions + x(2,:)) == 2) / sum(x(2,:) == 1);
        pfas(i,j) = sum((predictions - x(2,:)) == 1) / sum(x(2,:) == 0);
    end
end

figure(); 
subplot(2,2,1); plot(pfas(1,:),pds(1,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, Ratio = 1.01');
subplot(2,2,2);plot(pfas(2,:),pds(2,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, Ratio = 4');
subplot(2,2,3); plot(pfas(3,:),pds(3,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, Ratio =  16');
subplot(2,2,4); plot(pfas(4,:),pds(4,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, Ratio = 32');



%% Part 2 Photon Detection

rate_0 = 5;

%ROC Calculations
rate_1s = [6 10 15 50]; 
etas = linspace(-100,100,num_sims);
pds = zeros(length(snrs),length(etas)); %storage for P_d's 
pfas = zeros(length(snrs),length(etas)); %storage for P_fa's
for i = 1:length(rate_1s)
    rate_1 = rate_1s(i);
    x = zeros(2,num_sims);
    for j = 1:num_sims %generate the data at these rates
        if rand > .5 %rate 1
            x(1,j) = exprnd(1/rate_1);
            x(2,j) = 1;
        else %rate 0
            x(1,j) = exprnd(1/rate_0);
            x(2,j) = 0;
        end
    end
    pdf_there = exppdf(x(1,:),1/rate_1);
    pdf_not_there = exppdf(x(1,:),1/rate_0);
    for j = 1:length(etas) %neyman pearson test
        eta = etas(j);    
        predictions = (pdf_there ./ pdf_not_there > eta);
        pds(i,j) = sum((predictions + x(2,:)) == 2) / sum(x(2,:) == 1);
        pfas(i,j) = sum((predictions - x(2,:)) == 1) / sum(x(2,:) == 0);
    end
end

figure(); 
subplot(2,2,1); plot(pfas(1,:),pds(1,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, \lambda_1 = 6, \lambda_0 = 5');
subplot(2,2,2);plot(pfas(2,:),pds(2,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, \lambda_1 = 10, \lambda_0 = 5');
subplot(2,2,3); plot(pfas(3,:),pds(3,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, \lambda_1 = 15, \lambda_0 = 5');
subplot(2,2,4); plot(pfas(4,:),pds(4,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, \lambda_1 = 20, \lambda_0 = 5');


%% Intro to Pattern Classification and ML

data = csvread('iris.csv');
%grab half of each class for train and test
train_ex = [data(1:24,1:4); data(51:75,1:4); data(101:125,1:4)];
train_lab = [data(1:24,5); data(51:75,5); data(101:125,5)];
test_ex = [data(25:50,1:4); data(76:100,1:4); data(126:150,1:4)];
test_lab = [data(25:50,5); data(76:100,5); data(126:150,5)];

% estimate parameters associated with each class assuming each class
% follows a gaussian distribution
mu_hat_1 = mean(train_ex(1:24,:));
cov_hat_1 = cov(train_ex(1:24,:));
mu_hat_2 = mean(train_ex(25:49,:));
cov_hat_2 = cov(train_ex(25:49,:));
mu_hat_3 = mean(train_ex(50:74,:));
cov_hat_3 = mean(train_ex(50:74,:));

predictions = zeros(length(test_ex),1);
%calculate the class according to the likelihood test with eta = 1
for i = 1:length(test_ex) 
    pred = 1;
    tmp = test_ex(i,:);
    if mvnpdf(tmp,mu_hat_2,cov_hat_2) > mvnpdf(tmp,mu_hat_1,cov_hat_1)
        pred = 2;
    
        if mvnpdf(tmp,mu_hat_3,cov_hat_3) > mvnpdf(tmp,mu_hat_2,cov_hat_2)
            pred = 3;
        end
    elseif mvnpdf(tmp,mu_hat_3,cov_hat_3) > mvnpdf(tmp,mu_hat_1,cov_hat_1)
        pred = 3;
    end
        
    predictions(i) = pred;
end

%results -- full data
acc = sum(predictions == test_lab) / length(predictions)
c_mat = confusionmat(test_lab,predictions)


% reduce the dimensionality
train_ex = [data(1:24,1:2); data(51:75,1:2); data(101:125,1:2)];
train_lab = [data(1:24,5); data(51:75,5); data(101:125,5)];
test_ex = [data(25:50,1:2); data(76:100,1:2); data(126:150,1:2)];
test_lab = [data(25:50,5); data(76:100,5); data(126:150,5)];

% estimate parameters associated with each class assuming each class
% follows a gaussian distribution
mu_hat_1 = mean(train_ex(1:24,:));
cov_hat_1 = cov(train_ex(1:24,:));
mu_hat_2 = mean(train_ex(25:49,:));
cov_hat_2 = cov(train_ex(25:49,:));
mu_hat_3 = mean(train_ex(50:74,:));
cov_hat_3 = mean(train_ex(50:74,:));

predictions = zeros(length(test_ex),1);
%calculate the class according to the likelihood test with eta = 1
for i = 1:length(test_ex) 
    pred = 1;
    tmp = test_ex(i,:);
    if mvnpdf(tmp,mu_hat_2,cov_hat_2) > mvnpdf(tmp,mu_hat_1,cov_hat_1)
        pred = 2;
    
        if mvnpdf(tmp,mu_hat_3,cov_hat_3) > mvnpdf(tmp,mu_hat_2,cov_hat_2)
            pred = 3;
        end
    elseif mvnpdf(tmp,mu_hat_3,cov_hat_3) > mvnpdf(tmp,mu_hat_1,cov_hat_1)
        pred = 3;
    end
        
    predictions(i) = pred;
end

%results -- reduced dimensionality
acc = sum(predictions == test_lab) / length(predictions)
c_mat = confusionmat(test_lab,predictions)