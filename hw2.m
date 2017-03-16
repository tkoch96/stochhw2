clear all; close all; clc;
% Koch Gaitan Nemati HW2 Stoch
%% Part 1 radar Detection

snr = 10; 
not_present_prob = .8;
num_sims = 1e3;
A = 1;
var = A/snr;
format longG;

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
accuracy = sum(predictions == x(2,:)) / num_sims; %error rate is 1 - accuracy

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
    for j = 1:length(etas)
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
P_0s = linspace(0,1,num_sims);
P_1s = ones(1,num_sims) - P_0s;
pfas = zeros(1,num_sims);
risks = zeros(1,num_sims);
snr = 10;
var = A / snr;
for i = 1:num_sims
     x = zeros(2,num_sims);
    for j = 1:num_sims %generate the data according to these priors
        if rand > P_0s(i) %is present
            x(1,j) = sqrt(var) * randn(1) + A;
            x(2,j) = 1;
        else %isnt present
            x(1,j) = sqrt(var) * randn(1);
            x(2,j) = 0;
        end
    end
    pdf_there = 1/(sqrt(2 * pi * var)) * exp(-(x(1,:) - A) .^ 2 / (2 * var));
    pdf_not_there = 1/(sqrt(2 * pi * var)) * exp(-x(1,:) .^2 / (2 * var));
    eta = P_0s(i) / (factor * P_1s(i));
    predictions = (pdf_there ./ pdf_not_there > eta);
    pd = sum((predictions + x(2,:)) == 2) / sum(x(2,:) == 1);
    pfa = sum((predictions - x(2,:)) == 1) / sum(x(2,:) == 0);
    pfas(i) = pfa;
    risks(i) = P_0s(i) * pfa + factor * P_1s(i) * pd;
end
figure();
plot(P_0s,risks); xlabel('P_0'); ylabel('Risk'); title('Risk as a Function of A-Prioris, SNR = 10');


% assume aprioris not known
% 'decision rule' : p_d = 1 - (p_f_a / 10) (figure 2 shows decision rule)
% risk = p_f_a
hold on; plot(P_0s,pfas); 
legend('Neyman Pearson','MiniMax');


% assume not there -> Y = A + Z, Z is N(O,var_z)
% there -> Y = A + X, X is N(0,var_x), var_x < var_z

ratio_z_xs = [2 4 8 16];
x = zeros(2,num_sims);
A = 5;
var_x = 1;
var_z = 5 * var_x;
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
pdf_not_there = 1/(sqrt(2 * pi * var_z)) * exp(-x(1,:) .^2 / (2 * var_z));
predictions = (pdf_there * (1 - not_present_prob) > pdf_not_there * not_present_prob);
accuracy = sum(predictions == x(2,:)) / num_sims; %error rate is 1 - accuracy

%ROC Calculations
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
    for j = 1:length(etas)
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
