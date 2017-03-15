clear all; close all; clc;
% Koch Gaitan Nemati HW2 Stoch
%% Part 1 radar Detection

snr = 10; 
not_present_prob = .8;
num_sims = 1e3;
A = 1;
var = A/snr;

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
etas = linspace(-1000,1000,10000);
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
        pfas(i,j) = sum((predictions - x(2,:)) == 1) / sum(x(2,:) == 1);
    end
end

figure(); 
subplot(2,2,1); plot(pfas(1,:),pds(1,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, SNR = .01');
subplot(2,2,2);plot(pfas(2,:),pds(2,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, SNR = .1');
subplot(2,2,3); plot(pfas(3,:),pds(3,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, SNR = 1');
subplot(2,2,4); plot(pfas(4,:),pds(4,:)); xlabel('P_f_a'); ylabel('P_d'); title('ROC, SNR = 5');