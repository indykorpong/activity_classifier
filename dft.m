
filepath = 'C:/Users/Indy/Desktop/coding/Dementia_proj/testrun_ai_ped.csv';
f = csvread(filepath,1,6,[1,6,1938,6]);
disp(f(1,1));
y = fft(f);
m = abs(y);

p = unwrap(angle(y));
T = 0.16;
f = (0:length(y)-1)*(1/T)/length(y); 
plot(f,m);
ylim([0,15])
xlim([0,6.25])