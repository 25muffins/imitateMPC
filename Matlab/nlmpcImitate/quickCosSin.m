v30 = load('v22CosSin.mat');


v30d = v30.d;
s = size(v30.d);
d = zeros(1,42);

for i = 1:s(1)
    d(i, :) = [v30d(i,:),...
        sin(v30d(i,14)), cos(v30d(i,14)),...
        sin(v30d(i,22)), cos(v30d(i,22))];
end
 %sin(v30d(i,3)), cos(v30d(i,3)),...
        %sin(v30d(i,9)), cos(v30d(i,9)),...
        %sin(v30d(i,17)), cos(v30d(i,17))];

save('v22cossinwithextrasauce','d')
