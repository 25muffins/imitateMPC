v30 = load('v30.mat');


v30d = v30.d;
s = size(v30.d);
d = zeros(1,38);

for i = 1:s(1)
    d(i, :) = [v30d(i,:),...
        sin(v30d(i,3)), cos(v30d(i,3)),...
        sin(v30d(i,9)), cos(v30d(i,9)),...
        sin(v30d(i,17)), cos(v30d(i,17))];
end

save('v30CosSin','d')