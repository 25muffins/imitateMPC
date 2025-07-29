v22 = load('v22.mat');


v22d = v22.d;
s = size(v22.d);
d = zeros(1,38);

for i = 1:s(1)
    d(i, :) = [v22d(i,:),...
        sin(v22d(i,3)), cos(v22d(i,3)),...
        sin(v22d(i,9)), cos(v22d(i,9)),...
        sin(v22d(i,17)), cos(v22d(i,17))];
end

save('v22CosSin','d')