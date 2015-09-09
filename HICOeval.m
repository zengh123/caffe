clc;
clear all;
anno_file = '~/Desktop/anno.mat'
anno = load(anno_file);
resf = load('~/Desktop/test-result.mat');
backend = 'test'

num_class = 600;
allrec = zeros(num_class,1);



if strcmp(backend,'test') == 1
lball = anno.anno_test;
lball(lball == -1) = 0; lball(lball == -2) = 0;
scall = resf.all_score(1:9155,:);
scall = scall';
predict = zeros(size(scall));
predict(scall>0.5)=1;
allxor = ~xor(lball,predict);
accu = sum(sum(allxor)) / (size(scall,1) * size(scall,2));
for i = 1:num_class
    label = anno.anno_test(i,:);
    ii = label ~= 0;
    label = label(ii);
    score = resf.all_score(1:9155,i);
    score = score(ii);
    label = label';
%    score(label == -2) = -1e10;
    label(label == 0) = -1; label( label == -2) = -1;
    [rec, prec, ap(i)] = eval_pr_score_label(score, label, sum(label == 1), 0);
    i
end

end

if strcmp(backend,'train') == 1
lball = anno.anno_train;
lball(lball == -1) = 0; lball(lball == -2) = 0;
lball(:,6706:36630) = lball(:,6707:36631);
lball = lball(:,1:36630);
scall = resf.all_score(1:36630,:);
scall = scall';
predict = zeros(size(scall));
predict(scall>0.5)=1;
allxor = ~xor(lball,predict);
accu = sum(sum(allxor)) / (size(scall,1) * size(scall,2));

count_pos = sum(sum(lball == 1));
temp_predict = predict.* (lball == 1);
true_positive = sum(sum(temp_predict == 1));
temp_predict2 = predict.* (lball == 0);
false_positive = sum(sum(temp_predict2 == 1));

psu_recall = true_positive / count_pos;
psu_precision = true_positive / (true_positive + false_positive);

for i = 1:num_class
    label = anno.anno_train(i,:);
    %ii = (label ~= 0);
    %label = label(ii);
    score = resf.all_score(1:36631,i);
    %score = score(ii);
    label = label';
    label(6706:36630) = label(6707:36631);
    
    %score(label == -2) = -1e10;
    label(label == 0) = -1; label( label == -2) = -1;
    [rec, prec, ap(i)] = eval_pr_score_label(score(1:36630), label(1:36630), sum(label == 1), 0);
    i
end 
end

map = mean(ap);
[res,ind] = sort(ap);

K = 10;
%the lowest 10
low(1:K,1) = anno.list_action(ind(1:K));
for i=1:K
low(i).ap = ap(ind(i));
end

%the highest 10
high(1:K,1) = anno.list_action(ind(end-K+1:end));
for i=1:K
high(i).ap = ap(ind(end-K+i));
end