% Path to your CALS MEX
addpath("/home/chris/projects/CP-CALS-priv/cmake-build-release-clang-mat/")
% Path to TTB
addpath("/home/chris/projects/tensor_toolbox/")

clc
clear all
echo off
rand('state', 0)

mode1 = 100;
mode2 = 100;
mode3 = 100;
comp_min = 1;
comp_max = 10;
copies = 100;

tic;
% X = tensor(rand(mode1, mode2, mode3));
X = full(ktensor({rand(mode1, 5), rand(mode2, 5), rand(mode3, 5)}));
components = zeros(1, 10);
k = 1;
for i = comp_min:comp_max
    for j = 1:copies
        components(k) = i;
        k = k + 1;
    end    
end
comp_sum = sum(components);

ktensors = {};
for c = 1:size(components, 2)
    ktensors{c} = ktensor({rand(mode1, components(c)),
                           rand(mode2, components(c)),
                           rand(mode3, components(c))});
end
time_gen = toc;
fprintf('Time to generate tensor and models: %f\n', time_gen);

M = {};
disp(' ')
disp('TTB ALS...')
tic
for k = 1:size(ktensors, 2)
    M{k} = cp_als(X, size(ktensors{k}.lambda, 1), 'maxiters', 50, 'tol', 1e-4, 'init', ktensors{k}.U, 'printitn', 0);
end
time_tt = toc;
pause(3)
disp(' ')
disp('CALS...')
tic
M1 = cp_cals(X, components, ktensors, 'mttkrp-method', 'auto', 'tol', 1e-4, 'maxiters', 50, 'buffer-size',  comp_sum);
time_cals = toc;

t = size(M, 2) * [];
for i = 1:size(M,2)
    t(i) = norm(X - tensor(M{i})) - norm(X - tensor(M1{i}));
end
disp(' ')
disp('-----------------------------------------------------------')
fprintf('ALS time: %0.5f\n', time_tt);
fprintf('CALS time: %0.5f\n', time_cals);
fprintf('Speedup: %0.2f\n', time_tt / time_cals);
fprintf('Absolute difference of error per model:\n');
for i = 1:size(M,2)
    fprintf('%f %d \n', abs(t(i)), i);
end
fprintf('\n');
fprintf('Mean difference of errors: %f\n', mean(t));
fprintf('Max difference of errors: %f\n', max(t));
