% Path to your CALS MEX
addpath("/home/chris/projects/CP-CALS-priv/cmake-build-release-clang-mat/")
% Path to TTB
addpath("/home/chris/projects/tensor_toolbox/")

clc
clear all
echo off
rand('state', 0)

mode1 = 10;
mode2 = 10;
mode3 = 10;
comp = 5;

comp_min = 1;
comp_max = 10;
copies = 1;

components = zeros(1, (comp_max - comp_min + 1)*copies);
k = 1;
for i = comp_min:comp_max
    for j = 1:copies
        components(k) = i;
        k = k + 1;
    end    
end
comp_sum = sum(components);

tic;
X = full(ktensor({rand(mode1, comp), rand(mode2, comp), rand(mode3, comp)}));

time_gen = toc;
fprintf('Time to generate tensor and models: %f\n', time_gen);

disp('CALS JK...')
M1 = cp_cals_hybrid(X, components, [5], ...
                    'mttkrp-method', 'auto', ...
                    'tol', 1e-8, ...
                    'maxiters', 100, ...
                    'buffer-size',  5000);
