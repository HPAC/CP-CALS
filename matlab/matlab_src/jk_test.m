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

tic;
% X = tensor(rand(mode1, mode2, mode3));
X = full(ktensor({rand(mode1, comp), rand(mode2, comp), rand(mode3, comp)}));

ktensor_ref = ktensor({rand(mode1, comp), rand(mode2, comp), rand(mode3, comp)});

kt_ref_res = cp_als(X, size(ktensor_ref.lambda, 1), 'maxiters', 1000, 'tol', 1e-6, 'init', ktensor_ref.U, 'printitn', 0);

time_gen = toc;
fprintf('Time to generate tensor and models: %f\n', time_gen);

% M_ref = cp_als(X, size(ktensor_ref.lambda, 1), 'maxiters', 50, 'tol', 1e-4, 'init', ktensor_ref.U, 'printitn', 0);
tic;
for k = 1:mode1
    X_copy = X;
    X_copy = double(reshape(X_copy, [mode1 mode2*mode3]));
    X_copy(k, :) = [];
    X_copy = tensor(reshape(X_copy, [mode1-1 mode2 mode3]));
    
    kt_copy = kt_ref_res;
    kt_copy.U{1}(k, :) = [];
    M{k} = cp_als(X_copy, size(kt_copy.lambda, 1), 'maxiters', 100, 'tol', 1e-5, 'init', kt_copy.U, 'printitn', 0);
end
time_als = toc;
fprintf('Time to compute using ALS: %f\n', time_als);


kt_ttb = M{1};
lambda = double(kt_ttb.lambda');
for f = 1:3
    t =  double(kt_ttb.U{f});
    if (f == 1)
        t_1 = ones(mode1, comp);
        t_1(1,:) = 0;
        t_1(logical(t_1)) = t;
        t_1(1,:) = nan;
        U{f} = t_1;
    else
        U{f} = t;
    end
    
end
clear kt_ttb kt_copy X_copy

for k = 2:mode1
    lambda = [lambda, double(M{k}.lambda')];
    for f = 1:3
        t =  double(M{k}.U{f});
        if (f == 1)
            t_1 = ones(mode1, comp);
            t_1(k,:) = 0;
            t_1(logical(t_1)) = t;
            t_1(k,:) = nan;
            U{f} = [U{f} t_1];
        else
            U{f} = [U{f} t];
        end
    end
end
Uta{1} = reshape(U{1}, [mode1 comp mode1]);
Uta{2} = reshape(U{2}, [mode2 comp mode2]);
Uta{3} = reshape(U{3}, [mode3 comp mode3]);
kta = ktensor(lambda', U);
% X_jk_0 = X;
% X_jk_0 = double(reshape(X_jk_0, [mode1 mode2*mode3]));
% X_jk_0(1, :) = [];
% X_jk_0 = tensor(reshape(X_jk_0, [mode1-1 mode2 mode3]));
% 
% norm(X_jk_0 - tensor(M{1}))
clear U lambda


disp('CALS JK...')
cals_input = {};
cals_input{1} = kt_ref_res;
tic;
M1 = cp_cals_jk(X, cals_input, 'mttkrp-method', 'auto', 'tol', 1e-5, 'maxiters', 100, 'buffer-size',  50);
time_cals = toc;
fprintf('Time to compute using CALS: %f\n', time_cals);


% kt_agg.U{1};
% lambda = double(M1{1}{3}.lambda);
% lambda = lambda(1:5);
% U{1} = double(M1{1}{3}.U{1});
% U{2} = double(M1{1}{3}.U{2});
% U{3} = double(M1{1}{3}.U{3});
% Utc{1} = reshape(U{1}, [mode1 comp mode1]);
% Utc{2} = reshape(U{2}, [mode2 comp mode2]);
% Utc{3} = reshape(U{3}, [mode3 comp mode3]);

% U{1} = U{1}(2:end, 1:5);
% U{2} = U{2}(1:end, 1:5);
% U{3} = U{3}(1:end, 1:5);
% dummy = normalize(ktensor(lambda, U));
% norm(X_jk_0 - tensor(M{1}))
% norm(X_jk_0 - tensor(dummy))
% norm(tensor(dummy) - tensor(M{1}))
% dummy.U{1};
% dummy = normalize(dummy);
% M{1} = normalize(M{1});
% dummy.U{1}' * M{1}.U{1}

kt_ref_res1 = {kt_ref_res.U{1}, kt_ref_res.U{2}, kt_ref_res.U{3}};

jka = jkparafac({Uta{1}, Uta{2}, Uta{3}}, kt_ref_res1);

stdba = std(jka{2}, [], 3);
stdca = std(jka{3}, [], 3);

norm(stdba - M1{1}{3}{2})
norm(stdca - M1{1}{3}{3})