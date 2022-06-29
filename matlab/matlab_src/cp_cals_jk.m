function out = cp_cals_jk(X, init_ktensors, varargin)

%  cp_cals_jk Compute multiple jackknifes from ktensors using CALS
%
%  Usage:
%
%  out = cp_cals_jk(X,init_ktensors) performs jackknifing on each ktensor
%        in init_ktensors.
%     'X' - The target tensor.
%	  'init_ktensors' - Cell array containing all the (initial) Ktensors to
%	  be jackkniffed.
%	The result out is a cell array containing (for every ktensor in
%	init_ktensors):
%   the reference ktensor, the fitted and permutation adjusted jackknife
%   ktensors and the stdB, stdC matrices.
%
%  out = cp_cals_jk(X,init_ktensors,'param',value,...) specifies optional
%  parameters and values.
%
%      Valid parameters and their default values are:
%        'tol' - Tolerance on difference in fit {1.0e-4} 
%        'maxiters' - Maximum number of iterations {50} 
%        'buffer-size' - Maximum size of the factor matrices buffer {4200} 
%        'update-method' - Method for updating the factor matrices 
%                          {unconstrained} ['unconstrained', 'nnls']
%        'mttkrp-method' - method for computing MTTKRP {auto}
%                          ['mttkrp', 'twostep0', 'twostep1', 'auto']
%        'no-ls'(default)/'ls' - Whether to use line search. 
%        'ls-interval' - Interval (per individual model) to apply 
%                        line search {5}
%        'ls-step' - Factor with which to jump in line search {1.2}
%        'no-cuda'(default)/'cuda' - Whether to use cuda (make sure the
%        binary is compiled with cuda support)
tic;
out = cp_cals_jk_driver(X, init_ktensors, varargin{:});

mode2 = size(X, 2);
mode3 = size(X, 3);
for i=1:size(out, 1)
	if size(out{i}, 1) == 3
		comp = size(out{i}{1}.U{1}, 2);
		out{i}{3}{2} = std(reshape(double(out{i}{2}.U{2}), [mode2 comp mode2]), [], 3);
		out{i}{3}{3} = std(reshape(double(out{i}{2}.U{3}), [mode3 comp mode3]), [], 3);
	end
end
cals_time = toc;
fprintf('Time for CALS JK: %f\n', cals_time);

