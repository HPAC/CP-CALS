function out = cp_cals_hybrid(X, kt_components, jk_kt_components, varargin)

%  cp_cals_hybrid Compute CP decomposition and optionally also Jackknife.
%
%  initialize random CP models with kt_components number of components. 
%  Then, for the number of components specified in jk_kt_components, it
%  finds the model with the lowest error from the ones already fitted,
%  and for that model performs jackknife.
%
%  The differences with CP_ALS are:
%    - The input tensor must be a dense tensor
%    - The 'dimorder' parameter is not accepted
%
%  Usage:
%
%  out = cp_cals_hybrid(X,kt_components, jk_kt_components, varargin) 
%         performs jackknifing on each ktensor in init_ktensors.
%     'X' - The target tensor.
%	  'kt_components' - Array containing the number of components of 
%         (randomly initialized) Ktensors to be fitted to the target tensor 
%         using ALS. e.g. [2 2 2 2 3 3 3 3 4 4 4 4] to fit 4 randomly
%         initialized models of ranks 2, 3, and 4 each.
%	  'jk_kt_components' - Array containing the number of components for 
%         which to perform Jackknife (of the ones fitted). e.g. [3 4] to 
%         select the models of number of components 3 and 4 with the lowest
%         error and perform jackknife to them. 
%	The result out is a cell array containing (for every component number 
%   in kt_components): the (randomly initialized) ktensor, the fitted 
%   ktensor, (if selected for JK) the fitted and permutation adjusted
%   jackknife ktensors and the stdB, stdC matrices.
%
%
%  out = cp_cals_hybrid(X,kt_components, jk_kt_components, value, ...) 
%             specifies optional parameters and values.
%
%      Valid parameters and their default values are:
%        'tol' - Tolerance on difference in fit {1.0e-4}
%        'maxiters' - Maximum number of iterations {50}
%        'buffer-size' - Maximum size of the factor matrices buffer {4200}
%        'update-method' - Method for updating the factor matrices 
%                          {unconstrained} ['unconstrained', 'nnls']
%        'mttkrp-method' - method for computing MTTKRP {auto}
%                          ['mttkrp', 'twostep0', 'twostep1', 'auto']
%
%        'no-ls'(default)/'ls' - Whether to use line search.
%        'ls-interval' - Interval (per individual model) to apply line 
%                        search {5}
%        'ls-step' - Factor with which to jump in line search {1.2}
%        'no-cuda'(default)/'cuda' - Whether to use cuda (make sure the 
%                                    binary is compiled with cuda support)

tic;
out = cp_cals_hybrid_driver(X, kt_components, jk_kt_components, varargin{:});

mode2 = size(X, 2);
mode3 = size(X, 3);
for i=1:size(out, 1)
	if size(out{i}, 1) == 4
		comp = size(out{i}{2}.U{1}, 2);
		out{i}{4}{2} = std(reshape(double(out{i}{3}.U{2}), [mode2 comp mode2]), [], 3);
		out{i}{4}{3} = std(reshape(double(out{i}{3}.U{3}), [mode3 comp mode3]), [], 3);
	end
end
hybrid_time = toc;
fprintf('Time for CALS Hybrid: %f\n', hybrid_time);

