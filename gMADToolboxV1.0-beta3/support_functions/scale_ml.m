function S = scale_ml(counts)
% Use cvx to compute maximum likelihood scale values
% assuming Thurstone's case V model.
%
% S* = argmax P(counts| S) P(S)
%        S
%    = argmin ?log P(counts| S)
%        S
%
% Assume that mean(S)=0.
%
% CVX can be obtained at http://cvxr.com/cvx/
%
% 2011?06?05 Kristi Tsukida <kristi.tsukida@gmail.com>

[m,mm] = size(counts);
assert(m == mm, 'counts must be a square matrix');

counts(eye(m)>0) = 0; % set diagonal to zero

previous_quiet = cvx_quiet(1);
cvx_begin
    variables S(m,1) t;
    SS = repmat(S,1,m);
    delta = SS -  SS'; % ? (i,j) = S(i) ?  S(j)

    minimize( t );
    subject to
    -sum(sum(counts.*log_normcdf( delta )))<= t
    sum(S)==0
cvx_end
cvx_quiet(previous_quiet);