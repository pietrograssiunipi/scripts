%% PARAMETERS
% Define simulation range, state/action spaces, rates, and random model parameters.
N      = round(logspace(2,4,20));  
N      = unique(N);
S      = 3;                        
S      = S + 1;                    % Add a “no-transmission” absorbing state
A      = 5;                        
Rd     = 1.0;                      
Rr     = 0.2;                      
T      = 5.0;                      % Total simulation time horizon
alpha  = rand();                   
q      = sort([0; rand(A-2,1); 1])';
gamma  = 0.1*(1 - alpha*q(end))*rand(); 
p0S    = rand();                   
sigma2 = 1e-4*rand();              
C      = 4*rand();                 
beta   = 4*rand();                 
f       = phi(S, A, alpha, gamma, q, p0S);
% Sample one random deterministic action per state
actions = As(S, A);
% Enumerate all deterministic policies (columns of Udet)
Udet    = Ud(S, actions);
U       = size(Udet, 2);           
% Sample a product-form initial mean-field distribution m(s,u)
m       = mu0(S, U);
pS      = sum(m,2)';               % Marginal on states
pU      = sum(m,1);                % Marginal on policies
% Precompute the ODE generator for pure-state evolution: M(s,u→s′)
M       = matrixODE(S, U, Udet, f);

% Build initial state-action frequencies muSA0(s,a) = Σ_u m(s,u)·I[a=Udet(s,u)]
muSA0 = zeros(S, A);
for s = 1:S
  for uidx = 1:U
    a = Udet(s, uidx);
    muSA0(s,a) = muSA0(s,a) + m(s,uidx);
  end
end
% Initial payoff matrix Fmat(s,u) based on muSA0
Fmat = build_F(S, U, Udet, muSA0, f, actions, beta, q, C, sigma2);



%% DETERMINISTIC ODE SOLUTION
% Solve joint mean-field ODE for mu(t) over S×U
y0     = m(:);
odefun = @(t,y) odefun_joint(t, y, S, U, Rd, Rr, M, Udet, f, actions, beta, q, C, sigma2);
[tt, YY] = ode45(odefun, [0 T], y0);

% Interpolate onto fixed grid teval for comparison
teval     = linspace(0, T, 100);
Mu_interp = interp1(tt, YY, teval);  % 100×(S*U)



%% SUP-NORM ERROR AGAINST INDEPENDENT SIMULATIONS
R      = 5;    % Independent simulations                                 
supErr = zeros(length(N), R);

% Compute the state-only marginal from ODE: μ_S(t) = Σ_u μ(s,u;t)
Mu_det_state = zeros(length(teval), S);
for ti = 1:length(teval)
  Mfull               = reshape(Mu_interp(ti,:), S, U);
  Mu_det_state(ti,:)  = sum(Mfull, 2)';
end

% Precompute time-dipendent payoff F_interp(t,s,u)
F_interp = zeros(length(teval), S, U);
for ti = 1:length(teval)
  Mfull = reshape(Mu_interp(ti,:), S, U);
  muSA = zeros(S, A);
  for s_i = 1:S
    for u_i = 1:U
      a = Udet(s_i, u_i);
      muSA(s_i, a) = muSA(s_i, a) + Mfull(s_i, u_i);
    end
  end
  F_interp(ti,:,:) = build_F(S, U, Udet, muSA, f, actions, beta, q, C, sigma2);
end

for r = 1:R
  % Preallocate empirical distributions
  hatMu = zeros(length(N), S, length(teval));

  for ind = 1:length(N)
    n   = N(ind);
    % Sample n i.i.d. initial states/policies from m
    s0  = randsample(1:S, n, true, pS);
    u0  = randsample(1:U, n, true, pU);

    % Simulate each agent independently via exact jumps
    allState = zeros(n, length(teval));
    for i = 1:n
      % joint_evolution returns full path (sP,uP,tP) up to time T
      [sP,uP,tP] = joint_evolution_dynamic(S, U, s0(i), u0(i), T, f, Udet, Rd, Rr, Mu_interp, teval, F_interp );

      % For each evaluation time, pick the last state before that time
      for ti = 1:length(teval)
        idx              = find(tP <= teval(ti), 1, 'last');
        allState(i, ti)  = sP(idx);
      end
    end

    % Build empirical state-marginal hatMu(ind,s,ti)
    for s = 1:S
      hatMu(ind, s, :) = mean(allState == s, 1);
    end
  end

  % Compute sup-norm error vs ODE marginal at each N
  for ind = 1:length(N)
    P_emp         = squeeze(hatMu(ind, :, :))';  % 100×S
    P_det         = Mu_det_state;                % 100×S
    supErr(ind,r) = max(abs(P_emp - P_det), [], 'all');
  end
end

%% PLOT TRAJECTORIES OF THE SUP-NORM ERROR (log-log)
figure; hold on;
for r = 1:R
  loglog(N, supErr(:,r), '.-');        % individual replicate curves
end
meanSupErr = mean(supErr, 2);
hMean      = loglog(N, meanSupErr, '-k', 'LineWidth', 2, 'DisplayName', 'Mean error');

% Best-fit
p_fit = polyfit(log(N), log(meanSupErr), 1);  
p_exp = p_fit(1);          % exponent
cost    = exp(p_fit(2));      % coefficient
fprintf('Interpolating function: e_N ≈ %.3f · N^{%.2f}\n', cost, p_exp);
Nfit   = logspace(log10(min(N)), log10(max(N)), 200);
efit    = cost * Nfit.^p_exp;
hFit   = loglog(Nfit, efit, '--m', 'LineWidth', 1.5,'DisplayName', sprintf('%.2f·N^{%.2f}', cost, p_exp));

set(gca,'XScale','log','YScale','log');
xlabel('N'); ylabel('|\mu^N - \mu|_{\infty}');
title('Sup-Norm Error for Independent Simulations in Evolutionary Dynamics');
grid on;
legend([hMean, hFit],'Location','southwest');
hold off;

%% AUXILIARY FUNCTIONS

function Q = phi(S,A,alpha,gamma,q,p0S)
  % Construct Q(a,s→s'): for s>=2, 
  % stay with prob 1−αq(a)−γ, drop to s−1 with the complement;
  % for s=1, i.e., state 0, wrap back to S with prob p0S.
  Q = zeros(A, S, S);
  for a = 1:A
    val = 1 - alpha*q(a) - gamma;
    for s = 2:S
      Q(a, s, s)   = val;
      Q(a, s, s-1) = 1 - val;
    end
    Q(a,1,S) = p0S;
    Q(a,1,1) = 1 - p0S;
  end
end

function actions = As(S,A)
  % Sample one random action per state: generates a “prototype” mapping.
  actions = sort(randi([1 A], S, 1));
end

function Udet = Ud(S, actions)
  % Enumerate all deterministic policies of size S:
  % Each column of Udet specifies one action choice per state.
  radices = actions(:)';        % number of actions available per state
  K       = prod(radices);      % total policies
  Udet    = zeros(S, K);
  for idx = 0:K-1
    for s = 1:S
      div      = prod(radices(1:max(1,s-1)));
      digit    = floor(idx/div);
      Udet(s, idx+1) = mod(digit, radices(s)) + 1;
    end
  end
end

function m = mu0(S,U)
  % Sample an arbitrary product-form distribution m(s,u) = pS(s)*pU(u).
  pS = rand(1,S); pS = pS / sum(pS);
  pU = rand(1,U); pU = pU / sum(pU);
  m  = pS(:) * pU;
end

function M = matrixODE(S,U,Udet, f)
  % Build the S×U→S transition generator M(s,u→s'):
  % Each column (u fixed) integrates φ over the deterministic action
  M = zeros(S, U, S);
  for s = 1:S
    for u = 1:U
      upol = Udet(:,u);
      for sprime = 1:S
        % Sum φ(a, sprime → s) over the unique action a=upol(sprime)
        a = upol(sprime);
        M(s,u,sprime) = M(s,u,sprime) + f(a, sprime, s);
      end
    end
  end
  % Sanity check: each column of M must sum to 1
  cs = squeeze(sum(M,1));
  assert(all(abs(cs(:)-1)<1e-12), 'M is not column-stochastic!');
end

function F = build_F(S,U,Udet, muSA, f, ~, beta, q, C, sigma2)
  % Compute the value function F(s,u) under each fixed policy u:
  % 1) Build stage reward r(s,a) = q(a)/(σ² + C Σ q·w) − β q(a),
  %    where w(a) = Σ_s muSA(s,a).
  % 2) For each policy u, solve (I − βP_u) J = r_u for J_u(s).
  rSA = stage_reward(muSA, q, C, sigma2, beta);
  F   = zeros(S,U);
  I_S = eye(S);
  for u = 1:U
    [P_u, r_u] = Pu_ru_for_policy(Udet(:,u), f, rSA);
    F(:,u)     = (I_S - beta*P_u) \ r_u;  % linear solve for J_u
  end
end

function rh = rho_simple(F_s, x, Rr)
  % rho_simple: replicator‐style policy‐jump rates
  % Inputs:
  %     F_s  = 1×U vector of payoffs in state s
  %     x    = 1×U current population fraction over policies
  %     Rr   = maximum total outflow rate
  % Output:
  %     rh   = U×U matrix where rh(u,v) = rate of jump u→v

  U  = numel(F_s);
  rh = zeros(U,U);

  % 1) raw incentives weighted by current share x(v)
  for u = 1:U
    for v = 1:U
      if v ~= u
        diff = F_s(v) - F_s(u);
        if diff > 0
          rh(u,v) = x(v) * diff;
        end
      end
    end
  end

  % 2) normalize so that the maximum total outflow equals Rr
  rates = sum(rh, 2);     % total outflow rate from each u
  mrate = max(rates);
  if mrate > 0
    rh = (Rr / mrate) * rh;
  end
end

function rSA = stage_reward(muSA, q, C, sigma2, beta)
  % distribution over actions w(a) = sum_s muSA(s,a)
  wA   = sum(muSA, 1);  
  Iagg = sigma2 + C * sum(q .* wA);   % common denominator

  % immediate reward Rcol(a) = q(a)/Iagg  -  β·q(a)
  Rcol = q ./ Iagg - beta * q;        % 1×A

  % replicate for each state
  S    = size(muSA, 1);               
  rSA  = repmat(Rcol, S, 1);          % S×A
end

function [P_u, r_u] = Pu_ru_for_policy(u_col, f, rSA)
  % For a given deterministic policy u_col(s)=a:
  % Build transition matrix P_u(s→s') = φ(a,s→s') and reward vector r_u(s)=rSA(s,a).
  S   = numel(u_col);
  P_u = zeros(S,S);
  r_u = zeros(S,1);
  for s = 1:S
    a        = u_col(s);
    P_u(s,:) = squeeze(f(a, s, :))';  % row-vector of jump probs
    r_u(s)   = rSA(s, a);
  end
end

function dy = odefun_joint(~, y, S, U, Rd, Rr, M, Udet, f, actions, beta, q, C, sigma2)
  % Right-hand side of the joint ODE on μ(s,u):
  % 1) State drift: Rd [P_u μ_u − μ_u] for each policy u.
  % 2) Policy drift: replicator based on instantaneous F(s,u).
  mu = reshape(y, S, U);

  % (1) state evolution component
  f_d = zeros(S,U);
  for uidx = 1:U
    P_u        = squeeze(M(:, uidx, :));
    mu_u       = mu(:, uidx);
    f_d(:,uidx)= Rd * (P_u*mu_u - mu_u);
  end

  % (2) policy evolution component via replicator
  A    = size(f,1);
  muSA = zeros(S,A);
  for s = 1:S
    for uidx = 1:U
      a = Udet(s, uidx);
      muSA(s,a) = muSA(s,a) + mu(s, uidx);
    end
  end
  % Compute F(s,u) at current μ
  F    = build_F(S, U, Udet, muSA, f, actions, beta, q, C, sigma2);
  f_r  = zeros(S,U);
  for s = 1:S
    x    = mu(s,:);
    rhoM = rho_simple(F(s,:), x, Rr);
    inflow  = x * rhoM;
    outflow = sum(rhoM,2)';
    f_r(s,:) = inflow - x .* outflow;
  end

  dy = f_d + f_r;
  dy = dy(:);
end

function [sPath, uPath, tJump] = joint_evolution_dynamic(S, U, s0, u0, T, f, Udet, Rd, Rr, Mu_interp, teval, F_interp)

  % Preallocation
  maxJumps = ceil((Rd+Rr)*T*5);
  sPath    = zeros(1, maxJumps+1);
  uPath    = zeros(1, maxJumps+1);
  tJump    = zeros(1, maxJumps+1);

  % Inizialization
  idx = 1; t = 0; s = s0; up = u0;
  sPath(idx) = s;
  uPath(idx) = up;
  tJump(idx) = t;

  % Simulation jump-by-jump with F(t)
  while true
    dt_s = -log(rand)/Rd;
    % Payoff at current time t by interpolation of F_interp
    ti_corr = find(teval <= t, 1, 'last');
    Frow    = squeeze(F_interp(ti_corr, s, :))';  % 1×U

    % Policy‐jump rate using mean‐field μ(s,u;t)
    mu_full = reshape(Mu_interp(ti_corr,:), S, U);
    x_su    = mu_full(s, :);         % 1×U
    rh_mat  = rho_simple(Frow, x_su, Rr);
    totalPr = sum(rh_mat(up, :));    % outflow rate from current policy u       

    if totalPr>0
      dt_u = -log(rand) / totalPr;
    else
      dt_u = Inf;
    end

    if dt_s < dt_u
      % → jump in state
      t = t + dt_s;
      if t > T, break; end
      a   = Udet(s, up);
      pr  = squeeze(f(a, s, :))';
      s   = randsample(1:S, 1, true, pr);
    else
      % → replicator‐style policy jump
      t = t + dt_u;
      if t > T, break; end

      % rh_mat(u→v) holds the jump‐rates from current policy up
      % totalPr = sum(rh_mat(up,:)) as computed above
      probs = rh_mat(up, :) / totalPr;      % normalized probabilities
      up    = randsample(1:U, 1, true, probs);
    end

    % update
    idx = idx + 1;
    sPath(idx) = s;
    uPath(idx) = up;
    tJump(idx) = t;
  end

  % truncation
  sPath = sPath(1:idx);
  uPath = uPath(1:idx);
  tJump = tJump(1:idx);
end