%% PARAMETERS
% Define the range of population sizes N for which we will estimate convergence; using logarithmically spaced values to cover several orders of magnitude.
N     = round(logspace(2,4,20));  
N     = unique(N);
S     = 3;
S     = S + 1;                  % Add a “no-transmission” absorbing state
A     = 5;

% Rate parameters: Rd for state transitions, T for total horizon
Rd    = 1.0;
T     = 5.0;

% Randomly sample parameters for transition kernel φ(a,s→s′)
q      = sort([0; rand(A-2,1); 1]);  
alpha  = rand();
gamma  = 0.1 * (1 - alpha * q(end)) * rand();
p0S    = rand();                     

% Build the transition kernel φ: an A×S×S tensor where φ(a,s→s′) gives jump probabilities.
f      = phi(S, A, alpha, gamma, q, p0S);

% Sample an arbitrary initial state distribution m(s)
m      = mu0(S);

% Sample one fixed stochastic policy u(s,a) for all agents to follow initially
actions = As(S, A);                  
u       = u_fix(S, A, actions);

% Build the generator matrix M for the deterministic ODE (mean‐field) from φ and the fixed policy u.
M       = matrixODE(S, f, u, actions);

%% DETERMINISTIC ODE + SIMULATION
R      = 5;                          % Number of independent repetitions
supErr = zeros(length(N), R);        % Preallocate sup-norm errors
teval  = linspace(0, T, 100);        % Time grid for evaluation

for ind = 1:length(N)
  n = N(ind);                        % Current population size

  % 1) Generate a distinct fixed policy u_i(s,a) for each of the n agents
  U_all = zeros(n, S, A);
  for i = 1:n
    u_i = rand(S, A);                % propose random weights
    for s = 1:S
      u_i(s, actions(s,1)+1:A) = 0;  % zero out invalid actions
      u_i(s,:) = u_i(s,:) / sum(u_i(s,:));  % normalize to a distribution
    end
    U_all(i,:,:) = u_i;              % store per-agent policy
  end

  % 2) Compute the average generator M_bar = (1/n) Σ_i Q^{u_i} where Q^{u_i}(s→s′) = Σ_a φ(a,s→s′) · u_i(s,a).
  M_bar = zeros(S, S);
  for i = 1:n
    u_i = squeeze(U_all(i,:,:));    % extract i-th policy
    for s_from = 1:S
      for a = 1:actions(s_from,1)
        % accumulate transition probabilities weighted by policy
        M_bar(:, s_from) = M_bar(:, s_from) + squeeze(f(a, s_from, :)) * u_i(s_from, a);
      end
    end
  end
  M_bar = M_bar / n;                 % average over all agents

  % 3) Solve the deterministic ODE dμ/dt = Rd*(M_bar*μ - μ) with initial μ(0)=m
  odefun    = @(t, mu) Rd * (M_bar * mu - mu);
  [tt, Mu]  = ode45(odefun, [0, T], m);
  Mu_interp = interp1(tt, Mu, teval);  % interpolate to teval grid (100×S)

  % 4) For each replicate, run n independent continuous-time chain simulations
  for r = 1:R
    allIndi = zeros(n, length(teval));  % stores each agent’s state path
    s0      = randsample(1:S, n, true, m);  % random initial states

    for i = 1:n
      % Simulate one agent’s state path under its fixed policy U_all(i,:,:)
      [si, jumpTimes] = state_evolution(S, A, s0(i), T, f, squeeze(U_all(i,:,:)), Rd);

      % For each evaluation time, record the last state before teval(ti)
      for ti = 1:length(teval)
        k = find(jumpTimes <= teval(ti), 1, 'last');
        allIndi(i, ti) = si(k);
      end
    end

    % 5) Compute empirical state-distribution hatMu(s,ti) = fraction of agents in state s at time ti
    hatMu = zeros(S, length(teval));
    for s = 1:S
      for ti = 1:length(teval)
        hatMu(s, ti) = mean(allIndi(:, ti) == s);
      end
    end

    % 6) Compute sup-norm error ||hatMu - Mu_interp||_∞ over all times and states
    supErr(ind, r) = max(abs(hatMu' - Mu_interp), [], 'all');
  end
end

%% PLOT ERROR TRAJECTORIES (log-log)
figure; hold on;
for r = 1:R
  loglog(N, supErr(:,r), '.-');      % individual replicates
end

meanSupErr = mean(supErr, 2);
hMean      = loglog(N, meanSupErr, '-k', 'LineWidth',2, 'DisplayName','Error mean');

% Fit e_N ≈ C·N^p on log-log scale
p        = polyfit(log(N), log(meanSupErr), 1);
p_exp    = p(1);                     % empirical convergence exponent
cost     = exp(p(2));                % prefactor
fprintf('Interpolating function: e_N ≈ %.3f · N^{%.2f}\n', cost, p_exp);

% Overlay best-fit line
Nfit = logspace(log10(min(N)), log10(max(N)), 200);
efit = cost * Nfit.^p_exp;
hFit = loglog(Nfit, efit, '--m', 'LineWidth',1.5, 'DisplayName',sprintf('%.2f·N^{%.2f}',cost,p_exp));

set(gca,'XScale','log','YScale','log');
xlabel('N');
ylabel('|\mu^N - \mu|_{\infty}');
title('Sup-Norm Error for Independent Simulations');
grid on;
legend([hMean, hFit],'Location','southwest');
hold off;



%% AUXILIARY FUNCTIONS

function Q = phi(S,A,alpha,gamma,q,p0S)
  % Build the per-action transition kernel φ(a,s→s').
  Q = zeros(A,S,S);
  for a = 1:A
    stayProb = 1 - alpha*q(a) - gamma;
    for s = 2:S
      Q(a,s,s)   = stayProb;
      Q(a,s,s-1) = 1 - stayProb;
    end
    % special wrap-around for state 1
    Q(a,1,S) = p0S;
    Q(a,1,1) = 1 - p0S;
  end
end

function actions = As(S,A)
  actions = sort(randi([1 A], S, 1));
end

function m = mu0(S)
  % Sample an arbitrary initial distribution m(s) over S states.
  m = rand(1, S);
  m = m / sum(m);
end

function u = u_fix(S,A,actions)
  u = rand(S,A);
  for s = 1:S
    u(s, actions(s,1)+1:A) = 0;   % disable actions beyond the chosen index
    u(s,:) = u(s,:) / sum(u(s,:)); % normalize to sum=1
  end
end

function M = matrixODE(S, f, u, actions)
  % Construct the S×S generator matrix for the deterministic ODE: M(s′, s) = Σ_a φ(a,s→s′) · u(s,a)
  M = zeros(S, S);
  for s = 1:S
    for a = 1:actions(s,1)
      M(:,s) = M(:,s) + squeeze(f(a, s, :)) * u(s, a);
    end
  end
  % Ensure each column sums to 1 (stochastic)
  assert(all(abs(sum(M,1)-1)<1e-12), 'M is not stochastic!');
end

function [si, jumpTimes] = state_evolution(S, A, s0, T, f, u, Rd)
  % • State s jumps at rate Rd,  
  % • Action in state s is drawn from the fixed policy u(s,:),  
  % • Next state sampled via φ(a,s→·).
  maxJumps = ceil(Rd * T * 2);
  si        = zeros(1, maxJumps+1);
  jumpTimes = zeros(1, maxJumps+1);

  idx = 1; t = 0; s = s0;
  si(idx)        = s;
  jumpTimes(idx) = t;

  while true
    dt = -log(rand) / Rd;
    t  = t + dt;
    if t > T, break; end

    % sample action according to u(s,:)
    a     = randsample(1:A, 1, true, u(s,:));
    probs = squeeze(f(a, s, :))';
    s     = randsample(1:S, 1, true, probs);

    idx = idx + 1;
    si(idx)        = s;
    jumpTimes(idx) = t;
  end

  si        = si(1:idx);
  jumpTimes = jumpTimes(1:idx);
end