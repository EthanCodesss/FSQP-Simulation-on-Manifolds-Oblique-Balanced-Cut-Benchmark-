function Figure_oblique_balancedcut(L)
% Generate IEEE-standard figures for oblique balanced cut optimization
% Creates publication-ready figures comparing FSQP, SQP, fmincon, and ALM
    fprintf('Generating IEEE-standard figures for oblique balanced cut optimization...\n');
    
    % Create figures directory if it doesn't exist
    if ~exist('oblique_figures', 'dir')
        mkdir('oblique_figures');
        endyi
    
    % Problem Setup
    N = 50;       % Number of nodes
    rankY = 2;    % Rank of the solution matrix
    
    % =======================================================
    % [Core Modification] Generate Common Initial Point
    % =======================================================
    % To ensure fairness, we define the manifold first to generate a valid initial point.
    % Note: x0 must satisfy Oblique constraints (unit column norms).
    factory = obliquefactory(rankY, N);
    
    % Set random seed for reproducibility (Paper Reproducibility)
    % rng(1000); 
    x0_common = factory.rand(); 
    
    fprintf('Common initial point generated.\n');
    % =======================================================
    
    % --- 1. Run FSQP ---
    fprintf('Running oblique balanced cut optimization with FSQP...\n');
    % [Mod] Pass x0_common
    result_fsqp = run_oblique_with_fsqp(L, N, rankY, x0_common); 
    
    % --- 2. Run SQP ---
    fprintf('Running oblique balanced cut optimization with SQP...\n');
    % [Mod] Pass x0_common
    result_sqp = run_oblique_with_sqp(L, N, rankY, x0_common);
    
    % --- 3. Run fmincon ---
    fprintf('Running oblique balanced cut optimization with fmincon...\n');
    % [Mod] Pass x0_common
    result_fmincon = run_oblique_with_fmincon(L, N, rankY, x0_common);
    
    % --- 4. Run ALM ---
    fprintf('Running oblique balanced cut optimization with ALM...\n');
    % [Mod] Pass x0_common
    result_alm = run_oblique_with_almbddmultiplier(L, N, rankY, x0_common);
    
    % --- Extract Data ---
    x_final_fsqp = result_fsqp.x;
    
    % =======================================================
    % [New] Export FSQP Results
    % =======================================================
    % 1. Save as CSV (for easy viewing in Excel)
    writematrix(x_final_fsqp, 'oblique_figures/FSQP_solution_matrix.csv');
    fprintf('FSQP solution matrix saved to oblique_figures/FSQP_solution_matrix.csv\n');
    
    % 2. Save as .mat file (contains full structure)
    save('oblique_figures/FSQP_full_result.mat', 'result_fsqp');
    
    % 3. Print brief verification info to console
    verify_result_quality(x_final_fsqp);
    % =======================================================
    
    % FSQP info extraction
    info_fsqp = result_fsqp.info;
    
    % ... (Data extraction and plotting for SQP, fmincon, ALM remain unchanged) ...
    info_sqp = result_sqp.info;
    x_final_sqp = result_sqp.x;
    info_fmincon = result_fmincon.info;
    x_final_fmincon = result_fmincon.x;
    info_alm = result_alm.info;
    x_final_alm = result_alm.x;
    
    % 1. Figure 1: Objective function convergence (with optional zoom)
    create_figure_objective_with_zoom(info_fsqp, info_sqp, info_fmincon, info_alm);
    
    % 2. Figure 2: KKT Residual Convergence (Log Scale)
    create_figure_kkt_logscale(info_fsqp, info_sqp, info_fmincon, info_alm);
    
    % Other figures
    % create_figure1_convergence(info_fsqp, info_sqp, info_fmincon, info_alm);
    create_figure2_solution_comparison(x_final_fsqp, x_final_sqp, x_final_fmincon, x_final_alm);
    create_figure3_constraint_satisfaction(x_final_fsqp, x_final_sqp, x_final_fmincon, x_final_alm);
    % create_figure4_performance_metrics(result_fsqp, result_sqp, result_fmincon, result_alm);
    
    % Call at the end of the main function:
    print_complexity_table(result_fsqp, result_sqp, result_fmincon, result_alm);
    fprintf('\nAll IEEE-standard figures saved to oblique_figures/ directory\n');
end

% -----------------------------------------------------------
% [New Function] Print Complexity Table (For Reviewer Response)
% -----------------------------------------------------------
function print_complexity_table(res_fsqp, res_sqp, res_fmincon, res_alm)
    fprintf('\n=================================================================================\n');
    fprintf('                      COMPUTATIONAL COMPLEXITY & TIME ANALYSIS\n');
    fprintf('=================================================================================\n');
    fprintf('%-10s | %-12s | %-12s | %-15s | %-15s\n', ...
            'Method', 'Total Iter', 'Total Time(s)', 'Avg Time/Iter(s)', 'Final Cost');
    fprintf('---------------------------------------------------------------------------------\n');
    
    % Define helper printing logic
    print_row('FSQP', res_fsqp);
    print_row('SQP', res_sqp);
    print_row('fmincon', res_fmincon);
    print_row('ALM', res_alm);
    fprintf('=================================================================================\n\n');
    
    function print_row(name, res)
        total_iter = length(res.info);
        % Note: Different algorithms record time differently. Here we take the time 
        % from the last info entry as the total time.
        if isfield(res.info(end), 'time')
            total_time = res.info(end).time;
        else
            total_time = res.time; % Defensive programming
        end
        
        avg_time = total_time / max(1, total_iter);
        final_cost = res.cost;
        
        fprintf('%-10s | %-12d | %-12.4f | %-15.4e | %-15.4e\n', ...
                name, total_iter, total_time, avg_time, final_cost);
    end
end

%% [New] Verification Helper Function
function verify_result_quality(Y)
    [r, n] = size(Y);
    fprintf('\n--- Verification of FSQP Result ---\n');
    fprintf('Matrix Size: %d x %d (Should be 2 x 50)\n', r, n);
    
    % 1. Check Oblique Constraint (Columns should be unit norm)
    col_norms = sqrt(sum(Y.^2, 1));
    max_col_err = max(abs(col_norms - 1));
    fprintf('Max Column Norm Error (Oblique): %.5e\n', max_col_err);
    if max_col_err < 1e-6
        fprintf('  => [PASS] Points are on the manifold.\n');
    else
        fprintf('  => [FAIL] Points are NOT on the manifold.\n');
    end
    
    % 2. Check Balanced Constraint (Rows should sum to zero)
    row_sums = sum(Y, 2);
    max_row_err = max(abs(row_sums));
    fprintf('Max Row Sum Error (Balanced):    %.5e\n', max_row_err);
    if max_row_err < 1e-3
        fprintf('  => [PASS] Solution is balanced.\n');
    else
        fprintf('  => [FAIL] Solution is NOT balanced (Constraint Violation).\n');
    end
    fprintf('-----------------------------------\n');
end

%% Helper function to run oblique optimization with FSQP
function result = run_oblique_with_fsqp(L, N, rankY, x0)
    % Problem parameters are now passed as arguments
    
    % Manifold setup - use Euclidean manifold for simplicity
    manifold = obliquefactory(rankY, N);
    problem.M = manifold;
    problem.cost = @(u) costFun(u, L);
    problem.egrad = @(u) gradFun(u, L);
    problem.ehess = @(u, d) hessFun(u, d, L);
    
    % Set up equality constraints
    colones = ones(N, 1);
    eq_constraints_cost = cell(2,1);
    eq_constraints_cost{1} = @(U) U(1,:) * colones;
    eq_constraints_cost{2} = @(U) U(2,:) * colones;
    
    eq_constraints_grad = cell(2,1);
    eq_constraints_grad{1} = @(U) [ones(1,N); zeros(1,N)];
    eq_constraints_grad{2} = @(U) [zeros(1,N); ones(1,N)];
    
    eq_constraints_hess = cell(2,1);
    eq_constraints_hess{1} = @(U, D) 0;
    eq_constraints_hess{2} = @(U, D) 0;
    
    problem.eq_constraint_cost = eq_constraints_cost;
    problem.eq_constraint_grad = eq_constraints_grad;
    problem.eq_constraint_hess = eq_constraints_hess;
    
    % Initial point
    % x0 = problem.M.rand();
    
    % Solver options
    options = struct();
    options.maxiter = 600;
    options.maxtime = 60;
    options.tolKKTres = 1e-6;
    options.modify_hessian = 'eye';
    options.verbosity = 2;
    options.qp_verbosity = 0;
    options.debug_force_FR = true;
    options.debug_FR_iter  = 3;
    
    % Solve with FSQP_simulation (Assumed to be the main solver function)
    [xsol, costfinal, residual, info, options_used] = FSQP_simulation(problem, x0, options);
    
    result = struct();
    result.x = xsol;
    result.cost = costfinal;
    result.residual = residual;
    result.info = info;
    result.options = options_used;
    result.L = L;
    result.rankY = rankY;
end

%% Helper function to run oblique optimization with SQP
function result = run_oblique_with_sqp(L, N, rankY,x0)
    % Problem parameters are now passed as arguments
    
    % Manifold setup - use Euclidean manifold for simplicity
    manifold = obliquefactory(rankY, N);
    problem.M = manifold;
    problem.cost = @(u) costFun(u, L);
    problem.egrad = @(u) gradFun(u, L);
    problem.ehess = @(u, d) hessFun(u, d, L);
    
    % Set up equality constraints
    colones = ones(N, 1);
    eq_constraints_cost = cell(2,1);
    eq_constraints_cost{1} = @(U) U(1,:) * colones;
    eq_constraints_cost{2} = @(U) U(2,:) * colones;
    
    eq_constraints_grad = cell(2,1);
    eq_constraints_grad{1} = @(U) [ones(1,N); zeros(1,N)];
    eq_constraints_grad{2} = @(U) [zeros(1,N); ones(1,N)];
    
    eq_constraints_hess = cell(2,1);
    eq_constraints_hess{1} = @(U, D) 0;
    eq_constraints_hess{2} = @(U, D) 0;
    
    problem.eq_constraint_cost = eq_constraints_cost;
    problem.eq_constraint_grad = eq_constraints_grad;
    problem.eq_constraint_hess = eq_constraints_hess;
    
    % Initial point
    % x0 = problem.M.rand();
    
    % Solver options
    options = struct();
    options.maxiter = 600;
    options.maxtime = 60;
    options.tolKKTres = 1e-6;
    options.modify_hessian = 'eye';
    options.verbosity = 2;
    options.qp_verbosity = 0;
    
    % Solve with SQP
    [xsol, costfinal, residual, info, options_used] = SQP(problem, x0, options);
    
    result = struct();
    result.x = xsol;
    result.cost = costfinal;
    result.residual = residual;
    result.info = info;
    result.options = options_used;
    result.L = L;
    result.rankY = rankY;
end

%% Helper function to run oblique optimization with fmincon
function result = run_oblique_with_fmincon(L, N, rankY,x0)
    fprintf('Running oblique balanced cut optimization with fmincon (SQP)...\n');
    
    % 1. Initial point setup
    % Generate random point and project to Oblique manifold (normalize columns)
    % x0 = randn(rankY, N);
    % x0 = x0 ./ sqrt(sum(x0.^2, 1));
    x0_vec = x0(:); % Flatten for fmincon
    
    % Shared variables for nested functions
    colones = ones(N, 1);
    startTime = tic();
    
    % 2. History storage for OutputFcn
    history.iter = [];
    history.cost = [];
    history.time = [];
    history.KKT_residual = [];
    last_kkt   = NaN;    % Initialize in parent scope -> Allow nested function writing
    
    % 3. Solver Options
    % Using 'sqp' algorithm to match the comparison context
    options = optimoptions('fmincon', ...
        'Algorithm', 'sqp', ...
        'MaxIterations', 1000, ...       % Match FSQP settings
        'MaxFunctionEvaluations', 1e5, ...
        'SpecifyObjectiveGradient', true, ...
        'SpecifyConstraintGradient', true, ...
        'OutputFcn', @outfun, ...
        'Display', 'iter-detailed', ... % Show progress
        'StepTolerance', 0, ...
        'ConstraintTolerance', 0, ...
        'OptimalityTolerance', 0);
        
    % 4. Run fmincon
    % Objective: costFun_fmincon
    % Constraints: nonlcon_fmincon (handles both Manifold + Problem constraints)
    [xsol_vec, fval, ~, output] = fmincon(@(v) costFun_fmincon(v, L, N, rankY), ...
                                          x0_vec, [], [], [], [], [], [], ...
                                          @(v) nonlcon_fmincon(v, N, rankY, colones), ...
                                          options);
    
    % 5. Package results to match FSQP/SQP structure
    result = struct();
    result.x = reshape(xsol_vec, [rankY, N]);
    result.cost = fval;
    result.residual = last_kkt;
    result.L = L;
    result.rankY = rankY;
    result.options = options;
    
    % Convert history arrays to struct array (same format as FSQP info)
    % This ensures compatibility if you want to pass it to the plotting functions later
    num_iters = length(history.iter);
    info_struct = struct('iter', cell(1, num_iters), ...
                         'cost', cell(1, num_iters), ...
                         'time', cell(1, num_iters), ...
                         'KKT_residual', cell(1, num_iters));
    
    for i = 1:num_iters
        info_struct(i).iter = history.iter(i);
        info_struct(i).cost = history.cost(i);
        info_struct(i).time = history.time(i);
        info_struct(i).KKT_residual = history.KKT_residual(i);
    end
    result.info = info_struct;
    
    %% Nested Helper Functions
    
    % Objective Function and Gradient
    function [f, g] = costFun_fmincon(v, L, N, rankY)
        Y = reshape(v, [rankY, N]);
        f = trace(Y * L * Y');
        
        if nargout > 1
            grad = 2 * Y * L; % Gradient for trace(YLY') is 2YL (assuming sym L)
            g = grad(:);
        end
    end
    
    % Non-linear Constraints (Manifold + Problem Constraints)
    function [c, ceq, gradc, gradceq] = nonlcon_fmincon(v, N, rankY, colones)
        Y = reshape(v, [rankY, N]);
        c = []; % No inequality constraints
        gradc = [];
        
        % --- Equality Constraints ---
        % 1. Oblique Manifold: diag(Y'Y) == 1 (N constraints)
        % 2. Balanced Cut: Y * ones == 0 (rankY constraints)
        ceq = zeros(N + rankY, 1);
        
        % Part 1: Oblique constraints (Columns must be unit norm)
        % Using Y(:,i)'*Y(:,i) - 1 instead of full diag calculation for speed
        ceq(1:N) = sum(Y.^2, 1)' - 1;
        
        % Part 2: Linear constraints (Rows sum to zero)
        ceq(N+1 : N+rankY) = Y * colones;
        
        if nargout > 2
            % Jacobian of Equality Constraints
            % Size needs to be (Number of Variables) x (Number of Constraints)
            % Variables = rankY * N
            % Constraints = N + rankY
            gradceq = zeros(rankY * N, N + rankY);
            
            % Gradients for Oblique constraints (columns unit norm)
            % Derivative of sum(Y.^2, 1) w.r.t Y is 2*Y
            for i = 1:N
                % The i-th constraint depends only on the i-th column of Y
                grad_col = zeros(rankY, N);
                grad_col(:, i) = 2 * Y(:, i);
                gradceq(:, i) = grad_col(:);
            end
            
            % Gradients for Balanced constraints (row sums)
            % Derivative of Y * ones w.r.t Y is ones matrix logic
            for k = 1:rankY
                % The (N+k)-th constraint is sum(Y(k,:)) = 0
                grad_row = zeros(rankY, N);
                grad_row(k, :) = 1;
                gradceq(:, N+k) = grad_row(:);
            end
        end
    end
    
    % Output Function to record history
    function stop = outfun(x, optimValues, state)
        stop = false;
        if isequal(state, 'iter') || isequal(state, 'interrupt')
            [~, current_ceq] = nonlcon_fmincon(x, N, rankY, colones);
            constraint_part_sq = sum(current_ceq.^2);
            current_kkt = sqrt(optimValues.firstorderopt^2 + constraint_part_sq);
            last_kkt = current_kkt;   % Record the last KKT residual
            history.iter  = [history.iter;  optimValues.iteration];
            history.cost  = [history.cost;  optimValues.fval];
            history.time  = [history.time;  toc(startTime)];
            history.KKT_residual = [history.KKT_residual; current_kkt];
            
            % Stopping condition corresponding to reference code
            if toc(startTime) > 10
                fprintf("Time limit exceeded\n");
                stop = true;
            elseif current_kkt <= 1e-6
                fprintf("KKT residual tolerance reached\n");
                stop = true;
            end
        end
    end
end

%% Helper function to run oblique optimization with ALM (Augmented Lagrangian Method)
function result = run_oblique_with_almbddmultiplier(L, N, rankY,x0)
    fprintf('Running oblique balanced cut optimization with ALM...\n');
    % 1. Manifold and Problem Setup
    % To ensure fair comparison with FSQP/SQP/fmincon in your current script,
    % we use Euclidean manifold and treat Oblique constraints explicitly.
    manifold = obliquefactory(rankY, N);
    problem.M = manifold;
    problem.cost = @(u) costFun(u, L);
    problem.egrad = @(u) gradFun(u, L);
    problem.ehess = @(u, d) hessFun(u, d, L);
    
    % 2. Set up Equality Constraints
    % (Identical setup to your FSQP/SQP functions)
    colones = ones(N, 1);
    
    eq_constraints_cost = cell(2,1);
    eq_constraints_cost{1} = @(U) U(1,:) * colones;      % Row sum constraint 1
    eq_constraints_cost{2} = @(U) U(2,:) * colones;      % Row sum constraint 2
    
    eq_constraints_grad = cell(2,1);
    eq_constraints_grad{1} = @(U) [ones(1,N); zeros(1,N)];
    eq_constraints_grad{2} = @(U) [zeros(1,N); ones(1,N)];
    
    eq_constraints_hess = cell(2,1);
    eq_constraints_hess{1} = @(U, D) 0;
    eq_constraints_hess{2} = @(U, D) 0;
    
    problem.eq_constraint_cost = eq_constraints_cost;
    problem.eq_constraint_grad = eq_constraints_grad;
    problem.eq_constraint_hess = eq_constraints_hess;
    
    % Note: ALM typically requires inequality constraints to be defined even if empty
    problem.ineq_constraint_cost = {};
    problem.ineq_constraint_grad = {};
    problem.ineq_constraint_hess = {};
    
    % 3. Initial Point
    % Generate a random point on the manifold
    % x0 = problem.M.rand();
    
    % 4. Solver Options
    options = struct();
    options.maxiter = 600;       % Match other solvers
    options.maxtime = 60;
    options.tolKKTres = 1e-6;    % Stop when KKT residual is low
    options.verbosity = 1;       % 1 = Standard output
    
    % 5. Solve with ALM
    % Based on your reference code, the signature is:
    % [x, info, residual] = almbddmultiplier(problem, x0, options);
    [xsol, info, residual] = almbddmultiplier(problem, x0, options);
    
    % 6. Calculate Final Cost (if not directly returned)
    costfinal = problem.cost(xsol);
    
    % 7. Package Results
    result = struct();
    result.x = xsol;
    result.cost = costfinal;
    result.residual = residual; % Final KKT residual
    result.info = info;         % Iteration history
    result.options = options;
    result.L = L;
    result.rankY = rankY;
end

%% Figure 1: Convergence Analysis
% (Function body removed for brevity, assuming it's the same or similar to create_figure1_convergence)
% Note: The script below defines create_figure1_convergence. If you want a combined one, see create_figure_objective_with_zoom.

%% Figure 1: Convergence Analysis (4 Methods)
function create_figure1_convergence(info_fsqp, info_sqp, info_fmincon, info_alm)
    figure('Position', [100, 100, 900, 600]);
    set(gcf, 'Color', 'white', 'PaperPositionMode', 'auto');
    
    % Prepare Data
    iter_fsqp = [info_fsqp.iter]; cost_fsqp = [info_fsqp.cost]; resid_fsqp = [info_fsqp.KKT_residual];
    iter_sqp = [info_sqp.iter];   cost_sqp = [info_sqp.cost];   resid_sqp = [info_sqp.KKT_residual];
    iter_fmin = [info_fmincon.iter]; cost_fmin = [info_fmincon.cost]; resid_fmin = [info_fmincon.KKT_residual];
    iter_alm = [info_alm.iter];   cost_alm = [info_alm.cost];   resid_alm = [info_alm.KKT_residual];
    % Colors
    col_fsqp = [0, 0.4470, 0.7410];      % Blue
    col_sqp  = [0.4660, 0.6740, 0.1880]; % Green
    col_fmin = [0.8500, 0.3250, 0.0980]; % Orange
    col_alm  = [0.4940, 0.1840, 0.5560]; % Purple
    
    % --- Left Axis: Objective Function (Log Scale) ---
    yyaxis left
    h1 = plot(iter_fsqp, cost_fsqp, '-', 'Color', col_fsqp, 'LineWidth', 2.5, 'DisplayName', 'FSQP Obj'); hold on;
    h2 = plot(iter_sqp, cost_sqp, '-', 'Color', col_sqp, 'LineWidth', 2.5, 'DisplayName', 'SQP Obj');
    h3 = plot(iter_fmin, cost_fmin, '-', 'Color', col_fmin, 'LineWidth', 2.5, 'DisplayName', 'fmincon Obj');
    h4 = plot(iter_alm, cost_alm, '-', 'Color', col_alm, 'LineWidth', 2.5, 'DisplayName', 'ALM Obj');
    
    ylabel('Objective Function Value (log scale)', 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'YColor', 'k', 'YScale', 'log'); % Left axis log scale
    
    % --- Right Axis: KKT Residual (Log Scale) ---
    yyaxis right
    h5 = plot(iter_fsqp, resid_fsqp, '--', 'Color', col_fsqp, 'LineWidth', 1.5, 'DisplayName', 'FSQP KKT'); hold on;
    h6 = plot(iter_sqp, resid_sqp, '--', 'Color', col_sqp, 'LineWidth', 1.5, 'DisplayName', 'SQP KKT');
    h7 = plot(iter_fmin, resid_fmin, '--', 'Color', col_fmin, 'LineWidth', 1.5, 'DisplayName', 'fmincon KKT');
    h8 = plot(iter_alm, resid_alm, '--', 'Color', col_alm, 'LineWidth', 1.5, 'DisplayName', 'ALM KKT');
    
    ylabel('KKT Residual (log scale)', 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'YScale', 'log', 'YColor', 'k');
    
    % Formatting
    xlabel('Iteration', 'FontSize', 14, 'FontWeight', 'bold');
    title('Convergence: FSQP vs SQP vs fmincon vs ALM', 'FontSize', 16, 'FontWeight', 'bold');
    
    % Legend
    legend([h1, h2, h3, h4, h5, h6, h7, h8], ...
           'Location', 'best', 'NumColumns', 2, 'FontSize', 10);
           
    grid on; grid minor;
    set(gca, 'GridAlpha', 0.3, 'MinorGridAlpha', 0.1);
    set(gca, 'LineWidth', 1.2, 'TickDir', 'in');
    
    % Save
    print('oblique_figures/fig1_convergence_analysis', '-depsc', '-r300');
    print('oblique_figures/fig1_convergence_analysis', '-dpng', '-r300');
    savefig('oblique_figures/fig1_convergence_analysis.fig');
    fprintf('✓ Figure 1: Convergence analysis saved\n');
end

%% Figure 2: Solution Comparison
function create_figure2_solution_comparison(x_fsqp, x_sqp, x_fmincon, x_alm)
    figure('Position', [100, 100, 1000, 800]); 
    set(gcf, 'Color', 'white', 'PaperPositionMode', 'auto');
    
    % Common settings
    all_vals = [x_fsqp(:); x_sqp(:); x_fmincon(:); x_alm(:)];
    clims = [min(all_vals), max(all_vals)];
    
    % Subplot 1: FSQP
    subplot(2,2,1);
    imagesc(x_fsqp, clims); colorbar;
    title('FSQP Solution', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Dimension'); 
    xlabel('Node Index'); 
    % [Mod] Force Y-axis to show only 1 and 2, removing ticks like 0.5/2.5
    set(gca, 'FontSize', 11, 'LineWidth', 1, 'TickDir', 'in', 'YTick', [1 2]);
    
    % Subplot 2: SQP
    subplot(2,2,2);
    imagesc(x_sqp, clims); colorbar;
    title('SQP Solution', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Dimension');
    xlabel('Node Index'); 
    % [Mod]
    set(gca, 'FontSize', 11, 'LineWidth', 1, 'TickDir', 'in', 'YTick', [1 2]);
    
    % Subplot 3: fmincon
    subplot(2,2,3);
    imagesc(x_fmincon, clims); colorbar;
    title('fmincon Solution', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Dimension');
    xlabel('Node Index'); 
    % [Mod]
    set(gca, 'FontSize', 11, 'LineWidth', 1, 'TickDir', 'in', 'YTick', [1 2]);
    
    % Subplot 4: ALM
    subplot(2,2,4);
    imagesc(x_alm, clims); colorbar;
    title('ALM Solution', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Dimension');
    xlabel('Node Index'); 
    % [Mod]
    set(gca, 'FontSize', 11, 'LineWidth', 1, 'TickDir', 'in', 'YTick', [1 2]);
    
    print('oblique_figures/fig2_solution_comparison', '-depsc', '-r300');
    print('oblique_figures/fig2_solution_comparison', '-dpng', '-r300');
    savefig('oblique_figures/fig2_solution_comparison.fig');
    fprintf('✓ Figure 2: Solution comparison saved\n');
end

%% Figure 3: Constraint Satisfaction Analysis
function create_figure3_constraint_satisfaction(x_fsqp, x_sqp, x_fmincon, x_alm)
    figure('Position', [100, 100, 1200, 500]);
    set(gcf, 'Color', 'white', 'PaperPositionMode', 'auto');
    
    N = size(x_fsqp, 2);
    colones = ones(N, 1);
    
    % --- Calculate Violations ---
    % 1. Oblique (||u||^2 - 1)
    obl_viol_fsqp = max(abs(diag(x_fsqp' * x_fsqp) - 1));
    obl_viol_sqp  = max(abs(diag(x_sqp' * x_sqp) - 1));
    obl_viol_fmin = max(abs(diag(x_fmincon' * x_fmincon) - 1));
    obl_viol_alm  = max(abs(diag(x_alm' * x_alm) - 1));
    
    % 2. Equality (row sums)
    eq_viol_fsqp = max(sum(abs(x_fsqp * colones)));
    eq_viol_sqp  = max(sum(abs(x_sqp * colones)));
    eq_viol_fmin = max(sum(abs(x_fmincon * colones)));
    eq_viol_alm  = max(sum(abs(x_alm * colones)));
    
    % Colors
    c1 = [0.3, 0.5, 0.8]; % Blueish base
    c2 = [0.2, 0.7, 0.3]; % Greenish base
    col_fmin = [0.8500, 0.3250, 0.0980]; % Orange
    col_alm  = [0.4940, 0.1840, 0.5560]; % Purple
    
    labels = {'FSQP', 'SQP', 'fmincon', 'ALM'};
    
    % --- Subplot 1: Oblique Manifold Constraints ---
    subplot(1,2,1);
    bar_data_obl = [obl_viol_fsqp, obl_viol_sqp, obl_viol_fmin, obl_viol_alm];
    b1 = bar(bar_data_obl, 'FaceColor', c1, 'EdgeColor', 'none', 'LineWidth', 1.5);
    
    % Custom Colors
    b1.FaceColor = 'flat';
    b1.CData(3,:) = col_fmin; 
    b1.CData(4,:) = col_alm;
    
    title('Oblique Manifold Violation', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Max Violation (log scale)', 'FontSize', 12);
    set(gca, 'XTickLabel', labels, 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'YScale', 'log', 'LineWidth', 1, 'TickDir', 'in');
    grid on; grid minor;
    ylim([1e-16, 10]); 
    hold on; yline(1e-6, 'r--', 'Tolerance', 'LineWidth', 2);
    
    % --- Subplot 2: Equality Constraints ---
    subplot(1,2,2);
    bar_data_eq = [eq_viol_fsqp, eq_viol_sqp, eq_viol_fmin, eq_viol_alm];
    b2 = bar(bar_data_eq, 'FaceColor', c2, 'EdgeColor', 'none', 'LineWidth', 1.5);
    
    % Custom Colors
    b2.FaceColor = 'flat';
    b2.CData(3,:) = col_fmin;
    b2.CData(4,:) = col_alm;
    
    title('Equality Constraint Violation', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Max Violation (log scale)', 'FontSize', 12);
    set(gca, 'XTickLabel', labels, 'FontSize', 12, 'FontWeight', 'bold');
    set(gca, 'YScale', 'log', 'LineWidth', 1, 'TickDir', 'in');
    grid on; grid minor;
    ylim([1e-16, 10]);
    hold on; yline(1e-6, 'r--', 'Tolerance', 'LineWidth', 2);
    
    print('oblique_figures/fig3_constraint_analysis', '-depsc', '-r300');
    print('oblique_figures/fig3_constraint_analysis', '-dpng', '-r300');
    savefig('oblique_figures/fig3_constraint_analysis.fig');
    fprintf('✓ Figure 3: Constraint satisfaction analysis saved\n');
end

%% Figure 4: Performance Metrics
function create_figure4_performance_metrics(res_fsqp, res_sqp, res_fmincon, res_alm)
    figure('Position', [100, 100, 1400, 450]); % Wider for 4 bars
    set(gcf, 'Color', 'white', 'PaperPositionMode', 'auto');
    
    % --- Extract Scalar Metrics ---
    % Iterations
    iter_data = [length(res_fsqp.info), length(res_sqp.info), length(res_fmincon.info), length(res_alm.info)];
    
    % Time
    time_data = [res_fsqp.info(end).time, res_sqp.info(end).time, res_fmincon.info(end).time, res_alm.info(end).time];
    
    % Final Cost
    cost_data = [res_fsqp.cost, res_sqp.cost, res_fmincon.cost, res_alm.cost];
    
    % Final Residual
    resid_data = [res_fsqp.residual, res_sqp.residual, res_fmincon.residual, res_alm.residual];
    
    labels = {'FSQP', 'SQP', 'fmincon', 'ALM'};
    colors = [0, 0.4470, 0.7410;      % Blue
              0.4660, 0.6740, 0.1880; % Green
              0.8500, 0.3250, 0.0980; % Orange
              0.4940, 0.1840, 0.5560];% Purple
              
    % --- Subplot 1: Efficiency ---
    subplot(1,3,1);
    b = bar([iter_data; time_data]', 'grouped');
    ylabel('Value', 'FontSize', 12);
    title('Efficiency: Iterations & Time', 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'XTickLabel', labels, 'FontSize', 11, 'FontWeight', 'bold');
    legend({'Iterations', 'Time (s)'}, 'Location', 'northwest');
    grid on; set(gca, 'TickDir', 'in');
    
    % --- Subplot 2: Final Cost ---
    subplot(1,3,2);
    min_cost = min(cost_data);
    max_cost = max(cost_data);
    range = max_cost - min_cost;
    if range < 1e-6 
        y_lims = [min_cost - 0.1*abs(min_cost), max_cost + 0.1*abs(max_cost)];
    else
        y_lims = [min_cost - 0.5*range, max_cost + 0.5*range];
    end
    
    b2 = bar(cost_data);
    b2.FaceColor = 'flat';
    b2.CData = colors;
    
    ylim(y_lims);
    title('Final Objective Cost', 'FontSize', 14, 'FontWeight', 'bold');
    set(gca, 'XTickLabel', labels, 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    
    % --- Subplot 3: Final Residual ---
    subplot(1,3,3);
    b3 = bar(resid_data);
    b3.FaceColor = 'flat';
    b3.CData = colors;
    
    set(gca, 'YScale', 'log');
    title('Final KKT Residual', 'FontSize', 14, 'FontWeight', 'bold');
    ylabel('Residual (log scale)', 'FontSize', 12);
    set(gca, 'XTickLabel', labels, 'FontSize', 11, 'FontWeight', 'bold');
    grid on;
    
    print('oblique_figures/fig4_performance_metrics', '-depsc', '-r300');
    print('oblique_figures/fig4_performance_metrics', '-dpng', '-r300');
    savefig('oblique_figures/fig4_performance_metrics.fig');
    fprintf('✓ Figure 4: Performance metrics saved\n');
end

%% [New Figure 1] Objective Function with Inset Zoom
function create_figure_objective_with_zoom(info_fsqp, info_sqp, info_fmincon, info_alm)
    % Create wide figure to fit IEEE double or single column
    hFig = figure('Position', [100, 100, 800, 600]);
    set(gcf, 'Color', 'white', 'PaperPositionMode', 'auto');
    
    % Data Extraction
    iter_fsqp = [info_fsqp.iter]; cost_fsqp = [info_fsqp.cost];
    iter_sqp = [info_sqp.iter];   cost_sqp = [info_sqp.cost];
    iter_fmin = [info_fmincon.iter]; cost_fmin = [info_fmincon.cost];
    iter_alm = [info_alm.iter];   cost_alm = [info_alm.cost];
    
    % Unify colors and line styles
    col_fsqp = [0, 0.4470, 0.7410];      % Blue
    col_sqp  = [0.4660, 0.6740, 0.1880]; % Green
    col_fmin = [0.8500, 0.3250, 0.0980]; % Orange
    col_alm  = [0.4940, 0.1840, 0.5560]; % Purple
    
    lw = 2.0; % LineWidth
    
    % ==========================
    % 1. Draw Main Plot
    % ==========================
    ax1 = axes('Position', [0.12 0.12 0.85 0.85]); % Standard margins
    hold(ax1, 'on');
    % [Mod] Explicitly turn Box on to ensure borders on all sides
    set(ax1, 'Box', 'on');
    p1 = plot(ax1, iter_fsqp, cost_fsqp, '-', 'Color', col_fsqp, 'LineWidth', lw, 'DisplayName', 'FRSQP');
    p2 = plot(ax1, iter_sqp, cost_sqp, '--', 'Color', col_sqp, 'LineWidth', lw, 'DisplayName', 'RSQP');
    p3 = plot(ax1, iter_fmin, cost_fmin, '-.', 'Color', col_fmin, 'LineWidth', lw, 'DisplayName', 'ESQP');
    p4 = plot(ax1, iter_alm, cost_alm, ':', 'Color', col_alm, 'LineWidth', lw, 'DisplayName', 'ALM');
    
    % Beautify Main Plot
    grid(ax1, 'on'); set(ax1, 'GridAlpha', 0.15);
    xlabel(ax1, 'Iteration', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel(ax1, 'Objective Function Value', 'FontSize', 12, 'FontWeight', 'bold');
    title(ax1, 'Convergence of Objective Function', 'FontSize', 14, 'FontWeight', 'bold');
    legend(ax1, [p1, p2, p3, p4], 'Location', 'northeast', 'FontSize', 11, 'Box', 'off');
    set(ax1, 'FontSize', 11, 'TickDir', 'in', 'LineWidth', 1.2);
    xlim(ax1, [0, max([iter_fsqp(end), iter_sqp(end), iter_fmin(end), iter_alm(end)])]);
    
    % ==========================
    % 2. Draw Inset Zoom (Optional / Commented Out)
    % ==========================
    % Define position of inset [left bottom width height] (normalized 0-1)
    % Usually placed in empty space or center-right
    % ax2 = axes('Position', [0.45 0.35 0.35 0.35]); 
    % box(ax2, 'on'); hold(ax2, 'on');
    % 
    % Redraw data in inset
    % plot(ax2, iter_fsqp, cost_fsqp, '-', 'Color', col_fsqp, 'LineWidth', 1.5);
    % plot(ax2, iter_sqp, cost_sqp, '--', 'Color', col_sqp, 'LineWidth', 1.5);
    % plot(ax2, iter_fmin, cost_fmin, '-.', 'Color', col_fmin, 'LineWidth', 1.5);
    % plot(ax2, iter_alm, cost_alm, ':', 'Color', col_alm, 'LineWidth', 1.5);
    % 
    % Set Zoom X-axis range (e.g., last 50 iterations)
    % max_iter = max([iter_fsqp(end), iter_sqp(end), iter_fmin(end), iter_alm(end)]);
    % zoom_start = max(0, max_iter - 50); 
    % xlim(ax2, [zoom_start, max_iter]);
    % 
    % Set Zoom Y-axis range (Automatically find min/max in this range and expand slightly)
    % final_costs = [cost_fsqp(end), cost_sqp(end), cost_fmin(end), cost_alm(end)];
    % y_center = min(final_costs);
    % y_range = max(final_costs) - min(final_costs);
    % if y_range < 1e-6, y_range = 1e-4; end % Prevent range being too small
    % ylim(ax2, [y_center - 0.5*y_range, y_center + 1.5*y_range]);
    % 
    % set(ax2, 'Color', [0.97 0.97 0.97]); % Slightly gray background to distinguish
    % title(ax2, 'Zoom: Final Iterations', 'FontSize', 9);
    % grid(ax2, 'on');
    
    % Save
    print('oblique_figures/fig1_objective_convergence', '-depsc', '-r300');
    print('oblique_figures/fig1_objective_convergence', '-dpng', '-r300');
    fprintf('✓ Figure 1: Objective convergence (with zoom) saved\n');
end

%% [New Figure 2] KKT Residual Convergence (Log Scale)
function create_figure_kkt_logscale(info_fsqp, info_sqp, info_fmincon, info_alm)
    hFig = figure('Position', [150, 150, 800, 600]);
    set(gcf, 'Color', 'white', 'PaperPositionMode', 'auto');
    
    % Data Extraction
    iter_fsqp = [info_fsqp.iter]; resid_fsqp = [info_fsqp.KKT_residual];
    iter_sqp = [info_sqp.iter];   resid_sqp = [info_sqp.KKT_residual];
    iter_fmin = [info_fmincon.iter]; resid_fmin = [info_fmincon.KKT_residual];
    iter_alm = [info_alm.iter];   resid_alm = [info_alm.KKT_residual];
    
    % Colors
    col_fsqp = [0, 0.4470, 0.7410];      % Blue
    col_sqp  = [0.4660, 0.6740, 0.1880]; % Green
    col_fmin = [0.8500, 0.3250, 0.0980]; % Orange
    col_alm  = [0.4940, 0.1840, 0.5560]; % Purple
    
    lw = 2.0;
    % Plot (Log Scale)
    semilogy(iter_fsqp, resid_fsqp, '-', 'Color', col_fsqp, 'LineWidth', lw, 'DisplayName', 'FRSQP'); hold on;
    semilogy(iter_sqp, resid_sqp, '--', 'Color', col_sqp, 'LineWidth', lw, 'DisplayName', 'RSQP');
    semilogy(iter_fmin, resid_fmin, '-.', 'Color', col_fmin, 'LineWidth', lw, 'DisplayName', 'ESQP');
    semilogy(iter_alm, resid_alm, ':', 'Color', col_alm, 'LineWidth', lw, 'DisplayName', 'ALM');
    
    % Beautify
    grid on; grid minor;
    set(gca, 'GridAlpha', 0.2, 'MinorGridAlpha', 0.1);
    
    xlabel('Iteration', 'FontSize', 12, 'FontWeight', 'bold');
    ylabel('KKT Residual (log scale)', 'FontSize', 12, 'FontWeight', 'bold');
    title('Convergence of KKT Residual', 'FontSize', 14, 'FontWeight', 'bold');
    
    % Reference Line (Tolerance 1e-6)
    % yline(1e-6, 'k--', 'Tolerance (1e-6)', 'LabelHorizontalAlignment', 'left', 'FontSize', 10);
    legend('Location', 'northeast', 'FontSize', 11, 'Box', 'off');
    set(gca, 'FontSize', 11, 'TickDir', 'in', 'LineWidth', 1.2);
    
    % Set Y-axis range to avoid log(0) or tiny values affecting display
    % Automatically find suitable lower bound, usually not less than 1e-16
    all_res = [resid_fsqp, resid_sqp, resid_fmin, resid_alm];
    min_val = max(1e-16, min(all_res(all_res>0)));
    ylim([min_val*0.1, 1e6]);
    
    % Save
    print('oblique_figures/fig2_kkt_convergence', '-depsc', '-r300');
    print('oblique_figures/fig2_kkt_convergence', '-dpng', '-r300');
    fprintf('✓ Figure 2: KKT residual convergence saved\n');
end

%% Helper Functions
function L = generate_random_laplacian(N, density)
    % Generate a random Laplacian matrix for a graph
    % Create adjacency matrix
    A = rand(N, N) < density;
    A = A + A';  % Make symmetric
    A = A - diag(diag(A));  % Remove diagonal
    A = A > 0;  % Make binary
    
    % Create degree matrix
    D = diag(sum(A, 2));
    
    % Create Laplacian matrix
    L = D - A;
end

function val = costFun(u, L)
    val = trace(u * L * u');
end

function val = gradFun(u, L)
    val = u * L + u * L';
end

function val = hessFun(u, d, L)
    val = d * L + d * L';
end