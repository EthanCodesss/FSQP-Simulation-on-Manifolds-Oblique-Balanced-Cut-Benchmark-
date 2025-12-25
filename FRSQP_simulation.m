function [xfinal, costfinal, residual, info, options] = FSQP_simulation(problem0, x0, options)
    condet = constraintsdetail(problem0);
    
    % --- 1. Parameter Settings (Aligned with Paper) ---
    % Stopping criteria
    localdefaults.maxiter = 300;     % Paper: M
    localdefaults.maxtime = 3600;
    localdefaults.tolKKTres = 1e-8;  % Paper: epsilon_KKT
    
    % Hessian Regularization
    localdefaults.modify_hessian = 'mineigval_matlab';
    localdefaults.mineigval_correction = 1e-8; 
    localdefaults.mineigval_threshold = 1e-3;
    
    % Lagrangian Multipliers Initialization
    localdefaults.mus = ones(condet.n_ineq_constraint_cost, 1);
    localdefaults.lambdas = ones(condet.n_eq_constraint_cost, 1);    
    
    % Line Search Parameters (Theta_Filt)
    localdefaults.ls_max_steps  = 500; % Paper: l_max (conceptually)
    localdefaults.ls_threshold = 1e-8;
    localdefaults.tau = 0.9;   
    
    % Display & Verbosity
    localdefaults.verbosity = 1;
    localdefaults.qp_verbosity = 0;
    
    % Feasibility Restoration (FR)
    localdefaults.use_feas_restoration = true;
    localdefaults.fr_eps_direction     = 1e-6; % Paper: epsilon_dir
    
    % Merge Options
    localdefaults = mergeOptions(getGlobalDefaults(), localdefaults);
    if ~exist('options', 'var') || isempty(options)
        options = struct();
    end
    options = mergeOptions(localdefaults, options);
    
    % Ensure tau is set (if not passed in options, use 0.9 as per previous code logic)
    if ~isfield(options, 'tau')
        options.tau = 0.9; % Preserving code's preference from the loop
    end

    % Set Quadprog Options
    if options.qp_verbosity == 0
        qpoptions = optimset('Display','off');
    else
        qpoptions = [];
    end
    
    % --- 2. Initialization ---
    if ~exist('x0', 'var')|| isempty(x0)
        xCur = problem0.M.rand(); 
    else
        xCur = x0;
    end
    
    % Init Lagrangian Multipliers
    mus = options.mus;
    lambdas = options.lambdas;
    
    % Initial Stats
    iter = 0;
    xCurCost = getCost(problem0, xCur);
    xCurLagGrad = gradLagrangian(xCur, mus, lambdas);
    xCurLagGradNorm = problem0.M.norm(xCur, xCurLagGrad);
    xCurResidual = KKT_residual(xCur, mus, lambdas);
    [xCurMaxViolation, xCurMeanViolation] = const_evaluation(xCur);
    timetic = tic();
    
    % Stats logging
    stats = savestats();
    info(1) = stats;
    info(min(10000, options.maxiter+1)).iter = [];
    
    stop = false;
    totaltime = tic();
    
    % Filter Initialization
    filter = initializeGlobalConvergenceFilter(options);
    filter.initF = xCurCost;
    
    % --- 3. Main Loop ---
    while true
        % Display
        if options.verbosity >= 2
            fprintf('Iter: %d, Cost: %f, KKT residual: %.16e \n', iter, xCurCost, xCurResidual);
        elseif options.verbosity >= 1
            if mod(iter, 100) == 0 && iter ~= 0
                fprintf('Iter: %d, Cost: %f, KKT resiidual: %.16e \n', iter, xCurCost, xCurResidual);
            end
        end
        
        iter = iter + 1;
        timetic = tic();
        updateflag_Lagmult = false;
        
        % Subproblem Definition
        costLag = @(X) costLagrangian(X, mus, lambdas);
        gradLag = @(X) gradLagrangian(X, mus, lambdas); 
        hessLag = @(X, d) hessLagrangian(X, d, mus, lambdas); 
        auxproblem.M = problem0.M;
        auxproblem.cost = costLag;
        auxproblem.grad = gradLag;
        auxproblem.hess = hessLag;
        qpinfo = struct();
        
        % Hessian Modification
        if strcmp(options.modify_hessian, "eye")
            qpinfo.basis = tangentorthobasis(auxproblem.M, xCur, auxproblem.M.dim());
            qpinfo.n = numel(qpinfo.basis);
            qpinfo.H = eye(qpinfo.n);
        elseif strcmp(options.modify_hessian, 'mineigval_matlab') 
            [qpinfo.H, qpinfo.basis] = hessianmatrix(auxproblem, xCur);
            qpinfo.n = numel(qpinfo.basis);
            [U,T] = schur(qpinfo.H);
            for i = 1 : qpinfo.n
                if T(i,i) < 1e-5  
                    T(i,i) = options.mineigval_correction;
                end
            end
            qpinfo.H = U * T * U';
        elseif strcmp(options.modify_hessian, 'mineigval_manopt')
            [qpinfo.H, qpinfo.basis] = hessianmatrix(auxproblem, xCur);
            qpinfo.n = numel(qpinfo.basis);
            [~ ,qpinfo.mineigval] = hessianextreme(auxproblem, xCur);
            if qpinfo.mineigval < 0
                qpinfo.mineigval_diagcoeff = max(options.mineigval_threshold,...
                    abs(qpinfo.mineigval)) + options.mineigval_correction;
                qpinfo.H = qpinfo.H + qpinfo.mineigval_diagcoeff * eye(qpinfo.n);
            end
        else
            [qpinfo.H,qpinfo.basis] = hessianmatrix(auxproblem, xCur);
            qpinfo.n = numel(qpinfo.basis);
        end
        qpinfo.H = 0.5 * (qpinfo.H.'+qpinfo.H);
        
        % QP Gradient (f)
        f = zeros(qpinfo.n, 1);
        xCurGrad = getGradient(problem0, xCur);
        for fidx =1:qpinfo.n
            f(fidx) = problem0.M.inner(xCur, xCurGrad, qpinfo.basis{fidx});
        end
        qpinfo.f = f;
        
        % Inequality Constraints (A, b)
        if condet.has_ineq_cost
            row = condet.n_ineq_constraint_cost;
            col = qpinfo.n;
            A = zeros(row, col);
            b = zeros(row, 1);
            for ineqrow = 1:row
                ineqcosthandle = problem0.ineq_constraint_cost{ineqrow};
                b(ineqrow) = - ineqcosthandle(xCur);
                ineqgradhandle = problem0.ineq_constraint_grad{ineqrow};
                ineqconstraint_egrad = ineqgradhandle(xCur);
                ineqconstraint_grad = problem0.M.egrad2rgrad(xCur, ineqconstraint_egrad);
                for ineqcol = 1:col
                    base = qpinfo.basis{ineqcol};
                    A(ineqrow,ineqcol) = problem0.M.inner(xCur, ineqconstraint_grad, base);
                end
            end
        else
            A = [];
            b = [];
        end
        qpinfo.A = A;
        qpinfo.b = b;
        
        % Equality Constraints (Aeq, beq)
        if condet.has_eq_cost
            row = condet.n_eq_constraint_cost;
            col = qpinfo.n;
            Aeq = zeros(row, col);
            beq = zeros(row, 1);
            for eqrow = 1:row
                eqcosthandle = problem0.eq_constraint_cost{eqrow};
                beq(eqrow) = - eqcosthandle(xCur);
                eqgradhandle = problem0.eq_constraint_grad{eqrow};
                eqconstraint_egrad = eqgradhandle(xCur);
                eqconstraint_grad = problem0.M.egrad2rgrad(xCur, eqconstraint_egrad);
                for eqcol = 1:col
                    base = qpinfo.basis{eqcol};
                    Aeq(eqrow,eqcol) = problem0.M.inner(xCur, eqconstraint_grad, base);
                end
            end
        else
            Aeq = [];
            beq = [];
        end
        qpinfo.Aeq = Aeq;
        qpinfo.beq = beq;
        
        % --- 4. Solve QP ---
        [coeff, ~, qpexitflag, ~, Lagmultipliers] = quadprog(qpinfo.H, qpinfo.f,...
                qpinfo.A, qpinfo.b, qpinfo.Aeq, qpinfo.beq, [], [], [], qpoptions);     

        % --- 5. Feasibility Restoration (FR) Check ---
        if qpexitflag ~= 1 && options.use_feas_restoration
            if options.verbosity >= 1
                fprintf('Main QP failed, entering feasibility restoration QP...\n');
            end
            [x_FR, success_FR, vio_before, vio_after, qpexitflag_fr] = ...
                fr_qp_step(xCur, qpinfo, condet, options);
            
            if ~success_FR
                fprintf('Feasibility restoration QP failed (flag = %d, vio_before = %.2e, vio_after = %.2e).\n', ...
                        qpexitflag_fr, vio_before, vio_after);
                options.reason   = 'Main QP and feasibility restoration QP both failed';
                options.totaltime = toc(totaltime);
                break;
            end
            
            fprintf('FR-QP: violation %.2e -> %.2e\n', vio_before, vio_after);
            
            % Update State directly (Skip Line Search)
            newx = x_FR;
            
            % Distance calculation (Generic Riemannian)
            dist = problem0.M.dist(xCur, newx);

            stepsize = 1.0;
            ls_max_steps_flag = false;
            qpexitflag = qpexitflag_fr;
            
            % Update Filter with FR result
            filter.h = vio_after;
            filter.f = getCost(problem0, newx) / filter.initF;
            filter = UpdateFilter(filter, filter.h, filter.f);
            filter.PointAcceptedByFilter = true;
            filter.need_correction = false;
            
            % Reset multipliers after FR
            mus     = zeros(size(mus));
            lambdas = zeros(size(lambdas));
        
        else 
            % --- 6. Regular Step (Main QP Success) ---
            if qpexitflag ~= 1
                 % Fallback if FR is disabled and QP fails
                 options.reason = 'QP failed';
                 break;
            end

            deltaXast = problem0.M.zerovec(xCur);
            for i = 1:qpinfo.n
                deltaXast = problem0.M.lincomb(xCur, 1, deltaXast, coeff(i), qpinfo.basis{i});
            end
            
            % Line Search Setup
            f0 = problem0.cost(xCur);
            stepsize = 1;
            newx = problem0.M.retr(xCur, deltaXast, stepsize);
            newf = problem0.cost(newx);
            
            r = 0; % backtracking counter
            ls_max_steps_flag = false;
            options.ls_max_steps = 500;
            
            % Linear model m_k
            xCurGrad = getGradient(problem0, xCur);
            updatefilter = false;
            soc_done = false;
            
            % Cache original QP matrices for SOC
            A_orig = A; b_orig = b;
            Aeq_orig = Aeq; beq_orig = beq;
            original_deltaXast = deltaXast; 
            
            % --- 7. Line Search Loop ---
            while true
                if r > options.ls_max_steps
                    ls_max_steps_flag = true;
                    break;
                end
                
                newx = problem0.M.retr(xCur, deltaXast, stepsize);
                newf = problem0.cost(newx);
                
                % Evaluate Filter
                filter.h =  violation(problem0, newx, condet);
                filter.f =  getCost(problem0, newx)/ filter.initF;
                filter.need_correction = false;
                filter = EvaluateCurrentDesignPointToFilter(filter);
                
                % --- Second Order Correction (SOC) ---
                if ~filter.PointAcceptedByFilter && r == 0 && ~soc_done
                    if condet.has_ineq_cost
                        row = condet.n_ineq_constraint_cost;
                        b_ineq_sec = zeros(row, 1);
                        for ineqrow = 1:row
                            ineqcosthandle   = problem0.ineq_constraint_cost{ineqrow};
                            g_xCur           = ineqcosthandle(xCur);
                            g_newx           = ineqcosthandle(newx);
                            ineqgradhandle   = problem0.ineq_constraint_grad{ineqrow};
                            ineqconstraint_egrad = ineqgradhandle(xCur);
                            ineqconstraint_grad  = problem0.M.egrad2rgrad(xCur, ineqconstraint_egrad);
                            b_ineq_sec(ineqrow) = - ( g_newx - g_xCur ...
                                                      - problem0.M.inner(xCur, ineqconstraint_grad, deltaXast) );
                        end
                    else
                        b_ineq_sec = [];
                    end
                    
                    if condet.has_eq_cost
                        row = condet.n_eq_constraint_cost;
                        beq_sec = zeros(row, 1);
                        for eqrow = 1:row
                            eqcosthandle = problem0.eq_constraint_cost{eqrow};
                            eqgradhandle = problem0.eq_constraint_grad{eqrow};
                            eqconstraint_egrad = eqgradhandle(xCur);
                            eqconstraint_grad = problem0.M.egrad2rgrad(xCur, eqconstraint_egrad);
                            beq_sec(eqrow) = - (eqcosthandle(newx)- eqcosthandle(xCur) -problem0.M.inner(xCur, eqconstraint_grad, deltaXast));
                        end
                    else
                        beq_sec = [];
                    end
                    
                    % Re-solve QP with corrections
                    qpinfo.A   = A_orig;
                    qpinfo.b   = b_orig   + b_ineq_sec;
                    qpinfo.Aeq = Aeq_orig;
                    qpinfo.beq = beq_orig + beq_sec;
                    
                    [coeff, ~, qpexitflag_soc, ~, ~] = quadprog(qpinfo.H, qpinfo.f,...
                            qpinfo.A, qpinfo.b, qpinfo.Aeq, qpinfo.beq, [], [], [], qpoptions);     
                    
                    if qpexitflag_soc == 1
                        new_deltaXast = problem0.M.zerovec(xCur);
                        for i = 1:qpinfo.n
                            new_deltaXast = problem0.M.lincomb(xCur, 1, new_deltaXast, coeff(i), qpinfo.basis{i});
                        end
                        deltaXast = new_deltaXast; 
                        
                        % Retract SOC step
                        newx = problem0.M.retr(xCur, deltaXast, stepsize);
                        newf = problem0.cost(newx);
                        
                        % Re-evaluate Filter
                        filter.h =  violation(problem0, newx, condet);
                        filter.f =  getCost(problem0, newx)/ filter.initF;
                        filter = EvaluateCurrentDesignPointToFilter(filter);
                        
                        if ~filter.PointAcceptedByFilter
                            deltaXast = original_deltaXast; % Revert if SOC failed
                        end
                        if options.verbosity >= 2
                             fprintf('   -> SOC executed. New Filter accepted: %d\n', filter.PointAcceptedByFilter);
                        end
                    end
                    soc_done = true;
                end
                
                % --- Switching Strategy (Descent vs Feasibility) ---
                if (~filter.PointAcceptedByFilter)
                    % Rejected by filter -> Backtrack
                    r = r + 1;
                    stepsize = stepsize * options.tau; % Use tau for backtracking
                else
                    % Filter Accepted -> Check Descent Condition (Armijo)
                    step = problem0.M.lincomb(xCur, stepsize, deltaXast);
                    m_linear = - problem0.M.inner(xCur, xCurGrad, step); % m_linear > 0 means descent
                    
                    if m_linear > 0
                        % Descent Case: Check Armijo
                        % Paper: f(x+) <= f(x) + eta_f * m_k
                        % Code: m_linear * sigma > f0 - newf  => newf < f0 - sigma * m_linear
                        if m_linear * filter.eta_f > f0 - newf
                             % Armijo failed -> Backtrack
                             r = r + 1;
                             stepsize = stepsize * options.tau;
                        else
                             % Armijo passed -> Accept Step
                             break;
                        end
                    else
                        % Feasibility/Ascent Case:
                        % Just accept if filter accepted (already checked above)
                        updatefilter = true; % Add to filter for feasibility steps
                        break;
                    end
                end
            end % End Line Search
            
            if updatefilter
              filter = UpdateFilter(filter, filter.h, filter.f);
            end
            
            dist = problem0.M.dist(xCur, newx);
        end 
        
        % --- 8. Update Iterates ---
        xCur = newx;
        
        if ~(isequal(mus, Lagmultipliers.ineqlin)) ...
                || ~(isequal(lambdas, Lagmultipliers.eqlin))
            updateflag_Lagmult = true;
        end
        mus = Lagmultipliers.ineqlin;
        lambdas =  Lagmultipliers.eqlin;
        
        % Stats Update
        xCurCost = getCost(problem0, xCur);
        xCurLagGrad = gradLagrangian(xCur, mus, lambdas);
        xCurLagGradNorm = problem0.M.norm(xCur, xCurLagGrad);        
        xCurResidual = KKT_residual(xCur, mus, lambdas);
        [xCurMaxViolation, xCurMeanViolation] = const_evaluation(xCur);
        
        stats = savestats();
        info(iter+1) = stats;
        
        % --- 9. Check Stopping Criteria ---      
        if iter >= options.maxiter
            fprintf('Max iter count reached\n');
            options.reason = "Max iter count reached";
            stop = true;
        elseif toc(totaltime) >= options.maxtime
            fprintf('Max time exceeded\n');
            options.reason = "Max time exceeded";
            stop = true;
        elseif xCurResidual <= options.tolKKTres
            fprintf('KKT Residual tolerance reached\n');
            options.reason = "KKT Residual tolerance reached";
            stop = true;
        end
        
        if stop
            options.totaltime = toc(totaltime);
            break
        end
    end
    
    xfinal = xCur;
    residual  = xCurResidual;
    costfinal = problem0.cost(xfinal);
    
    % --- Nested Functions ---
    function stats = savestats()
        stats.iter = iter;
        stats.cost = xCurCost;
        stats.gradnorm = xCurLagGradNorm;
        if iter == 0
            stats.time = toc(timetic);
            stats.stepsize = NaN;
            stats.ls_max_steps_break = NaN;
            stats.dist =  NaN;
            stats.qpexitflag = NaN;
        else
            stats.time = info(iter).time + toc(timetic);
            stats.stepsize = stepsize;
            stats.ls_max_steps_break = ls_max_steps_flag;
            stats.dist = dist;
            stats.qpexitflag = qpexitflag;
        end
        stats.KKT_residual = xCurResidual;
        stats.maxviolation = xCurMaxViolation;
        stats.meanviolation = xCurMeanViolation;
        stats = applyStatsfun(problem0, xCur, [], [], options, stats);
    end

    function val = costLagrangian(x, mus, lambdas)
        val = getCost(problem0, x);
        if condet.has_ineq_cost
            for numineq = 1: condet.n_ineq_constraint_cost
                costhandle = problem0.ineq_constraint_cost{numineq};
                cost_numineq = costhandle(x);
                val = val + mus(numineq) * cost_numineq;
            end
        end
        if condet.has_eq_cost
            for numeq = 1: condet.n_eq_constraint_cost
                costhandle = problem0.eq_constraint_cost{numeq};
                cost_numeq = costhandle(x);
                val = val + lambdas(numeq) * cost_numeq;
            end
        end
    end

    function gradLag = gradLagrangian(x, mus, lambdas)
        gradLag = getGradient(problem0, x);
        if condet.has_ineq_cost
            for numineq = 1: condet.n_ineq_constraint_cost
                gradhandle = problem0.ineq_constraint_grad{numineq};
                constraint_grad = gradhandle(x);
                constraint_grad = problem0.M.egrad2rgrad(x, constraint_grad);
                gradLag = problem0.M.lincomb(x, 1, gradLag, mus(numineq), constraint_grad);
            end
        end
        if condet.has_eq_cost
            for numeq = 1:condet.n_eq_constraint_cost
                gradhandle = problem0.eq_constraint_grad{numeq};
                constraint_grad = gradhandle(x);
                constraint_grad = problem0.M.egrad2rgrad(x, constraint_grad);
                gradLag = problem0.M.lincomb(x, 1, gradLag, lambdas(numeq), constraint_grad);
            end
        end
    end

    function hessLag = hessLagrangian(x, dir, mus, lambdas)
        hessLag = getHessian(problem0, x, dir);
        if condet.has_ineq_cost
            for numineq = 1 : condet.n_ineq_constraint_cost
                gradhandle = problem0.ineq_constraint_grad{numineq};
                constraint_egrad = gradhandle(x);
                hesshandle = problem0.ineq_constraint_hess{numineq};
                constraint_ehess = hesshandle(x, dir);
                constraint_hess = problem0.M.ehess2rhess(x, constraint_egrad,...
                                                         constraint_ehess, dir);
                hessLag = problem0.M.lincomb(x, 1, hessLag,...
                    mus(numineq), constraint_hess);
            end
        end
        if condet.has_eq_cost
            for numeq = 1 : condet.n_eq_constraint_cost
                gradhandle = problem0.eq_constraint_grad{numeq};
                constraint_egrad = gradhandle(x);
                hesshandle = problem0.eq_constraint_hess{numeq};
                constraint_ehess = hesshandle(x, dir);
                constraint_hess = problem0.M.ehess2rhess(x, constraint_egrad,...
                                                        constraint_ehess, dir);
                hessLag = problem0.M.lincomb(x, 1, hessLag,...
                    lambdas(numeq), constraint_hess);
            end
        end
    end

    function [maxviolation, meanviolation] = const_evaluation(xCur)
        maxviolation = 0;
        meanviolation = 0;
        
        if condet.has_ineq_cost
            for numineq = 1: condet.n_ineq_constraint_cost
                costhandle = problem0.ineq_constraint_cost{numineq};
                cost_at_x = costhandle(xCur);
                maxviolation = max(maxviolation, cost_at_x);
                meanviolation = meanviolation + max(0, cost_at_x);
            end
        end
        if condet.has_eq_cost
            for numeq = 1: condet.n_eq_constraint_cost
                costhandle = problem0.eq_constraint_cost{numeq};
                cost_at_x = abs(costhandle(xCur));
                maxviolation = max(maxviolation, cost_at_x);
                meanviolation = meanviolation + cost_at_x;
            end
        end
        if condet.has_ineq_cost || condet.has_eq_cost
            meanviolation = meanviolation / (condet.n_ineq_constraint_cost + condet.n_eq_constraint_cost);
        end
    end
    
    function val = KKT_residual(xCur, mus, lambdas)
        xGrad = gradLagrangian(xCur, mus, lambdas);
        val = problem0.M.norm(xCur, xGrad)^2;
        
        manpowvio = manifoldPowerViolation(xCur);         
        compowvio = complementaryPowerViolation(xCur, mus);
        muspowvio = musposiPowerViolation(mus);
        
        val = val + manpowvio;
        val = val + compowvio;
        val = val + muspowvio;
        
        if condet.has_ineq_cost
            for numineq = 1: condet.n_ineq_constraint_cost
                costhandle = problem0.ineq_constraint_cost{numineq};
                cost_at_x = costhandle(xCur);
                violation = max(0, cost_at_x);
                val = val + violation^2;
            end
        end
        if condet.has_eq_cost
            for numeq = 1: condet.n_eq_constraint_cost
                costhandle = problem0.eq_constraint_cost{numeq};
                cost_at_x = abs(costhandle(xCur));
                val = val + cost_at_x^2;
            end
        end        
        val = sqrt(val);
    end

    function compowvio = complementaryPowerViolation(xCur, mus)
        compowvio = 0;
        if condet.has_ineq_cost
            for numineq = 1: condet.n_ineq_constraint_cost
                costhandle = problem0.ineq_constraint_cost{numineq};
                cost_at_x = costhandle(xCur);
                violation = mus(numineq) * cost_at_x;
                compowvio = compowvio + violation^2;
            end
        end
    end
    function musvio = musposiPowerViolation(mus)
        musvio = 0;
        if condet.has_ineq_cost
            for numineq = 1: condet.n_ineq_constraint_cost
                violation = max(-mus(numineq), 0);
                musvio = musvio + violation^2;
            end
        end
    end
    function manvio = manifoldPowerViolation(xCur)
        manvio = 0;
        if contains(problem0.M.name(),'Sphere')         
            y = xCur(:);
            manvio = abs(y.'*y - 1)^2;
        elseif contains(problem0.M.name(),'Oblique')
            [~,N] = size(xCur);
            for imani = 1:N
                manvio = manvio + abs(xCur(:,imani).' * xCur(:,imani) - 1)^2;
            end
        else 
            manvio = 0;
        end
    end
    
    function filter = initializeGlobalConvergenceFilter(options)
      % Parameters aligned with Paper (Beta_h, Beta_f, Eta_f)
      % Values preserved from original code as requested
      filter.SmallVal = 1.0e-5; 
      filter.beta_f = filter.SmallVal;       % Paper: beta_f (slope for objective)
      filter.beta_h = 1.0 - filter.SmallVal; % Paper: beta_h (margin for constraints)
      filter.eta_f = 0.2;                    % Paper: eta_f (Armijo factor)
      
      filter.delta = filter.SmallVal;
      filter.vals = zeros(options.maxiter,2); 
      filter.nVals = 1; 
      filter.PointAcceptedByFilter = false; 
      filter.h = 1.0e30; 
      filter.f = 1.0e30; 
      filter.initF = 0.0; 
      
      filter.vals(1,1) = inf;  
      filter.vals(1,2) = inf;  
    end

    function filter = EvaluateCurrentDesignPointToFilter(filter)
      h = filter.h;
      f = filter.f;
      for ii = 1:filter.nVals
        hi = filter.vals(ii,1);
        fi = filter.vals(ii,2);
        % Paper Logic: h <= beta_h * hi OR f <= fi - beta_f * h
        % Code Logic:  h <= hi * beta_h OR (f + beta_f * h) <= fi
        if ( (h <= hi*filter.beta_h) || (f+filter.beta_f*h) <= fi )
          filter.PointAcceptedByFilter = true;
        elseif ( (h > hi*filter.beta_h) && (f+filter.beta_f*h) > fi )
            filter.need_correction = true;
        else
          filter.PointAcceptedByFilter = false;
          break
        end 
      end 
    end

    function maxviolation = violation(problem, x, condet)
        maxviolation = 0;
        meanviolation = 0;
        cost = getCost(problem, x);

        for numineq = 1: condet.n_ineq_constraint_cost
            costhandle = problem.ineq_constraint_cost{numineq};
            cost_at_x = costhandle(x);
            maxviolation = max(maxviolation, cost_at_x);
            meanviolation = meanviolation + max(0, cost_at_x);
        end
        
        for numeq = 1: condet.n_eq_constraint_cost
            costhandle = problem.eq_constraint_cost{numeq};
            cost_at_x = abs(costhandle(x));
            maxviolation = max(maxviolation, cost_at_x);
            meanviolation = meanviolation + cost_at_x;
        end

        meanviolation = meanviolation / (condet.n_ineq_constraint_cost + condet.n_eq_constraint_cost);
    end
    function newFilter = UpdateFilter(filter, hk, fk)
      newFilter = filter;
      newFilter.nVals = 0;
      newFilter.vals(:,:) = 0;
      Update = true;
      ii = 0;
      if (filter.nVals >= 1)
        while (Update)
            ii = ii + 1;
            hi = filter.vals(ii,1);
            fi = filter.vals(ii,2);
          if ( (hk <=hi ) && (fk <= (fi)) )
             % Dominated, remove it
          else 
            newFilter.nVals = newFilter.nVals + 1;
            newFilter.vals(newFilter.nVals,1) = hi;
            newFilter.vals(newFilter.nVals,2) = fi;
          end
          if (ii >= filter.nVals)
            Update = false;
          end
        end 
      end
      newFilter.nVals = newFilter.nVals + 1;
      newFilter.vals(newFilter.nVals,1) = hk;
      newFilter.vals(newFilter.nVals,2) = fk;
   end


   function [x_new, success, vio_before, vio_after, qpexitflag_fr] = ...
            fr_qp_step(xCur_FR, qpinfo_FR, condet_FR, options_FR)
        M = problem0.M;
        success = false;
        x_new  = xCur_FR;
        qpexitflag_fr = -99;
        n     = qpinfo_FR.n;  
        if condet_FR.has_eq_cost
            meq   = condet_FR.n_eq_constraint_cost;
        else
            meq   = 0;
        end
        if condet_FR.has_ineq_cost
            mineq = condet_FR.n_ineq_constraint_cost;
        else
            mineq = 0;
        end
        if meq + mineq == 0
            vio_before = 0;
            vio_after  = 0;
            success    = true;
            return;
        end
        vio_before = violation(problem0, xCur_FR, condet_FR);
        eps_dir = options_FR.fr_eps_direction; 
        H_fr = blkdiag(eps_dir*eye(n), eye(meq), eye(mineq));  
        f_fr = zeros(n + meq + mineq, 1);
        A    = qpinfo_FR.A;
        b    = qpinfo_FR.b;
        Aeq  = qpinfo_FR.Aeq;
        beq  = qpinfo_FR.beq;
        if meq > 0
            Aeq_fr = [Aeq, -eye(meq), zeros(meq, mineq)];
            beq_fr = beq;
        else
            Aeq_fr = [];
            beq_fr = [];
        end
        if mineq > 0
            A_fr_1 = [A, zeros(mineq, meq), -eye(mineq)];
            b_fr_1 = b;
            A_fr_2 = [zeros(mineq, n+meq), -eye(mineq)];
            b_fr_2 = zeros(mineq, 1);
            A_fr   = [A_fr_1; A_fr_2];
            b_fr   = [b_fr_1; b_fr_2];
        else
            A_fr = [];
            b_fr = [];
        end
        lb = []; ub = [];
        [z, ~, qpexitflag_fr, ~] = quadprog(H_fr, f_fr, ...
                                            A_fr, b_fr, ...
                                            Aeq_fr, beq_fr, ...
                                            lb, ub, [], qpoptions);
        if qpexitflag_fr ~= 1
            vio_after = vio_before;
            return;
        end
        p = z(1:n);
        deltaX_fr = M.zerovec(xCur_FR);
        for i = 1:n
            deltaX_fr = M.lincomb(xCur_FR, 1, deltaX_fr, p(i), qpinfo_FR.basis{i});
        end
        x_trial = M.retr(xCur_FR, deltaX_fr);
        vio_after = violation(problem0, x_trial, condet_FR);
        if vio_after < vio_before
            success = true;
            x_new   = x_trial;
            if options_FR.verbosity >= 1
                fprintf('FR-QP success: violation %.2e -> %.2e\n', vio_before, vio_after);
            end
        else
            if options_FR.verbosity >= 1
                fprintf('FR-QP did not improve violation: %.2e -> %.2e\n', vio_before, vio_after);
            end
        end
   end

end