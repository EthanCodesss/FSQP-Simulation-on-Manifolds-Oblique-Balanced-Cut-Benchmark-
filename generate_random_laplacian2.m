function L = generate_random_laplacian2()
    % Generate a random Laplacian matrix for a graph
    N = 50;
    density = 0.1;
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
