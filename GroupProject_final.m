clearvars
for index=1:3
    [N, M, D, E, y] = prep_distance('Project\ObservedDataSet' + string(index) + '_dist.txt');
    G = NaN(N);
    epsilon = 0.01;
    while (isnan(G))
        epsilon = epsilon * 2;
        G = perform_cvx(N, y, E, epsilon);
    end
    epsilon

    % Calculating the eigenvalues of G
    [Q, V] = eig(G);
    % Get the order index from the sort method
    [E_G, order] = sort(diag(V), 'descend');
    % Re-sort the Q and V to descending order
    Q = Q(:, order);
    V = V(order, order);

    Q_j = Q(:, 1:3);
    V_j = V(1:3, 1:3);
    Y = V_j.^0.5 * Q_j';


    % Now we do the transformation section of code
    % We'll run this in the for loop so we can just loop through the
    % targets for each observed dataset
    X = Y;
    x_bar = (1/N) .* X * ones([N, 1]);
    X_tilde = X - x_bar * ones([N, 1])';
    % Going to keep the error calculations just in case
    a_error = zeros(1, 3);

    for i=1:3

        [~, M, t_coords] = read_to_list('Project/Target' + string(i) + '_coord.txt', 3);
        % This is to transform the targets into row vectors rather than
        % columns.
        t_coords = t_coords';
        y_bar = (1/N) * t_coords * ones([N, 1]);
        Y_tilde = t_coords - y_bar * ones([1, N]);

        R_hat = X_tilde * transpose(Y_tilde);
        [U, S, V] = svd(R_hat);
        Q_hat = V * transpose(U);

        a_hat = trace(S)/(norm(X_tilde, 'fro')^2);

        z_hat = x_bar - (1/a_hat)*transpose(Q_hat)*y_bar;

        % Should be the updated coordinates
        final_coords = a_hat*Q_hat*(X - z_hat*ones(1, N));

        a_e = norm((final_coords - t_coords), 'fro')^2;
        a_error(i) = a_e;

        if a_e < 10

            Q_det = det(Q_hat);

            % Actual transformation functions here
            a = @(t) (1 - t + t*a_hat);
            Q = @(t, J) (J' * expm(t * logm(J * Q_hat)));
            z = @(t) (t * z_hat);

            Q_det

            J = eye(size(Q_hat, 1));
            if (Q_det < 0)
                J(1, 1) = -1;
            end
            v = VideoWriter(['Project\Target', num2str(i), 'ComparedToObserved', num2str(index), '.avi']);
            open(v);

            figure
            for k = 0:160
                t = min(1, k/100);
                X_t = a(t)*Q(t, J)*(X - t*z_hat*ones([N, 1])');
                plot_graph(X_t, t_coords, N);
                frame = getframe(gcf);
                writeVideo(v, frame);
            end
            close(v);
        end

    end
    format long
    a_error
end


% Below is the CVX things.
function G = perform_cvx(N, y, E, epsilon)

m = size(y, 1);

cvx_begin sdp
variable G(N,N) semidefinite symmetric;
minimize trace(G);
subject to

G*ones(N, 1) == 0;

abs(diag(E'*G*E) - y') <= epsilon * ones(m, 1);

cvx_end
end

% Below are the data and plot management functions

% Used for coordinate lists
function [N, M, arr] = read_to_list(filename, to_read)
T = readtable(filename);
arr = table2array(T(:, 1:to_read));
N = size(arr, 2);
M = size(arr, 1);
end

% Used for distance lists
function [N, M, arr] = read_to_arr(filename)
T = readtable(filename);
N = table2array(T(1, 1));
M = table2array(T(1, 2));
arr = table2array(T(2:M+1, :));
end

% Initially assign D to zero to allocate space and to make it easy to tell
% when a point has no value given (0 distances!??)
function [N, M, D, E, y] = prep_distance(filename)
[N, M, arr] = read_to_arr(filename);
D = zeros(N);
E = zeros(N, M);
y = zeros(1, M);
for index=1:M
    point = arr(index, 1:2);
    D(point(1), point(2)) = arr(index, 3);
    D(point(2), point(1)) = arr(index, 3);
    E(point(1), index) = 1;
    E(point(2), index) = -1;
    y(index) = arr(index, 3)^2;
end

end


function plot_graph(X, Y, N)

scatter3(X(1, :), X(2, :), X(3, :), 'bo')
axis([-10000 10000 -10000 10000 -1 1])
hold on
scatter3(Y(1, :), Y(2, :), zeros(N, 1), 'ro')
hold off
title('Target Coordinates versus Estimated Coordinates')
legend( 'Estimated', 'Target')

end
