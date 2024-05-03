clear vars
[Nodes, M, D, E, y] = prep_distance('Project\ObservedDataSet3_dist.txt');
G = NaN(Nodes);
epsilon = 8;
while (isnan(G))
    epsilon = epsilon * 2;
    G = perform_cvx(Nodes, y, E, epsilon);
end

[N, M, t_coords] = read_to_list('Project/Target2_coord.txt', 3);
% figure
% % Scatters the X and Y value of the Y matrix.
% scatter(t_coords(:, 1), t_coords(:, 2), 'b*')
% title('2D Plot of Target')
% ylabel('Y')
% xlabel('X')


% Calculating the eigenvalues of G
[Q, V] = eig(G);
% Get the order index from the sort method
[E_G, order] = sort(diag(V), 'descend');
% Re-sort the Q and V to descending order
Q = Q(:, order);
V = V(order, order);

for j=[2 3]
    % Create a Q for the smaller dimensions
    Q_j = Q(:, 1:j);
    V_j = V(1:j, 1:j);

    % Compute the Y matrix for each dimension
    Y = V_j.^0.5 * Q_j';

end

n = 3
X = Y';
x_bar = ((1/n) .* X) * ones([n, 1]);
X_tilde = X - x_bar * ones([n, 1])';

y_bar = (1/n) .* t_coords * ones([n, 1]);
Y_tilde = t_coords - y_bar * ones([n, 1])';

R_hat = X_tilde * transpose(Y_tilde);
[U, S, V] = svd(R_hat);
Q_hat = V * transpose(U);

z_hat = x_bar - Q_hat'*y_bar;
a_hat = trace(S)/(norm(X_tilde, 'fro')^2);

final_coords = (X);
figure
hold on
scatter(final_coords(:, 1), final_coords(:, 2), 'r*')
%scatter(t_coords(:, 1), t_coords(:, 2), 'b*')
hold off
title('2D Plot of Final')
ylabel('Y')
xlabel('X')


function G = perform_cvx(N, y, E, eps)

m = size(y, 1);

cvx_begin sdp
variable G(N,N) semidefinite symmetric
minimize trace(G);
subject to

G*ones(N, 1) == 0;

abs(diag(E'*G*E) - y) <= eps * ones(m, 1);

cvx_end

end

function [N, M, D, E, y] = prep_distance(filename)
[N, M, arr] = read_to_arr(filename);
D = zeros(N);
E = zeros(N, M);
y = zeros(M, 1);
for index=1:M
    point = arr(index, 1:2);
    D(point(1), point(2)) = arr(index, 3);
    D(point(2), point(1)) = arr(index, 3);
    E(point(1), index) = 1;
    E(point(2), index) = -1;
    y(index) = arr(index, 3)^2;
end
end
% Used for distance lists
function [N, M, arr] = read_to_arr(filename)
T = readtable(filename);
N = table2array(T(1, 1));
M = table2array(T(1, 2));
arr = table2array(T(2:M+1, :));
end
% Used for coordinate lists
function [N, M, arr] = read_to_list(filename, to_read)
T = readtable(filename);
arr = table2array(T(:, 1:to_read));
N = size(arr, 2);
M = size(arr, 1);
end
