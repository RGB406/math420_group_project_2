
[N, M, t_coords] = read_to_list('Project/Target3_coord.txt', 3);


figure
% Scatters the X and Y value of the Y matrix.
hold on
scatter(t_coords(:, 1), t_coords(:, 2), 'b*')
scatter(-8458.67, 2610.42, 'ro')
hold off
title('2D Plot of Target 3')
ylabel('Y')
xlabel('X')

% t1 = 10
% t2 = 33
% t3 = 11

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
