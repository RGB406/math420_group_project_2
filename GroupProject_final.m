clearvars

for index=1:3
    [N, M, D, E, y] = prep_distance('Project\ObservedDataSet' + string(index) + '_dist.txt');
    G = NaN(N);
    epsilon = 0.1;
    while (isnan(G))
        epsilon = epsilon * 2
        G = perform_cvx(N, y, E, epsilon);
    end

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

%         figure
%         % Scatters the X, Y, and new Z of Y.
%         scatter3(Y(1, :), Y(2, :), Y(3, :), 'black*')
%         title('3D Plot of Embedding ' + string(index))
%         ylabel('Y')
%         xlabel('X')
%         zlabel('Z')

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


%                 figure
%         % Scatters the X, Y, and new Z of Y.
%         scatter3(final_coords(1, :), final_coords(2, :), final_coords(3, :), 'blueo')
%         hold on
%         scatter3(t_coords(1, :), t_coords(2, :), t_coords(3, :), 'red*')
%         hold off
%         title('3D Plot of Embedding ' + string(index) + ' to Target ' + string(i))
%         legend( 'Estimated', 'Target')
%         ylabel('Y')
%         xlabel('X')
%         zlabel('Z')


        a_e = norm((final_coords - t_coords), 'fro')^2;
        a_error(i) = a_e;

        if a_e == 10

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
            v = VideoWriter(['Project\Target', num2str(i), 'ComparedToObserved', num2str(index), '.mp4'], 'MPEG-4');
            open(v);

            figure('Renderer', 'painters', 'Position', [10 10 1200 900])
            view(2)
            for k = 0:130
                t = min(1, k/100);
                X_t = a(t)*Q(t, J)*(X - t*z_hat*ones([N, 1])');
                plot_graph(X_t, t_coords, N, index);
                frame = getframe(gcf);
                writeVideo(v, frame);
            end
            close(v);
        end

    end
    format long
    a_error
end

% % Below is testing partial reading
% [~, M, t_coords] = read_to_list('Project/Target2_coord.txt', 3);
% t_coords = t_coords';
% y_bar = (1/N) * t_coords * ones([N, 1]);
% Y_tilde = t_coords - y_bar * ones([1, N]);
% 
% error_list = zeros(10, 1);
% epsilon_list = zeros(10, 1);
% for index=1:10
%     [N, M, D, E, y] = prep_distance_partial('Project\ObservedDataSet3_dist.txt', 0.1*index);
%     G = NaN(N);
%     epsilon = 1;
%     while (isnan(G))
%         epsilon = epsilon * 2;
%         G = perform_cvx(N, y, E, epsilon);
%     end
%     epsilon_list(index) = epsilon
% 
%     % Calculating the eigenvalues of G
%     [Q, V] = eig(G);
%     % Get the order index from the sort method
%     [E_G, order] = sort(diag(V), 'descend');
%     % Re-sort the Q and V to descending order
%     Q = Q(:, order);
%     V = V(order, order);
% 
%     Q_j = Q(:, 1:3);
%     V_j = V(1:3, 1:3);
%     Y = V_j.^0.5 * Q_j';
% 
%     X = Y;
%     x_bar = (1/N) .* X * ones([N, 1]);
%     X_tilde = X - x_bar * ones([N, 1])';
% 
%     R_hat = X_tilde * transpose(Y_tilde);
%     [U, S, V] = svd(R_hat);
%     Q_hat = V * transpose(U);
% 
%     a_hat = trace(S)/(norm(X_tilde, 'fro')^2);
% 
%     z_hat = x_bar - (1/a_hat)*transpose(Q_hat)*y_bar;
% 
%     % Should be the updated coordinates
%     final_coords = a_hat*Q_hat*(X - z_hat*ones(1, N));
% 
%     error_list(index) = norm((final_coords - t_coords), 'fro')^2;
% end
% 
% figure
% plot([epsilon_list(1:10)], 'ro-')
% title('Epsilon Values for Partials of Data')
% xlabel('Tenths of data given')
% ylabel('Epsilon value')
% legend('Epsilon for tenths of values')
% 
% figure
% plot(error_list(1:10), 'bo-')
% title('Error Values for Partials of Data')
% xlabel('Tenths of data given')
% ylabel('Error value')
% legend('Error for tenths of values')


% Below is the CVX things.
function G = perform_cvx(N, y, E, epsilon)

m = size(y, 1);

cvx_begin sdp
variable G(N,N) semidefinite symmetric
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


% Initially assign D to zero to allocate space and to make it easy to tell
% when a point has no value given (0 distances!??)
function [N, M, D, E, y] = prep_distance_partial(filename, to_keep)
[N, M, arr] = read_to_arr(filename);
M = floor(M*to_keep);
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





function plot_graph(X, Y, N, index)
city_array = [10, 33, 11];
city_index = city_array(index);
city_X = X(:, city_index);
city_Y = Y(:, city_index);

scatter3(X(1, :), X(2, :), X(3, :), 'bo')
axis([-10000 5000 -10000 10000 -2 2])
view([45, 50])
hold on
scatter3(city_X(1), city_X(2), city_X(3), 'g*')
scatter3(city_Y(1), city_Y(2), 0, 'y*')
scatter3(Y(1, :), Y(2, :), zeros(N, 1), 'ro')
hold off
title('Target Coordinates versus Estimated Coordinates')
legend( 'Estimated', 'Estimated City', 'Target City', 'Target')

end
