function array3D = cell_3D(Cell)
[~, k] = size(Cell);
[n, m] = size(Cell{1});
array3D = zeros(k, n, m);
for i = 1:k
    array3D(i, :, :) = permute(Cell{i}, [1, 2, 3]);
end
end