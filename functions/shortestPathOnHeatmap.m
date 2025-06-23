function pathRC = shortestPathOnHeatmap(heatmap, occluderMask, start_rc, end_rc)
% shortestPathOnHeatmap  Finds a minimal-cost path on a grid using heatmap
%   pathRC = shortestPathOnHeatmap(heatmap, occluderMask, start_rc, end_rc)

    [H, W] = size(heatmap);

    % Check that start/end are in bounds
    if any(start_rc < 1) || start_rc(1) > H || start_rc(2) > W
        error('start_rc out of bounds');
    end
    if any(end_rc < 1) || end_rc(1) > H || end_rc(2) > W
        error('end_rc out of bounds');
    end

    % Ensure start and end are within the occluder mask
    if ~occluderMask(start_rc(1), start_rc(2))
        [start_rc, valid] = snapToValid(start_rc, occluderMask);
        if ~valid
            error('start_rc not in occluderMask and no nearby valid pixel found');
        end
    end
    if ~occluderMask(end_rc(1), end_rc(2))
        [end_rc, valid] = snapToValid(end_rc, occluderMask);
        if ~valid
            error('end_rc not in occluderMask and no nearby valid pixel found');
        end
    end

    fprintf('Using start_rc = [%d,%d], end_rc = [%d,%d]\n', start_rc(1), start_rc(2), end_rc(1), end_rc(2));

    % Convert to linear index
    start_idx = sub2ind([H, W], start_rc(1), start_rc(2));
    end_idx   = sub2ind([H, W], end_rc(1), end_rc(2));

    neighbors_8 = [-1 -1; -1 0; -1 1; 0 -1; 0 1; 1 -1; 1 0; 1 1];
    infVal = 1e12;
    dist = inf(H*W, 1);
    dist(start_idx) = 0;
    visited = false(H*W,1);
    predecessor = zeros(H*W,1,'uint32');
    unvisitedSet = true(H*W,1);

    while true
        [minDist, current] = min(dist(unvisitedSet));
        if isinf(minDist)
            warning('No path found (Dijkstra exhausted).');
            pathRC = [];
            return;
        end
        unvisitedIndices = find(unvisitedSet);
        current_idx = unvisitedIndices(current);

        if current_idx == end_idx
            break;
        end

        visited(current_idx) = true;
        unvisitedSet(current_idx) = false;

        [curRow, curCol] = ind2sub([H, W], current_idx);
        curCost = dist(current_idx);

        for n = 1:size(neighbors_8, 1)
            nr = curRow + neighbors_8(n,1);
            nc = curCol + neighbors_8(n,2);
            if nr >= 1 && nr <= H && nc >= 1 && nc <= W
                neighbor_idx = sub2ind([H, W], nr, nc);
                if ~visited(neighbor_idx)
                    if occluderMask(curRow,curCol) && occluderMask(nr,nc)
                        costEdge = 0.5 * ((1 - heatmap(curRow, curCol)) + (1 - heatmap(nr, nc)));
                        altDist = curCost + costEdge;
                        if altDist < dist(neighbor_idx)
                            dist(neighbor_idx) = altDist;
                            predecessor(neighbor_idx) = uint32(current_idx);
                        end
                    end
                end
            end
        end
    end

    % Reconstruct path
    pathRC = [];
    idx = end_idx;
    while idx ~= 0
        pathRC = [idx; pathRC]; %#ok<AGROW>
        if idx == start_idx
            break;
        end
        idx = predecessor(idx);
    end

    % Convert to row,col
    pathRC = arrayfun(@(x) ind2sub2([H W], x), pathRC, 'UniformOutput', false);
    pathRC = cell2mat(pathRC);
end

function rc = ind2sub2(sz, idx)
    [r, c] = ind2sub(sz, idx);
    rc = [r, c];
end

function [snappedRC, valid] = snapToValid(rc, mask)
    [H, W] = size(mask);
    [y, x] = find(mask);
    if isempty(x)
        valid = false;
        snappedRC = rc;
        return;
    end
    dists = sqrt((y - rc(1)).^2 + (x - rc(2)).^2);
    [~, minIdx] = min(dists);
    snappedRC = [y(minIdx), x(minIdx)];
    valid = true;
end