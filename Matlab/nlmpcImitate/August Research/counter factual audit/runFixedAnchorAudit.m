function resultTable = runFixedAnchorAudit(anchors, solveFcn, howMany)
% runFixedAnchorAudit
%
% solveFcn:
%   u = solveFcn(anchor, chosenG2Pos)
%
% chosenG2Pos is a 2x1 [x;y] position.

    n = howMany;

    % Preallocate
    anchor_id = zeros(n,1);
    phase = strings(n,1);
    turn_type = strings(n,1);

    base_vx = zeros(n,1);
    base_vy = zeros(n,1);
    base_w  = zeros(n,1);

    left_vx = zeros(n,1);
    left_vy = zeros(n,1);
    left_w  = zeros(n,1);

    right_vx = zeros(n,1);
    right_vy = zeros(n,1);
    right_w  = zeros(n,1);

    nonext_vx = zeros(n,1);
    nonext_vy = zeros(n,1);
    nonext_w  = zeros(n,1);
    
    rng(1232);
    idx = randperm(numel(anchors), n);

    for a = 1:length(idx)

        anchor = anchors(idx(a));
        if mod(idx(a), 19) == 0
            startnum = idx(a);
        else
            startnum = (floor(idx(a)/19)*19) + 1;
        end
        
        start = anchors(startnum);
        diff = idx(a) - startnum + 1;

        % =====================================================
        % Metadata
        % =====================================================

        anchor_id(a) = anchor.anchorId;
        phase(a) = string(anchor.phase);
        turn_type(a) = string(anchor.turnType);


        % =====================================================
        % Original g1 -> g2 geometry
        % =====================================================

        g1pos = anchor.g1(1:2);
        g2pos = anchor.g2Original(1:2);

        outgoing = g2pos - g1pos;

        if norm(outgoing) < 1e-9
            warning("Skipping anchor %d: zero-length segment.", ...
                anchor.anchorId);
            continue;
        end


        % =====================================================
        % BASE
        % =====================================================

        uHistoryBase = solveFcn(anchor, g2pos, start);
        
        uBase = uHistoryBase(diff, :);
        base_vx(a) = uBase(1);
        base_vy(a) = uBase(2);
        base_w(a)  = uBase(3);


        % =====================================================
        % LEFT (+45 degrees)
        % =====================================================

        Rleft = [ ...
            cosd(45), -sind(45);
            sind(45),  cosd(45)];

        g2Left = g1pos + Rleft * outgoing;

        uHistoryLeft = solveFcn(anchor, g2Left, start);
        uLeft = uHistoryLeft(diff, :);
        left_vx(a) = uLeft(1);
        left_vy(a) = uLeft(2);
        left_w(a)  = uLeft(3);


        % =====================================================
        % RIGHT (-45 degrees)
        % =====================================================

        Rright = [ ...
            cosd(-45), -sind(-45);
            sind(-45),  cosd(-45)];

        g2Right = g1pos + Rright * outgoing;

        uHistoryRight = solveFcn(anchor, g2Right, start);
        uRight = uHistoryRight(diff, :);
        right_vx(a) = uRight(1);
        right_vy(a) = uRight(2);
        right_w(a)  = uRight(3);


        % =====================================================
        % NO NEXT SEGMENT
        % =====================================================

        g2NoNext = g1pos;

        uNoNext = solveFcn(anchor, g2NoNext, start);

        nonext_vx(a) = uNoNext(1);
        nonext_vy(a) = uNoNext(2);
        nonext_w(a)  = uNoNext(3);


        % =====================================================
        % Progress
        % =====================================================

        if mod(a,10) == 0 || a == 1
            fprintf("Completed %d / %d anchors\n", a, n);
        end

    end


    % =========================================================
    % Construct table explicitly
    % =========================================================

    resultTable = table( ...
        anchor_id, ...
        phase, ...
        turn_type, ...
        base_vx, ...
        base_vy, ...
        base_w, ...
        left_vx, ...
        left_vy, ...
        left_w, ...
        right_vx, ...
        right_vy, ...
        right_w, ...
        nonext_vx, ...
        nonext_vy, ...
        nonext_w);


    % =========================================================
    % Explicit variable names
    % =========================================================

    resultTable.Properties.VariableNames = { ...
        'anchor_id', ...
        'phase', ...
        'turn_type', ...
        'base_vx', ...
        'base_vy', ...
        'base_w', ...
        'left_vx', ...
        'left_vy', ...
        'left_w', ...
        'right_vx', ...
        'right_vy', ...
        'right_w', ...
        'nonext_vx', ...
        'nonext_vy', ...
        'nonext_w'};


    % Sort by anchor ID
    resultTable = sortrows(resultTable, 'anchor_id');

end