function anchors = makeAuditAnchors(d)

    n = size(d,1);

    anchors = repmat(struct( ...
        'anchorId', [], ...
        'phase', "", ...
        'turnType', "", ...
        'x', [], ...
        'g1', [], ...
        'g2Original', [], ...
        'previousControl', [], ...
        'goalSwitch', [], ...
        'nextVels', [], ...
        'history', []), n, 1);

    for i = 1:n

        anchors(i).anchorId = i;

        % Current state
        anchors(i).x = d(i,1:6)';

        % g1 = [x y theta vx vy omega dist angle]
        anchors(i).g1 = d(i,7:14)';

        % g2 = [x y theta vx vy omega dist angle]
        anchors(i).g2Original = d(i,15:22)';

        % Previous control
        anchors(i).previousControl = d(i,26:28)';

        % Goal switch
        anchors(i).goalSwitch = d(i,29);

        % Next velocity
        anchors(i).nextVels = d(i,30:32)';

        % History
        anchors(i).history = d(i,39:110);

        % Phase
        if anchors(i).goalSwitch == 0
            anchors(i).phase = "pre_switch";
        else
            anchors(i).phase = "post_switch";
        end

        anchors(i).turnType = "unknown";
    end
end
