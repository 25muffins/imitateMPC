clear;
clc;

load("GRUDiffV2.mat", "d");
anchors = makeAuditAnchors(d);
fprintf("Total available anchors: %d\n", numel(anchors));



numAuditAnchors = 1;

%idx = randperm(numel(anchors), numAuditAnchors);

%anchorsAudit = anchors(idx);
anchorsAudit = anchors;

resultTable = runFixedAnchorAudit( ...
    anchorsAudit, ...
    @solveTeacherAtFrozenAnchor, numAuditAnchors);

writetable(resultTable, "asdf.csv");