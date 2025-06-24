function procrustesRef = simpleProcrustesReference(segment, p1, p2)
% Creates a quickly scaled/rotated version of 'segment' so that its first point
% aligns with p1 and its last point aligns with p2 (Procrustes-like transform).

    seg_shifted = segment - segment(1,:);
    seg_vec     = seg_shifted(end,:) - seg_shifted(1,:);
    tgt_vec     = p2 - p1;

    scale_factor = norm(tgt_vec) / (norm(seg_vec) + eps);
    angle = atan2(tgt_vec(2), tgt_vec(1)) - atan2(seg_vec(2), seg_vec(1));
    R = [cos(angle), -sin(angle); sin(angle), cos(angle)];

    X0 = (seg_shifted * R') * scale_factor + p1;
    X0(1,:)   = p1;
    X0(end,:) = p2;
    procrustesRef = X0;
end