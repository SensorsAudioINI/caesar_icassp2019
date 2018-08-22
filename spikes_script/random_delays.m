function [ch0, ch1, ang1, ang2] = random_delays(y1, y2, fs)

d = 0.2;  % distance in m
c = 342.;  % speed of sound in m/s
radius = 2.;  % radius of semicircle in m

mics = [[-d / 2., 0.]; [d / 2., 0.]];

degs = [0:10:180] * pi / 180.;  % degs in rad

%% random choice 1
ang1 = degs(randi(length(degs)));
% ang = ang;
pos = [radius * cos(ang1), radius * sin(ang1)];

s_diff = norm(pos - mics(1, :)) - norm(pos - mics(2, :));
t_diff = s_diff / c;

% now delay in terms of n_samples 
samples_diff = round(t_diff * fs);

if samples_diff > 0
    y11 = y1(1+samples_diff:end);
    y12 = y1(1:end - samples_diff);
else
    y11 = y1(1:end + samples_diff);
    y12 = y1(1-samples_diff:end);
end

%% random choice 2
ang2 = degs(randi(length(degs)));
% ang = ang * pi / 180.
pos = [radius * cos(ang2), radius * sin(ang2)];

s_diff = norm(pos - mics(1, :)) - norm(pos - mics(2, :));
t_diff = s_diff / c;

% now delay in terms of n_samples 
samples_diff = round(t_diff * fs);

if samples_diff > 0
    y21 = y2(1 + samples_diff:end);
    y22 = y2(1:end - samples_diff);
else
    y21 = y2(1:end + samples_diff);
    y22 = y2(1 - samples_diff:end);
end

%% output
max_len = max([length(y11), length(y21), length(y12), length(y22)]);
y11 = [y11, zeros(1, max_len - length(y11))];
y12 = [y12, zeros(1, max_len - length(y12))];
y21 = [y21, zeros(1, max_len - length(y21))];
y22 = [y22, zeros(1, max_len - length(y22))];
ch0 = y11 + y21;
ch1 = y12 + y22;

end

