function [ch0, ch1, ang] = delays4priors(y1, fs, ang)

d = 0.2;  % distance in m
c = 342.;  % speed of sound in m/s
radius = 2.;  % radius of semicircle in m

mics = [[-d / 2., 0.]; [d / 2., 0.]];

%% random choice 1
ang = ang * pi / 180.;
% ang = ang;
pos = [radius * cos(ang), radius * sin(ang)];

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



%% output
max_len = max([length(y11), length(y12), ]);
y11 = [y11; zeros(max_len - length(y11), 1)];
y12 = [y12; zeros(max_len - length(y12), 1)];
ch0 = y11;
ch1 = y12;

end

