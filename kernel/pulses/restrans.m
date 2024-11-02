% RLC circuit response calculation - converts a waveform from the
% ideal shape emitted by the instrument into the shape that comes
% out of the RLC circuit of the probe. Syntax:
%
%          [X,Y,dt, J_X, J_Y]=restrans(X_user,Y_user,dt_user,...
%                            omega,Q,model,up_factor)
%
% Parameters:
%
%       X_user    - in-phase part of the rotating frame
%                   pulse waveform, a column vector of 
%                   real numbers
%
%       Y_user    - out-of-phase part of the rotating 
%                   frame pulse waveform, a column vec-
%                   tor of real numbers
%
%       dt_user   - time slice duration, seconds
%
%       omega     - RLC circuit resonance frequency in
%                   radians per second, a real number
%
%       Q         - RLC circuit quality factor, a real
%                   positive number
%
%       model     - input signal model, use 'pwc' for
%                   piecewise-constant, and 'pwl' for
%                   piecewise-linear input; time shift
%                   compensation for piecewise-linear
%                   is requested by 'pwl_tsc'
%
%       up_factor - the output waveform will have more
%                   discretisation points than the in-
%                   put waveform by this factor, about
%                   100 is a safe guess
%
% Outputs:
%
%       X        - in-phase part of the rotating frame
%                  pulse waveform distorted by the RLC
%                  response, a column vector of real 
%                  numbers
%
%       Y        - out-of-phase part of the rotating
%                  frame pulse waveform distorted by
%                  the RLC response, a column vector
%                  of real numbers
%
%       dt       - slice duration in the distorted wave-
%                  form, seconds
%
%       J_X      - Jacobian matrix of X with respect to 
%                  X_user and Y_user
%
%       J_Y      - Jacobian matrix of Y with respect to 
%                  X_user and Y_user
%
% u.rasulov@soton.ac.uk
% i.kuprov@soton.ac.uk
%
% <https://spindynamics.org/wiki/index.php?title=restrans.m>

function [X, Y,dt,Jacobian_X, Jacobian_Y] = restrans(X_user, Y_user, dt_user, omega, Q, model, up_factor)
    % Check consistency
    grumble(X_user, Y_user, dt_user, omega, Q);

    % 16 x Nyquist oversampling
    dt = pi / (16 * omega);

    % Signal model
    switch model
        case 'pwc'
            % Piecewise-constant
            nslices = numel(X_user); tmax = nslices * dt_user;
            user_time_grid = dt_user * (cumsum(ones(nslices, 1)) - 1/2);

            % Time grid required by the RLC circuit
            circuit_time_grid = linspace(0, tmax, tmax / dt + 1)';

            % Interpolate user waveform onto RLC circuit time grid
            X0 = interp1(user_time_grid, X_user, circuit_time_grid, 'nearest', 'extrap');
            Y0 = interp1(user_time_grid, Y_user, circuit_time_grid, 'nearest', 'extrap');

            if nargout >=4
                % Compute the Jacobian of the interpolation for X and Y
                Jacobian_X_interp = interp1_jacobian_nearest(user_time_grid, circuit_time_grid, numel(X_user));
                Jacobian_Y_interp = interp1_jacobian_nearest(user_time_grid, circuit_time_grid, numel(Y_user));
            end

        case {'pwl', 'pwl_tsc'}
            % Piecewise-linear
            nslices = numel(X_user) - 1; tmax = nslices * dt_user;
            user_time_grid = linspace(0, tmax, nslices + 1);

            % Time grid required by the RLC circuit
            circuit_time_grid = linspace(0, tmax, tmax / dt + 1)';

            % Interpolate user waveform onto RLC circuit time grid
            X0 = interp1(user_time_grid, X_user, circuit_time_grid, 'linear');
            Y0 = interp1(user_time_grid, Y_user, circuit_time_grid, 'linear');
            
            if nargout >=4
                % Compute the Jacobian of the interpolation for X and Y
                Jacobian_X_interp = interp1_jacobian_linear(user_time_grid, circuit_time_grid, numel(X_user));
                Jacobian_Y_interp = interp1_jacobian_linear(user_time_grid, circuit_time_grid, numel(Y_user));
            end
            otherwise
            % Complain and bomb out
            error('Unknown signal model.');
    end

    % Generate wall clock input signal
    inp_amp = sqrt(X0.^2 + Y0.^2);
    inp_phi = atan2(Y0, X0);
    inp_signal = inp_amp .* cos(omega * circuit_time_grid + inp_phi);

    % Build the RLC circuit response kernel
    sys = tf(1 / Q, [1/(omega^2), 1/(omega*Q), 1]);

    % Apply the RLC circuit response kernel
    [out_signal, ~, ~] = lsim(sys, inp_signal, circuit_time_grid);

    % Heterodyne out the carrier frequency
    X_out=2*lowpass(out_signal.*sin(omega*circuit_time_grid),...
                1,64,ImpulseResponse="iir",Steepness=0.95);
    Y_out=2*lowpass(out_signal.*cos(omega*circuit_time_grid),...
                1,64,ImpulseResponse="iir",Steepness=0.95);

    % Downsample the rotating frame waveform
    down_factor = numel(X_out) / (up_factor * numel(X_user));
    down_factor = floor(down_factor); 
    dt = dt * down_factor;
    X = X_out(1:down_factor:end); 
    Y = Y_out(1:down_factor:end);

    if nargout >= 4
        [Jacobian_X, Jacobian_Y] = calculate_jacobians(X0, Y0,Jacobian_X_interp,Jacobian_Y_interp, inp_amp, inp_phi, omega, sys, circuit_time_grid, down_factor);
    end
   

    % Downsample the time grid
    circuit_time_grid = circuit_time_grid(1:down_factor:end);

    % Diagnostic plotting if no outputs
    if nargout == 0
        plot(user_time_grid, X_user, '.', 'Color', [0.8500, 0.3250, 0.0980]); hold on;
        plot(user_time_grid, Y_user, '.', 'Color', [0.0000, 0.4470, 0.7410]);
        plot(circuit_time_grid, X0, '-', 'Color', [0.8500, 0.3250, 0.0980] + 1);
        plot(circuit_time_grid, Y0, '-', 'Color', [0.0000, 0.4470, 0.7410] + 1);
        plot(circuit_time_grid, X, '-', 'Color', [0.8500, 0.3250, 0.0980]);
        plot(circuit_time_grid, Y, '-', 'Color', [0.0000, 0.4470, 0.7410]);
        xlabel('time, seconds'); xlim('tight');
        ylabel('amplitude, rad/s'); grid on;
        legend({'X, user', 'Y, user', 'X, input', 'Y, input', 'X, output', 'Y, output'}, 'Location', 'south');
    end
end

% Consistency enforcement function (unchanged)
function grumble(X_user, Y_user, dt_user, omega, Q)
    if (~isnumeric(dt_user)) || (~isreal(dt_user)) || (~isscalar(dt_user)) || (dt_user <= 0)
        error('dt_user must be a real positive scalar.');
    end
    if (~isnumeric(X_user)) || (~isreal(X_user)) || (~iscolumn(X_user)) || ...
       (~isnumeric(Y_user)) || (~isreal(Y_user)) || (~iscolumn(Y_user))
        error('X_user and Y_user must be real column vectors.');
    end
    if (~isnumeric(omega)) || (~isreal(omega)) || (~isscalar(omega))
        error('omega must be a real scalar.');
    end
    if (~isnumeric(Q)) || (~isreal(Q)) || (~isscalar(Q)) || (Q <= 0)
        error('Q must be a positive real scalar.');
    end
    if dt_user < (pi / omega)
        error('dt_user breaks rotating frame approximation.');
    end
end

function Jacobian = interp1_jacobian_nearest(user_time_grid, circuit_time_grid, num_elements)
    % Calculate the Jacobian matrix for nearest interpolation
    Jacobian = zeros(numel(circuit_time_grid), num_elements);
    [~, indices] = min(abs(circuit_time_grid - user_time_grid'), [], 2);
    for i = 1:numel(indices)
        Jacobian(i, indices(i)) = 1;
    end
end

function Jacobian = interp1_jacobian_linear(user_time_grid, circuit_time_grid, num_elements)
    % Calculate the Jacobian matrix for linear interpolation
    Jacobian = zeros(numel(circuit_time_grid), num_elements);
    for i = 1:(numel(user_time_grid)-1)
        idx = find(circuit_time_grid >= user_time_grid(i) & circuit_time_grid <= user_time_grid(i+1));
        t1 = user_time_grid(i);
        t2 = user_time_grid(i+1);
        for j = idx'
            t = circuit_time_grid(j);
            Jacobian(j, i) = (t2 - t) / (t2 - t1);
            Jacobian(j, i+1) = (t - t1) / (t2 - t1);
        end
    end
end


function [Jacobian_X, Jacobian_Y] = calculate_jacobians(X0, Y0,Jacobian_X_interp,Jacobian_Y_interp, inp_amp, inp_phi, omega, sys, circuit_time_grid, down_factor)
    
        % % Transformation Jacobians (Vectorized)
        % partial_amp_X = X0 ./ sqrt(X0.^2 + Y0.^2);
        % partial_amp_Y = Y0 ./ sqrt(X0.^2 + Y0.^2);
        % partial_phi_X = -Y0 ./ (X0.^2 + Y0.^2);
        % partial_phi_Y = X0 ./ (X0.^2 + Y0.^2);
        % 
        % % Compute common terms
        % cos_term = cos(omega * circuit_time_grid + inp_phi);
        % sin_term = sin(omega * circuit_time_grid + inp_phi);
        % 
        % % Derivatives of inp_signal with respect to X0 and Y0
        % d_inp_signal_d_X0 = cos_term .* partial_amp_X - inp_amp .* sin_term .* partial_phi_X;
        % d_inp_signal_d_Y0 = cos_term .* partial_amp_Y - inp_amp .* sin_term .* partial_phi_Y;
        % 
        % % Combine Jacobians using element-wise multiplication
        % Jacobian_inp_signal = (d_inp_signal_d_X0 .* Jacobian_X_interp) + (d_inp_signal_d_Y0 .* Jacobian_Y_interp);
        % 
        % 
        % % Apply the RLC system response (linear operator)
        % for i = 1:size(Jacobian_inp_signal, 2)
        %     Jacobian_inp_signal(:, i) = lsim(sys, Jacobian_inp_signal(:, i), circuit_time_grid);
        % end
        % 
        % % Heterodyning
        % sin_omega_t = sin(omega * circuit_time_grid);
        % cos_omega_t = cos(omega * circuit_time_grid);
        % 
        % % Compute final Jacobians using element-wise multiplication
        % Jacobian_X = sin_omega_t .* Jacobian_inp_signal;
        % Jacobian_Y = cos_omega_t .* Jacobian_inp_signal;
        % 
        % % Apply the same lowpass filter to the Jacobians
        % Jacobian_X = 2*lowpass(Jacobian_X, 1, 64, ImpulseResponse="iir", Steepness=0.95);
        % Jacobian_Y = 2*lowpass(Jacobian_Y, 1, 64, ImpulseResponse="iir", Steepness=0.95);
        % 
        % % Downsample
        % Jacobian_X = Jacobian_X(1:down_factor:end, :);
        % Jacobian_Y = Jacobian_Y(1:down_factor:end, :);


        % Initialize Jacobians
        Jacobian_X = Jacobian_X_interp;
        Jacobian_Y = Jacobian_Y_interp;
    
        % Step 2: Transformation Jacobians (Vectorized)
        partial_amp_X = X0 ./ sqrt(X0.^2 + Y0.^2);
        partial_amp_Y = Y0 ./ sqrt(X0.^2 + Y0.^2);
        partial_phi_X = -Y0 ./ (X0.^2 + Y0.^2);
        partial_phi_Y = X0 ./ (X0.^2 + Y0.^2);
        
        % Compute common terms
        cos_term = cos(omega * circuit_time_grid + inp_phi);
        sin_term = sin(omega * circuit_time_grid + inp_phi);
        
        % Update Jacobian with the effects of amplitude and phase
        Jacobian_X = Jacobian_X .* (cos_term .* partial_amp_X - inp_amp .* sin_term .* partial_phi_X);
        Jacobian_Y = Jacobian_Y .* (cos_term .* partial_amp_Y - inp_amp .* sin_term .* partial_phi_Y);
    
        num_params = size(Jacobian_X, 2);
        for i = 1:num_params
            [Jacobian_X(:,i), ~, ~] = lsim(sys, Jacobian_X(:,i), circuit_time_grid);
            [Jacobian_Y(:,i), ~, ~] = lsim(sys, Jacobian_Y(:,i), circuit_time_grid);
        end

    
        % Step 3: Hetorodyne out
        Jacobian_X = Jacobian_X.*2  .* sin(omega * circuit_time_grid);
        Jacobian_Y = Jacobian_Y.*2  .* cos(omega * circuit_time_grid);
        
        Jacobian_X=lowpass(Jacobian_X,1,64,ImpulseResponse="iir",Steepness=0.95);
        Jacobian_Y=lowpass(Jacobian_Y,1,64,ImpulseResponse="iir",Steepness=0.95);
    
        % Downsample the Jacobians
        Jacobian_X = Jacobian_X(1:down_factor:end, :);
        Jacobian_Y = Jacobian_Y(1:down_factor:end, :);

end
