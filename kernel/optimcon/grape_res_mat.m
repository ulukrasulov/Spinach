function [traj_data,fidelity,grad]=grape_res_mat(spin_system,drifts,controls,...
                                                  waveform,rho_init,rho_targ,...
                                                  fidelity_type)

    switch exist(spin_system.control.response,"var")
        % Response Function Provided
        case true
            % Extract the response function, jacobian function, and parameters
            response_cell = spin_system.control.response;
            response_function = response_cell{1};
            jacobian_function = response_cell{2};
            parameters = response_cell{3};
        
            % Preallocate gradient results
            grad = zeros(size(waveform));
            nctrls = size(waveform,1);
        
        
            [X, Y, ~] = response_function(waveform(1,:)', waveform(2,:)', spin_system.control.pulse_dt(1), parameters{:});
            distorted_waveform(1,:) = X;
            distorted_waveform(2,:) = Y;
        
            % Check if the waveform is upsampled; adjust spin_system if necessary
            if size(distorted_waveform,2) ~= size(waveform,2)
                % Create a new spin_system with upsampled parameters
                upsample_spin_system = spin_system;
                total_pulse = spin_system.control.pulse_dt(1) * spin_system.control.pulse_nsteps;
                dt = total_pulse / size(distorted_waveform,2);
        
                % Update time grid and other parameters
                upsample_spin_system.control.pulse_dt = dt * ones(1, size(distorted_waveform,2));
                upsample_spin_system.control.pulse_nsteps = size(distorted_waveform,2);
                upsample_spin_system.control.pulse_ntpts = size(distorted_waveform,2);
                upsample_spin_system.control.keyholes = cell(1, upsample_spin_system.control.pulse_ntpts);
        
                % Apply GRAPE to the distorted waveform
                [traj_data, fidelity, grad_grape_distorted] = grape(upsample_spin_system, drifts, controls, ...
                                                                    distorted_waveform, rho_init, rho_targ, ...
                                                                    fidelity_type);
            else
                % Apply GRAPE to the distorted waveform without upsampling
                [traj_data, fidelity, grad_grape_distorted] = grape(spin_system, drifts, controls, ...
                                                                    distorted_waveform, rho_init, rho_targ, ...
                                                                    fidelity_type);
            end
        
            % Compute the gradient based on the provided Jacobian function
            if isempty(jacobian_function)
                % Jacobian function not provided; compute using finite differences
        
                Jacobian_X = compute_jacobian(response_function, waveform(1,:), waveform(2,:), ...
                                              spin_system.control.pulse_dt(1), size(distorted_waveform,2), ...
                                              parameters, 'X');
                Jacobian_Y = compute_jacobian(response_function, waveform(1,:), waveform(2,:), ...
                                              spin_system.control.pulse_dt(1), size(distorted_waveform,2), ...
                                              parameters, 'Y');
                Jacobian = {Jacobian_X, Jacobian_Y};
            elseif isequal(response_function, jacobian_function)
                % Response and Jacobian functions are the same; assume linear response
        
        
                % Use the Jacobian function to obtain the Jacobian matrices
                [Jacobian_X, Jacobian_Y] = jacobian_function(waveform(1,:)', waveform(2,:)', ...
                                                             spin_system.control.pulse_dt(1), parameters{:});
                Jacobian = {Jacobian_X, Jacobian_Y};
            else
                % Use the provided Jacobian function
                [Jacobian_X, Jacobian_Y] = jacobian_function(waveform(1,:)', waveform(2,:)', ...
                                                             spin_system.control.pulse_dt(1), parameters{:});
                Jacobian = {Jacobian_X, Jacobian_Y};
            end
        
            % Compute the gradient for each control channel
            for k = 1:nctrls
                jacobian_matrix = Jacobian{k};
                grad(k,:) = grad_grape_distorted(k,:) * jacobian_matrix;
            end
        case false
            %Response Function not provided proceed with standard GRAPE
             [traj_data, fidelity, grad] = grape(spin_system, drifts, controls, ...
                                                        waveform, rho_init, rho_targ, ...
                                                        fidelity_type);
    end


end

% Jacobian calculation using finite differences
function J = compute_jacobian(f_handle, X, Y, dt_user, size_distorted_waveform, f_parameters, which)
    epsilon = 1e-6;  % Perturbation value for finite differences
    n = length(X);   % Number of variables
    J = zeros(size_distorted_waveform, n);  % Initialize Jacobian matrix

    % Loop through each variable
    for i = 1:n
        if strcmp(which, 'X')
            % Perturb X
            X_perturb_pos = X;
            X_perturb_pos(i) = X_perturb_pos(i) + epsilon;
            X_perturb_neg = X;
            X_perturb_neg(i) = X_perturb_neg(i) - epsilon;

            % Evaluate the function at perturbed inputs
            [F_pos_X, ~, ~] = f_handle(X_perturb_pos, Y, dt_user, f_parameters{:});
            [F_neg_X, ~, ~] = f_handle(X_perturb_neg, Y, dt_user, f_parameters{:});

            % Compute the finite difference
            J(:, i) = (F_pos_X - F_neg_X) / (2 * epsilon);
        elseif strcmp(which, 'Y')
            % Perturb Y
            Y_perturb_pos = Y;
            Y_perturb_pos(i) = Y_perturb_pos(i) + epsilon;
            Y_perturb_neg = Y;
            Y_perturb_neg(i) = Y_perturb_neg(i) - epsilon;

            % Evaluate the function at perturbed inputs
            [~, F_pos_Y, ~] = f_handle(X, Y_perturb_pos, dt_user, f_parameters{:});
            [~, F_neg_Y, ~] = f_handle(X, Y_perturb_neg, dt_user, f_parameters{:});

            % Compute the finite difference
            J(:, i) = (F_pos_Y - F_neg_Y) / (2 * epsilon);
        end
    end
end
