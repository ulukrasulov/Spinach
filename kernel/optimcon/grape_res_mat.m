function [traj_data,fidelity,grad]=grape_res_mat(spin_system,drifts,controls,...
                                                  waveform,rho_init,rho_targ,...
                                                  fidelity_type)
    if isfield(spin_system.control,'response')  && isa(spin_system.control.response{1,1}, 'function_handle')
        % Response Function Provided
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
                [Jacobian_X, Jacobian_Y,~] = jacobian_function(waveform(1,:)', waveform(2,:)', ...
                                                             spin_system.control.pulse_dt(1), parameters{:});
                Jacobian = {Jacobian_X, Jacobian_Y};
            else
                % Use the provided Jacobian function
                [Jacobian_X, Jacobian_Y,~] = jacobian_function(waveform(1,:)', waveform(2,:)', ...
                                                             spin_system.control.pulse_dt(1), parameters{:});
                Jacobian = {Jacobian_X, Jacobian_Y};
            end
        
            % Compute the gradient for each control channel
            for k = 1:nctrls
                jacobian_matrix = Jacobian{k};
                grad(k,:) = grad_grape_distorted(k,:) * jacobian_matrix;
            end
    elseif isfield(spin_system.control,'response')  && ismatrix(spin_system.control.response{1,1})

        response_matricies=spin_system.control.response;

        % Preallocate grad results
        grad=zeros(size(waveform));
        
        nctrls=size(waveform,1);
        
        % Initilise distorted waveform array. Must have number of rows as response
        % matrix. This can be larger than initial waveform if there is upsampling
        distorded_waveform=zeros(nctrls,size(response_matricies{1,1},1));
        
        % Apply response matrix to waveform. If not supplied it should be an
        % identity matrix of size waveform
        for k=1:nctrls
            distorded_waveform(k,:)=response_matricies{1,k}*waveform(k,:)';
        end
        
        % Check if waveform is upsampled. Time grid need to be upsampled to.
        if size(distorded_waveform,2) ~=size(waveform,2)
            % Create new struct with new upsampled parameters to input into GRAPE
        
            upsample_spin_system= spin_system;
        
            total_pulse= spin_system.control.pulse_dt(1)*spin_system.control.pulse_nsteps;
            dt=total_pulse/size(distorded_waveform,2);
            % Change the time grid and dt
            upsample_spin_system.control.pulse_dt=dt*ones(1,size(distorded_waveform,2));
            upsample_spin_system.control.pulse_nsteps=size(distorded_waveform,2);
            upsample_spin_system.control.pulse_ntpts=size(distorded_waveform,2);
        
            % Change empty keyholes for now
            upsample_spin_system.control.keyholes=cell(1,upsample_spin_system.control.pulse_ntpts);
        
            % Apply normal GRAPE to distorted waveform
            [traj_data,fidelity,grad_grape_distorted]=grape(upsample_spin_system,drifts,controls,...
                                                      distorded_waveform,rho_init,rho_targ,...
                                                      fidelity_type);
        else
            % Apply normal GRAPE to distorted waveform
            [traj_data,fidelity,grad_grape_distorted]=grape(spin_system,drifts,controls,...
                                                          distorded_waveform,rho_init,rho_targ,...
                                                          fidelity_type);
        end
        

        % The gradient of the final fidelity with respect to each amplitude when a
        % response function is applied is a weighed sum of grad_grape where the
        % weights are the columns the response matrix
        
        % Loop through each control channel (X and Y)
        for k=1:size(waveform,1) 
            % Extract the response matrix for the current user
            response_matrix = response_matricies{1,k};
            
            % Compute the gradient for this user
            grad(k,:) = grad_grape_distorted(k,:) * response_matrix;
        end

    else
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
