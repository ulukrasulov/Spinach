function [traj_data, fidelity, grad] = grape_res_mat(spin_system, drifts, controls, ...
                                                    waveform, rho_init, rho_targ, ...
                                                    fidelity_type)
    % Check if response functions are provided
    if isfield(spin_system.control, 'response') 
        response_input = spin_system.control.response;
        
        % Determine if response_input is a single set or multiple sets
        if iscell(response_input) && size(response_input,1) ~=1
            % Multiple response sets (Ensemble)
            response_sets = response_input;
            num_sets = size(response_sets,1);
            
            % Initialize accumulators
            traj_data_accum = [];
            fidelity_accum = 0;
            grad_accum = 0;
            
            for set_idx = 1:num_sets
                current_response = {response_sets{set_idx,:}};
                
                % Create a temporary spin_system with the current response set
                spin_system_current = spin_system;
                spin_system_current.control.response = current_response;
                
                % Call grape_res_mat_single_set for the current set
                [temp_traj_data, temp_fidelity, temp_grad] = grape_res_mat_single_set( ...
                    spin_system_current, drifts, controls, waveform, rho_init, rho_targ, fidelity_type);
                
                % % Initialize struct for traj_data same as returned by GRAPE
                % traj_data_accum = struct('forward', zeros(size(temp_traj_data.forward)), 'backward', zeros(size(temp_traj_data.forward)));


                % Accumulate traj_data
                if isempty(traj_data_accum)
                    traj_data_accum = temp_traj_data;
                else
                    traj_data_accum.forward = traj_data_accum.forward + temp_traj_data.forward;
                    traj_data_accum.backward = traj_data_accum.backward + temp_traj_data.backward;
                end
                
                % Accumulate fidelity and grad
                fidelity_accum = fidelity_accum + temp_fidelity;
                grad_accum = grad_accum + temp_grad;
            end
            
            % Average the accumulated results
            fidelity = fidelity_accum / num_sets;
            grad = grad_accum / num_sets;
            
            % Average traj_data
            traj_data_accum.forward = traj_data_accum.forward / num_sets;
            traj_data_accum.backward = traj_data_accum.backward / num_sets;
            traj_data = traj_data_accum;
            
        else
            % Single response set
            [traj_data, fidelity, grad] = grape_res_mat_single_set(spin_system, drifts, controls, ...
                                                                   waveform, rho_init, rho_targ, ...
                                                                   fidelity_type);
        end
    else
        % No response function provided, use standard GRAPE
        [traj_data, fidelity, grad] = grape(spin_system, drifts, controls, ...
                                            waveform, rho_init, rho_targ, ...
                                            fidelity_type);
    end
end

function [traj_data, fidelity, grad] = grape_res_mat_single_set(spin_system, drifts, controls, ...
                                                               waveform, rho_init, rho_targ, ...
                                                               fidelity_type)
    % Existing grape_res_mat functionality for a single set of response functions
    
    % Multiple response functions are provided as a cell array of function handles
    response_cell = spin_system.control.response;
    num_responses = numel(response_cell);  % Number of response functions

    % Initialize variables
    nctrls = size(waveform, 1);             % Number of control channels
    n_time_steps = size(waveform, 2);       % Number of time steps
    grad = zeros(nctrls, n_time_steps);     % Initialize gradient matrix

    % Preallocate cell arrays to store intermediate distorted waveforms, dt, and Jacobians
    distorted_waveform_intermediate = cell(num_responses + 1, 1);
    distorted_waveform_intermediate{1} = waveform;  % Original waveform

    dt_intermediate = cell(num_responses + 1, 1);
    dt_intermediate{1} = spin_system.control.pulse_dt(1);  % Initial dt

    Jacobian_X_cell = cell(num_responses, 1);  % To store Jacobian_X for each response
    Jacobian_Y_cell = cell(num_responses, 1);  % To store Jacobian_Y for each response

    % Apply all response functions sequentially and store Jacobians
    for n = 1:num_responses  
        response_function_handle = response_cell{n};

        % Current distorted waveform and dt
        current_wf = distorted_waveform_intermediate{n};
        current_dt = dt_intermediate{n};

        % Apply the response function
        % Each response function returns [X_new, Y_new, updated_dt, Jacobian_X, Jacobian_Y]
        [X_new, Y_new, updated_dt, Jacobian_X, Jacobian_Y] = response_function_handle(current_wf(1,:)', current_wf(2,:)', current_dt);

        % Update the distorted waveform
        distorted_waveform_X = X_new';
        distorted_waveform_Y = Y_new';
        distorted_waveform_intermediate{n + 1} = [distorted_waveform_X; distorted_waveform_Y];
        dt_intermediate{n + 1} = updated_dt;

        % Store Jacobians
        Jacobian_X_cell{n} = Jacobian_X;
        Jacobian_Y_cell{n} = Jacobian_Y;
    end

    % Check for upsampling by comparing the number of time steps
    if size(distorted_waveform_intermediate{end}, 2) ~= n_time_steps
        % Upsampling required
        upsample_spin_system = spin_system;
        total_pulse = spin_system.control.pulse_dt(1) * spin_system.control.pulse_nsteps;
        dt_new = total_pulse / size(distorted_waveform_intermediate{end}, 2);

        % Update pulse timing parameters
        upsample_spin_system.control.pulse_dt = dt_new * ones(1, size(distorted_waveform_intermediate{end}, 2));
        upsample_spin_system.control.pulse_nsteps = size(distorted_waveform_intermediate{end}, 2);
        upsample_spin_system.control.pulse_ntpts = size(distorted_waveform_intermediate{end}, 2);
        upsample_spin_system.control.keyholes = cell(1, upsample_spin_system.control.pulse_ntpts);

        % Apply GRAPE to the upsampled distorted waveform
        [traj_data, fidelity, grad_grape_distorted] = grape(upsample_spin_system, drifts, controls, ...
                                                            distorted_waveform_intermediate{end}, rho_init, rho_targ, ...
                                                            fidelity_type);
    else
        % No upsampling required, apply GRAPE directly
        [traj_data, fidelity, grad_grape_distorted] = grape(spin_system, drifts, controls, ...
                                                            distorted_waveform_intermediate{end}, rho_init, rho_targ, ...
                                                            fidelity_type);
    end

    % Initialize overall Jacobians for X and Y channels as identity matrices
    Overall_Jacobian_X = eye(n_time_steps);
    Overall_Jacobian_Y = eye(n_time_steps);

    % Compute the overall Jacobian by sequentially multiplying Jacobians
    for n = 1:num_responses
        % Retrieve stored Jacobians
        Jacobian_X = Jacobian_X_cell{n};
        Jacobian_Y = Jacobian_Y_cell{n};

        % Accumulate the overall Jacobians
        Overall_Jacobian_X = Jacobian_X * Overall_Jacobian_X;
        Overall_Jacobian_Y = Jacobian_Y * Overall_Jacobian_Y;
    end

    % Adjust the gradient for each control channel
    for k = 1:nctrls
        % Adjust gradient for the X component
        grad(k, :) = grad_grape_distorted(k, :) * Overall_Jacobian_X;
    end
end
