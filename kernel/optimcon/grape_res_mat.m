function [traj_data,fidelity,grad]=grape_res_mat(spin_system,drifts,controls,...
                                              waveform,rho_init,rho_targ,...
                                              fidelity_type) %#ok<*PFBNS>
% Get the response matricies
response_matricies=spin_system.control.response;

% Preallocate grad results
grad = zeros(size(waveform));

nctrls = size(waveform,1);

% Apply response matrix (or function handle) to waveform
% Check if the response is a function handle
if isa(response_matricies{1,1}, 'function_handle')
    % Call the function handle, passing in the parameters from response{1,2}
    [X,Y,dt] = response_matricies{1,1}(waveform(1,:)',waveform(2,:)',spin_system.control.pulse_dt(1),response_matricies{1,2}{:});
    distorted_waveform(1,:)=X;
    distorted_waveform(2,:)=Y;
else
    % Initilise distorted waveform array. Must have number of rows as response
    % matrix. This can be larger than initial waveform if there is upsampling
    distorted_waveform = zeros(nctrls, size(response_matricies{1,1}, 1));
    for k = 1:nctrls
       % Apply response matrix to waveform. If not supplied it should be an
% identity matrix of size waveform
        distorted_waveform(k,:) = response_matricies{1,k} * waveform(k,:)';
    end
end

% Check if waveform is upsampled. Time grid need to be upsampled to.
if size(distorted_waveform,2) ~=size(waveform,2)
    % Create new struct with new upsampled parameters to input into GRAPE

    upsample_spin_system= spin_system;

    total_pulse= spin_system.control.pulse_dt(1)*spin_system.control.pulse_nsteps;
    dt=total_pulse/size(distorted_waveform,2);
    % Change the time grid and dt
    upsample_spin_system.control.pulse_dt=dt*ones(1,size(distorted_waveform,2));
    upsample_spin_system.control.pulse_nsteps=size(distorted_waveform,2);
    upsample_spin_system.control.pulse_ntpts=size(distorted_waveform,2);

    % Change empty keyholes for now
    upsample_spin_system.control.keyholes=cell(1,upsample_spin_system.control.pulse_ntpts);

    % Apply normal GRAPE to distorted waveform
    [traj_data,fidelity,grad_grape_distorted]=grape(upsample_spin_system,drifts,controls,...
                                              distorted_waveform,rho_init,rho_targ,...
                                              fidelity_type);
else
    % Apply normal GRAPE to distorted waveform
    [traj_data,fidelity,grad_grape_distorted]=grape(spin_system,drifts,controls,...
                                                  distorted_waveform,rho_init,rho_targ,...
                                                  fidelity_type);
end


% The gradient of the final fidelity with respect to each amplitude when a
% response function is applied is a weighed sum of grad_grape where the
% weights are the columns the response matrix

if isa(response_matricies{1,1}, 'function_handle')
    Jacobian_X=compute_jacobian(response_matricies{1,1},waveform(1,:),waveform(2,:),spin_system.control.pulse_dt(1),size(distorted_waveform,2),response_matricies{1,2},'X');
    Jacobian_Y=compute_jacobian(response_matricies{1,1},waveform(2,:),waveform(2,:),spin_system.control.pulse_dt(1),size(distorted_waveform,2),response_matricies{1,2},'Y');
    Jacobian={Jacobian_X Jacobian_Y};
    % Loop through each control channel (X and Y)
    for k=1:size(waveform,1) 
        % Extract the response matrix for the current user
        jacobian_matrix = Jacobian{1,k};
        
        % Compute the gradient for this user
        grad(k,:) = grad_grape_distorted(k,:) * jacobian_matrix;
    end
else
    % Loop through each control channel (X and Y)
    for k=1:size(waveform,1) 
        % Extract the response matrix for the current user
        response_matrix = response_matricies{1,k};
        
        % Compute the gradient for this user
        grad(k,:) = grad_grape_distorted(k,:) * response_matrix;
    end
end




end

% Jacobian calculation using finite difference
function J = compute_jacobian(f_handle, X,Y, dt_user, size,f_parameters,which)
    epsilon = 1e-6;  % Perturbation value for finite differences
    n = length(X);    % Number of variables
    J = zeros(size, n);  % Initialize Jacobian matrix

    % Loop through each variable
    for i = 1:n

        if which =='X'
            % Perturb the input in the positive direction
            X_perturb_pos = X;
            X_perturb_pos(i) = X_perturb_pos(i) + epsilon;
    
            % Perturb the input in the negative direction
            X_perturb_neg = X;
            X_perturb_neg(i) = X_perturb_neg(i) - epsilon;

            % Evaluate the function at perturbed inputs
            [F_pos_X, ~, ~] = f_handle(X_perturb_pos, Y, dt_user, f_parameters{:});
            [F_neg_X, ~, ~] = f_handle(X_perturb_neg, Y, dt_user, f_parameters{:});
    
            % Compute the finite difference for each element of the Jacobian
            J(:, i) = (F_pos_X- F_neg_X) / (2 * epsilon);
        elseif which =='Y'
            Y_perturb_pos = Y;
            Y_perturb_pos(i) = Y_perturb_pos(i) + epsilon;
    
            % Perturb the input in the negative direction
            Y_perturb_neg = Y;
            Y_perturb_neg(i) = Y_perturb_neg(i) - epsilon;

             % Evaluate the function at perturbed inputs
            [~, F_pos_Y, ~] = f_handle(X, Y_perturb_pos, dt_user, f_parameters{:});
            [~, F_neg_Y, ~] = f_handle(X, Y_perturb_neg, dt_user, f_parameters{:});
    
            % Compute the finite difference for each element of the Jacobian
            J(:, i) = (F_pos_Y- F_neg_Y) / (2 * epsilon);
        end


    
    end
end
