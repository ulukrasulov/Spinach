function [traj_data,fidelity,grad]=grape_res_mat(spin_system,drifts,controls,...
                                              waveform,rho_init,rho_targ,...
                                              fidelity_type) %#ok<*PFBNS>
% Get the response matricies
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


end
