function [J] = jacobian(A, w, w_3d, cam_3d)
    
    image_3d = A*cam_3d;
    J = zeros(2*size(w_3d,2), 6);
    R = rotationVectorToMatrix(w');
    v = skew_symm(w); 
    I = eye(3);
    dR_dp1 = (1/norm(w)^2)*(w(1)*v + skew_symm(cross(w,(I - R)*I(:,1))))*R ;
    dR_dp2 = (1/norm(w)^2)*(w(2)*v + skew_symm(cross(w,(I - R)*I(:,2))))*R ;
    dR_dp3 = (1/norm(w)^2)*(w(3)*v + skew_symm(cross(w,(I - R)*I(:,3))))*R ;

    for i=1:size(w_3d,2)
        dm_dmh = [1.0/image_3d(3,i) 0  -image_3d(1,i)/image_3d(3,i)^2;
                   0       1.0/image_3d(3,i) -image_3d(2,i)/image_3d(3,i)^2];
        dmh_dM = A;
        dM_dp = [dR_dp1*w_3d(:,i) dR_dp2*w_3d(:,i) dR_dp3*w_3d(:,i) eye(3)];
        J(2*(i-1)+1:2*(i-1)+2, :) = dm_dmh * dmh_dM * dM_dp;
    end
end

function q = skew_symm(w)
    q =[0 -w(3) w(2);  w(3) 0  -w(1); -w(2) w(1) 0];
end