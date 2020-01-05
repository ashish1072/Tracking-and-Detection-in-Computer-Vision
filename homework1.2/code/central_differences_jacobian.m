function Jacobian = central_differences_jacobian(eps, pose, camera_param, M, e)
  
    esigma = 1.48257968*median(abs(e));
    dh = zeros(6,1);
    Jacobian = zeros(length(e),6);
    
    for p = 1:6
        dh(p,1) = eps;
        pose_idh = (pose' + dh)';
        pose_ddh = (pose' - dh)';
        fposei = project3d2image(M, camera_param, rotationVectorToMatrix(pose_idh(1:3)), pose_idh(4:6));
        for i = 1:length(fposei)
            if i == 1
                einc = fposei(:,i);
            else
                einc = vertcat(einc, fposei(:,i));
            end
        end
        fposed = project3d2image(M, camera_param, rotationVectorToMatrix(pose_ddh(1:3)), pose_ddh(4:6));
        %fposed = (worldToImage(camera_param, rotationVectorToMatrix(pose_ddh(1:3)), pose_ddh(4:6), M))';
        for i = 1:length(fposed)
            if i == 1
                edec = fposed(:,i);
            else
                edec = vertcat(edec, fposed(:,i));
            end
        end
        dh(p,1) = 0;
        Jacobian(:,p) =  (einc -edec)./(2*eps);
    end

 end














% function J = finite_differences_jacobian(A, R, T, w_3d, step)
%     c_3d = A*(R*w_3d + T);
%     fp = c_3d(1:2, :) ./ c_3d(3, :);
% %     disp('fp size:');disp(size(fp));
%     
%     % r1......................................................
%     w = rotationMatrixToVector(R);
%     w(1) = w(1) + step;
%     R_w1 = rotationVectorToMatrix(w);
%     
%     c_3d_w1 = A*(R_w1*w_3d + T);
%     fp_w1 = c_3d_w1(1:2, :) ./ c_3d_w1(3, :);
% %     disp('fp_w1 size:');disp(size(fp_w1));
%     
%     % r2
%     w = rotationMatrixToVector(R);
%     w(2) = w(2) + step;
%     R_w2 = rotationVectorToMatrix(w);
%     
%     c_3d_w2 = A*(R_w2*w_3d + T);
%     fp_w2 = c_3d_w2(1:2, :) ./ c_3d_w2(3, :);
%     
%     % r3
%     w = rotationMatrixToVector(R);
%     w(3) = w(3) + step;
%     R_w3 = rotationVectorToMatrix(w);
%     
%     c_3d_w3 = A*(R_w3*w_3d + T);
%     fp_w3 = c_3d_w3(1:2, :) ./ c_3d_w3(3, :);
%     
%     
%     % t1........................................................
%     T_t1 = [T(1) + step; T(2); T(3)];
%     c_3d_t1 = A*(R*w_3d + T_t1);
%     fp_t1 = c_3d_t1(1:2, :) ./ c_3d_t1(3, :);
%     
%     % t2
%     T_t2 = [T(1); T(2) + step; T(3)];
%     c_3d_t2 = A*(R*w_3d + T_t2);
%     fp_t2 = c_3d_t2(1:2, :) ./ c_3d_t2(3, :);
%     
%     % t3
%     T_t3 = [T(1); T(2); T(3) + step];
%     c_3d_t3 = A*(R*w_3d + T_t3);
%     fp_t3 = c_3d_t3(1:2, :) ./ c_3d_t3(3, :);
%     
%     de1 = fp_w1-fp;
%     de1 = reshape(de1,[],1);
%     
%     de2 = fp_w2-fp;
%     de2 = reshape(de2,[],1);
%     
%     de3 = fp_w3-fp;
%     de3 = reshape(de3,[],1);
%     
%     de4 = fp_t1-fp;
%     de4 = reshape(de4,[],1);
%     
%     de5 = fp_t2-fp;
%     de5 = reshape(de5,[],1);
%     
%     de6 = fp_w1-fp;
%     de6 = reshape(de6,[],1);
%     
%     % jacobian
%     J = [ de1./step, de2./step, de3./step, de4./step, de5./step, de6./step];
% %     disp('J');disp(size(J));
%     
% end




