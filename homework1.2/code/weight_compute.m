function W =  weight_compute( w, T, camera_params, image_2d, world_3d)
    
     R = rotationVectorToMatrix(w);
     projected_2d = worldToImage(camera_params, R' ,(-R*T')', world_3d');
     e = projected_2d' - image_2d;
     e = reshape(e,[],1);

     sigma = 1.48257968*median(abs(e));
     r = e/sigma;                    % scaled residuals to bring them into Gaussian Distribtn
     w = zeros(size(e,1),1);
     c = 4.685;                      % M-estimator constant
     
     for i=1:size(e,1)
         if abs(r(i)) < c
            w(i) = (1 - (r(i)/c)^2)^2;
         end
     end
     
     W = diag(w);     
end