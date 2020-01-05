clear
clc
close all
addpath('helper_functions')
run('vlfeat-0.9.21/toolbox/vl_setup')

%% Setup
% path to the images folder
path_img_dir = '../data/tracking/valid/img';
% path to object ply file
object_path = '../data/teabox.ply';
% path to results folder
results_path = '../data/tracking/valid/results';

% Read the object's geometry 
% Here vertices correspond to object's corners and faces are triangles
[vertices, faces] = read_ply(object_path);

% Create directory for results
if ~exist(results_path,'dir') 
    mkdir(results_path); 
end

% Load Ground Truth camera poses for the validation sequence
% Camera orientations and locations in the world coordinate system
load('gt_valid.mat')

% TODO: setup camera parameters (camera_params) using cameraParameters()
IntrinsicMatrix = [2960.37845 0 0; 0 2960.37845 0; 1841.68855 1235.23369 1];
camera_params = cameraParameters('IntrinsicMatrix',IntrinsicMatrix);

%% Get all filenames in images folder

FolderInfo = dir(fullfile(path_img_dir, '*.JPG'));
Filenames = fullfile(path_img_dir, {FolderInfo.name} );
num_files = length(Filenames);
% num_files = 25;


% Place predicted camera orientations and locations in the world coordinate system for all images here
cam_in_world_orientations = zeros(3,3,num_files);
cam_in_world_locations = zeros(1,3,num_files);

%% Detect SIFT keypoints in all images

% You will need vl_sift() and vl_ubcmatch() functions
% download vlfeat (http://www.vlfeat.org/download.html) and unzip it somewhere
% Don't forget to add vlfeat folder to MATLAB path

% Place SIFT keypoints and corresponding descriptors for all images here
% keypoints = cell(num_files,1); 
% descriptors = cell(num_files,1); 
% 
% for i=1:length(Filenames)
%     fprintf('Calculating sift features for image: %d \n', i)
%     
% %    TODO: Prepare the image (img) for vl_sift() function
%     img = single(rgb2gray(imread(char(Filenames(i)))));
%     [keypoints{i}, descriptors{i}] = vl_sift(img) ;
% end

% Save sift features and descriptors and load them when you rerun the code to save time
% save('sift_descriptors.mat', 'descriptors')
% save('sift_keypoints.mat', 'keypoints')

load('sift_descriptors.mat');
load('sift_keypoints.mat');
 
%% Initialization: Compute camera pose for the first image

% As the initialization step for tracking
% you need to compute the camera pose for the first image 
% The first image and it's camera pose will be your initial frame 
% and initial camera pose for the tracking process

% You can use estimateWorldCameraPose() function or your own implementation
% of the PnP+RANSAC from previous tasks

% You can get correspondences for PnP+RANSAC either using your SIFT model from the previous tasks
% or by manually annotating corners (e.g. with mark_images() function)


% TODO: Estimate camera position for the first image
% imshow('vertices.png')
% labeled_points = mark_image("../data/tracking/valid/img/color_000006.JPG",8);
% save('labeled_points.mat', 'labeled_points');
load('labeled_points.mat');

%   filtering of NaN values
    [rows, columns] = find(~isnan(labeled_points(:,:,1)));
    nonnan_idx = unique(rows);
    
    image_points = labeled_points(nonnan_idx,:,1);
    world_points = vertices(nonnan_idx, :);
        
[init_orientation, init_location] = estimateWorldCameraPose(image_points, world_points, camera_params, 'MaxReprojectionError', 2);

cam_in_world_orientations(:,:, 1) = init_orientation;
cam_in_world_locations(:,:, 1) = init_location;

% Visualise the pose for the initial frame
edges = [[1, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7]
    [2, 4, 5, 3, 6, 4, 7, 8, 6, 8, 7, 8]];
figure()
hold on;
imshow(char(Filenames(1)), 'InitialMagnification', 'fit');
title(sprintf('Initial Image Camera Pose'));
% Plot bounding box
points = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,1), cam_in_world_locations(:, :, 1));
for j=1:12
    plot(points(1, edges(:, j)), points(2, edges(:,j)), 'color', 'b');
end
hold off;


%% IRLS nonlinear optimisation

% Now you need to implement the method of iteratively reweighted least squares (IRLS)
% to optimise reprojection error between consecutive image frames

% Method steps:
% 1) Back-project SIFT keypoints from the initial frame (image i) to the object using the
% initial camera pose and the 3D ray intersection code from the task 1. 
% This will give you 3D coordinates (in the world coordinate system) of the
% SIFT keypoints from the initial frame (image i) that correspond to the object
% 2) Find matches between descriptors of back-projected SIFT keypoints from the initial frame (image i) and the
% SIFT keypoints from the subsequent frame (image i+1) using vl_ubcmatch() from VLFeat library
% 3) Project back-projected SIFT keypoints onto the subsequent frame (image i+1) using 3D coordinates from the
% step 1 and the initial camera pose 
% 4) Compute the reprojection error between 2D points of SIFT
% matches for the subsequent frame (image i+1) and 2D points of projected matches
% from step 3
% 5) Implement IRLS: for each IRLS iteration compute Jacobian of the reprojection error with respect to the pose
% parameters and update the camera pose for the subsequent frame (image i+1)
% 6) Now the subsequent frame (image i+1) becomes the initial frame for the
% next subsequent frame (image i+2) and the method continues until camera poses for all
% images are estimated

% We suggest you to validate the correctness of the Jacobian implementation
% either using Symbolic toolbox or finite differences approach

% TODO: Implement IRLS method for the reprojection error optimisation
% You can start with these parameters to debug your solution 
% but you should also experiment with their different values

threshold_irls = 0.005; % update threshold for IRLS
N =20;                  % number of iterations
threshold_ubcmatch = 6; % matching threshold for vl_ubcmatch()
rng(400)

% Place model's SIFT keypoints coordinates and descriptors here

% vertices for trianglerayintersection
vert0 = vertices((faces(:,1)+1),:);
vert1 = vertices((faces(:,2)+1),:);
vert2 = vertices((faces(:,3)+1),:);

sift_matches=cell(num_files,1);
num_samples = 33000;

for i=1:(num_files-1)
% for i=1:5
    fprintf("Running IRLS for image: %d \n",i);
    
% 1) Back projection of SIFT points    
%     num_samples = size(keypoints{i},2);
   
    perm = randperm(size(keypoints{i},2)) ;
    sel = perm(1:num_samples);
    model.coord3d = [];
    model.descriptors = [];
    
    P = IntrinsicMatrix.'*[cam_in_world_orientations(:,:,i) -cam_in_world_orientations(:,:,i)*cam_in_world_locations(:,:,i).'];
    Q = P(:,1:3);
    q = P(:,4);
    orig = -inv(Q)*q; 
    
    for j=1:num_samples
        sift_coord = keypoints{i}(1:2, sel(j));
        dir = (inv(Q)* [sift_coord;1])'; 
        [intersect, t, u, v, xcoor] = TriangleRayIntersection (orig', dir, vert0, vert1, vert2, 'planeType', 'one sided');
        if any(intersect == 1)
           p = find(intersect == 1);
           model.coord3d = [model.coord3d , xcoor(p,:)'];  
           model.descriptors = [model.descriptors, descriptors{i}(:, sel(j))];
        end
    end  
    
% 2) Sift_matches
     sift_matches{i} = vl_ubcmatch(descriptors{i+1}, model.descriptors, threshold_ubcmatch);  
    
% 5) IRLS 

     R = cam_in_world_orientations(:,:,i);
     T = cam_in_world_locations(:,:,i);
     w = rotationMatrixToVector(R); 
     p = [w T];
     u = threshold_irls + 1;
     lambda = 0.001;
     step = .01;
     A = IntrinsicMatrix';
     j=0;

    while (j <= N && u > threshold_irls)
       
        R = rotationVectorToMatrix(p(1:3));
        T = p(4:6);
        
% 3) Projection of sift matched 3D World Points  
        world_3d = model.coord3d(:, sift_matches{i}(2,:));
        projected_2d = worldToImage(camera_params, R', (-R*T')', world_3d');

% 4) Reproj Error
        image_2d = keypoints{i+1}(1:2, sift_matches{i}(1,:));
        f = projected_2d' - image_2d;
        e = reshape(f,[],1);
        
        % Weight Matrix
        W = weight_compute(p(1:3), p(4:6), camera_params, image_2d, world_3d);
        
        % Energy Function
        E = energy(p(1:3), p(4:6), camera_params, image_2d, world_3d);
        
        % Jacobian
        J = central_differences_jacobian(step, p, camera_params, world_3d, e);
        % cam_3d = R*world_3d -R*T';
        % J = jacobian(A, p(1:3)', world_3d, cam_3d);    // analytical jocobian
        
        delta = - inv(J'*W*J + lambda*eye(size(J,2)))*(J'*W*e);
        p_new = p + delta'; 
        
        E_new = energy(p_new(1:3), p_new(4:6), camera_params, image_2d, world_3d);
        
        if( E_new > E)
           lambda = 10*lambda ;
        else
           lambda = lambda*0.1;
           p = p_new;
        end
        
        u = norm(delta);
        j = j+1;
    end
 
    cam_in_world_orientations(:,:,(i+1)) = rotationVectorToMatrix(p_new(1:3));
    cam_in_world_locations(:,:,(i+1)) = p_new(4:6);
    
end


%% Plot camera trajectory in 3D world CS + cameras

figure()
% Predicted trajectory
visualise_trajectory(vertices, edges, cam_in_world_orientations, cam_in_world_locations, 'Color', 'b');
hold on;
% Ground Truth trajectory
visualise_trajectory(vertices, edges, gt_valid.orientations, gt_valid.locations, 'Color', 'g');
hold off;
title('\color{green}Ground Truth trajectory \color{blue}Predicted trajectory')

%% Visualize bounding boxes

figure()
for i=1:num_files
    
    imshow(char(Filenames(i)), 'InitialMagnification', 'fit');
    title(sprintf('Image: %d', i))
    hold on
    % Ground Truth Bounding Boxes
    points_gt = project3d2image(vertices',camera_params, gt_valid.orientations(:,:,i), gt_valid.locations(:, :, i));
    % Predicted Bounding Boxes
    points_pred = project3d2image(vertices',camera_params, cam_in_world_orientations(:,:,i), cam_in_world_locations(:, :, i));
    for j=1:12
        plot(points_gt(1, edges(:, j)), points_gt(2, edges(:,j)), 'color', 'g');
        plot(points_pred(1, edges(:, j)), points_pred(2, edges(:,j)), 'color', 'b');
    end
    hold off;
    
    filename = fullfile(results_path, strcat('image', num2str(i), '.png'));
    saveas(gcf, filename)
end

%% Bonus part

% Save estimated camera poses for the validation sequence using Vision TUM trajectory file
% format: https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
% Then estimate Absolute Trajectory Error (ATE) and Relative Pose Error for
% the validation sequence using python tools from: https://vision.in.tum.de/data/datasets/rgbd-dataset/tools
% In this task you should implement you own function to convert rotation matrix to quaternion

% Save estimated camera poses for the test sequence using Vision TUM 
% trajectory file format

% Attach the file with estimated camera poses for the test sequence to your code submission
% If your code and results are good you will get a bonus for this exercise
% We are expecting the mean absolute translational error (from ATE) to be
% approximately less than 1cm

% TODO: Estimate ATE and RPE for validation and test sequences

