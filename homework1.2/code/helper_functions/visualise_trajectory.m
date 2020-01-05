function visualise_trajectory(vertices, edges, cam_in_world_orientations, cam_in_world_locations, varargin)
% vertices - vertices of the object Nx3 double
% edges - edges of the object 2xN
% cam_in_world_orientations - orientaions of cameras in world coordinate
% system
% cam_in_world_locations - locationss of cameras in world coordinate
% system


% Visualize computed camera poses
% fig = figure();

num_files = length(cam_in_world_orientations);
pcshow(vertices,'VerticalAxis','Y','VerticalAxisDir','down', 'MarkerSize',30) % plot the corners of the box

hold on

% connect vertices
for i=1:length(edges)
    plot3([vertices(edges(1,i),1), vertices(edges(2, i),1)],[vertices(edges(1, i),2), vertices(edges(2,i),2)],[vertices(edges(1,i),3), vertices(edges(2, i),3)], 'color', 'blue')
end

% plot the cameras
for i=1:num_files
    
    plotCamera('Size',0.01,'Orientation',cam_in_world_orientations(:,:,i),'Location',...
        cam_in_world_locations(:,:,i), varargin{:})
    text(cam_in_world_locations(1,1,i), cam_in_world_locations(1,2,i), cam_in_world_locations(1,3,i),char(string(i)))
end

for i=1:num_files-1
    % draws a line between first_cameraLocation and second_cameraLocation
    first_cameraLocation = cam_in_world_locations(:,:,i);
    second_cameraLocation = cam_in_world_locations(:,:,i+1);
    plot3([first_cameraLocation(1), second_cameraLocation(1)],[first_cameraLocation(2), second_cameraLocation(2)],[first_cameraLocation(3), second_cameraLocation(3)], 'color', 'blue')
end

xlabel('x');
ylabel('y');
zlabel('z');
view(3);

grid on
axis equal

hold off


end