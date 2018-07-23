n = 30000;
kappa1 = 50;
kappa2 = 10;
mu_i = 1;
mu1 = [mu_i; mu_i; mu_i];
mu2 = [-1; -0.5; -0.5];
% a = 5;
% b = 1;
% r = 1;
% col = 0;
display_points_size = 3;
% c = [col col col];



% fid = fopen('~/Dropbox/movmf_cuda/dataset.dat', 'w+');
% fprintf(fid, '3\n%i\n', n);
% for i = 1:n
% fprintf(fid, '%f %f %f\n', x(i), y(i), z(i));
% endfor
% fclose(fid);
vectors1  = vsamp(mu1, kappa1, n);

x1 = vectors1(:,1);
y1 = vectors1(:,2);
z1 = vectors1(:,3);

vectors2  = vsamp(mu2, kappa2, n);

x2 = vectors2(:,1);
y2 = vectors2(:,2);
z2 = vectors2(:,3);

x = [x1; x2];
y = [y1; y2];
z = [z1; z2];

vectors3 = [vectors1; vectors2];

fid = fopen('dataset.dat', 'w+');
fprintf(fid, '3\n%i\n', 2*n);
for i = 1:n
	fprintf(fid, '%f %f %f\n', x1(i), y1(i), z1(i));
endfor
for i = 1:n
	fprintf(fid, '%f %f %f\n', x2(i), y2(i), z2(i));
endfor
fclose(fid);

% scatter3(x,y,z, display_points_size,z);
% xlabel ("x");
% ylabel ("y");
% zlabel ("z");
