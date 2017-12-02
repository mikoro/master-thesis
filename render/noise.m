N = 2048;
image = zeros(N,N,3);

F1 = fftshift(linspace(-pi, pi*(1-2/N), N));
[X,Y] = meshgrid(F1,F1);
g = exp(2i*pi*rand(N,N)) ./ sqrt(X.^2+Y.^2);
g(1,1) = 1;
I = real(ifft2(g));
I = I - min(min(I));
I = I / max(max(I));
image(:,:,1)=I;

g = exp(2i*pi*rand(N,N)) ./ sqrt(X.^2+Y.^2);
g(1,1) = 1;
I = real(ifft2(g));
I = I - min(min(I));
I = I / max(max(I));
image(:,:,2)=I;

g = exp(2i*pi*rand(N,N)) ./ sqrt(X.^2+Y.^2);
g(1,1) = 1;
I = real(ifft2(g));
I = I - min(min(I));
I = I / max(max(I));
image(:,:,3)=I;

%imshow(max(0.0, image).^0.454545);
imwrite(max(0.0, image).^0.454545, 'noise_texture.png');
