/**********************************************************************
Copyright �2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

�	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
�	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/
/*
 * For a description of the algorithm and the terms used, please see the
 * documentation for this sample.
 *
 * Each thread calculates a pixel component(rgba), by applying a filter 
 * on group of 8 neighbouring pixels in both x and y directions. 
 * Both filters are summed (vector sum) to form the final result.
 */

__constant sampler_t imageSampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_LINEAR; 

__kernel void sobel_filter(__read_only image2d_t inputImage, __write_only image2d_t outputImage)
{
	int2 coord = (int2)(get_global_id(0), get_global_id(1));

	float prog_binaryzacji = 100.0;

	bool wynik_erozji;	
	bool wynik_dylatacji;	


	// Zadanie 1

	// Erosion 3x3 square "and"
	const int n = 3;
	float4 pixels[n][n];
	pixels[n / 2][n / 2] = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));
	wynik_erozji = step(prog_binaryzacji, pixels[n / 2][n / 2]).y;
	
	for (int i = 0; i < n; i++)
	{
		for (int j = 0; j < n; j++)
		{
			pixels[i][j] = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x + i - (n / 2), coord.y + j - (n / 2))));
			wynik_erozji = wynik_erozji && step(prog_binaryzacji, pixels[i][j]).y;
		}
	}
	
	write_imageui(outputImage, coord,255*(uint)wynik_erozji);


	// Erosion 33x33 square "and"
	// const int n = 33;
	// float4 pixels[n][n];
	// pixels[n / 2][n / 2] = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));
	// wynik_erozji = step(prog_binaryzacji, pixels[n / 2][n / 2]).y;
	
	// for (int i = 0; i < n; i++)
	// {
	// 	for (int j = 0; j < n; j++)
	// 	{
	// 		pixels[i][j] = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x + i - (n / 2), coord.y + j - (n / 2))));
	// 		wynik_erozji = wynik_erozji && step(prog_binaryzacji, pixels[i][j]).y;
	// 	}
	// }
	
	// write_imageui(outputImage, coord,255*(uint)wynik_erozji);


	// Erosion 5x5 circle "and"
	// const int n = 5;
	// float4 pixels[n][n];
	// pixels[n / 2][n / 2] = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));
	// wynik_erozji = step(prog_binaryzacji, pixels[n / 2][n / 2]).y;
	
	// for (int i = 0; i < n; i++)
	// {
	// 	for (int j = 0; j < n; j++)
	// 	{
	// 		if (abs(i - (n / 2)) != (n / 2) && abs(j - (n / 2)) != (n / 2))
	// 		{
	// 			pixels[i][j] = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x + i - (n / 2), coord.y + j - (n / 2))));
	// 			wynik_erozji = wynik_erozji && step(prog_binaryzacji, pixels[i][j]).y;
	// 		}
	// 	}
	// }
	
	// write_imageui(outputImage, coord,255*(uint)wynik_erozji);



	// Dilate 3x3 square "or"
	// const int n = 3;
	// float4 pixels[n][n];
	// pixels[n / 2][n / 2] = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));
	// wynik_dylatacji = step(prog_binaryzacji, pixels[n / 2][n / 2]).y;
	
	// for (int i = 0; i < n; i++)
	// {
	// 	for (int j = 0; j < n; j++)
	// 	{
	// 		pixels[i][j] = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x + i - (n / 2), coord.y + j - (n / 2))));
	// 		wynik_dylatacji = wynik_dylatacji || step(prog_binaryzacji, pixels[i][j]).y;
	// 	}
	// }
	
	// write_imageui(outputImage, coord,255*(uint)wynik_dylatacji);


	// Dilate 33x33 square "or"
	// const int n = 33;
	// float4 pixels[n][n];
	// pixels[n / 2][n / 2] = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));
	// wynik_dylatacji = step(prog_binaryzacji, pixels[n / 2][n / 2]).y;
	
	// for (int i = 0; i < n; i++)
	// {
	// 	for (int j = 0; j < n; j++)
	// 	{
	// 		pixels[i][j] = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x + i - (n / 2), coord.y + j - (n / 2))));
	// 		wynik_dylatacji = wynik_dylatacji || step(prog_binaryzacji, pixels[i][j]).y;
	// 	}
	// }
	
	// write_imageui(outputImage, coord,255*(uint)wynik_dylatacji);


	// Dilate 5x5 circle "or"
	// const int n = 5;
	// float4 pixels[n][n];
	// pixels[n / 2][n / 2] = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)));
	// wynik_dylatacji = step(prog_binaryzacji, pixels[n / 2][n / 2]).y;
	
	// for (int i = 0; i < n; i++)
	// {
	// 	for (int j = 0; j < n; j++)
	// 	{
	// 		if (abs(i - (n / 2)) != (n / 2) && abs(j - (n / 2)) != (n / 2))
	// 		{
	// 			pixels[i][j] = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x + i - (n / 2), coord.y + j - (n / 2))));
	// 			wynik_dylatacji = wynik_dylatacji || step(prog_binaryzacji, pixels[i][j]).y;
	// 		}
	// 	}
	// }
	
	// write_imageui(outputImage, coord,255*(uint)wynik_dylatacji);


	// Zadanie 2

	wynik_erozji = step(prog_binaryzacji, convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)))).y;
	wynik_dylatacji = step(prog_binaryzacji, convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x, coord.y)))).y;

	// Erosion 3x3 square "min"
	// const int n = 3;
	
	// for (int i = 0; i < n; i++)
	// {
	// 	for (int j = 0; j < n; j++)
	// 	{
	// 		float4 pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x + i - (n / 2), coord.y + j - (n / 2))));
	// 		if (pixel.y < wynik_erozji)
	// 		{
	// 			wynik_erozji = step(prog_binaryzacji, pixel).y;
	// 		}
	// 	}
	// }
	
	// write_imageui(outputImage, coord, 255*(uint)wynik_erozji);


	// Erosion 33x33 square "min"
	// const int n = 33;
	
	// for (int i = 0; i < n; i++)
	// {
	// 	for (int j = 0; j < n; j++)
	// 	{
	// 		float4 pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x + i - (n / 2), coord.y + j - (n / 2))));
	// 		if (pixel.y < wynik_erozji)
	// 		{
	// 			wynik_erozji = step(prog_binaryzacji, pixel).y;
	// 		}
	// 	}
	// }
	
	// write_imageui(outputImage, coord, 255*(uint)wynik_erozji);


	// Erosion 5x5 circle "min"
	// const int n = 5;
	
	// for (int i = 0; i < n; i++)
	// {
	// 	for (int j = 0; j < n; j++)
	// 	{
	// 		if (abs(i - (n / 2)) != (n / 2) && abs(j - (n / 2)) != (n / 2))
	// 		{
	// 			float4 pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x + i - (n / 2), coord.y + j - (n / 2))));
	// 			if (pixel.y < wynik_erozji)
	// 			{
	// 				wynik_erozji = step(prog_binaryzacji, pixel).y;
	// 			}
	// 		}
	// 	}
	// }
	
	// write_imageui(outputImage, coord, 255*(uint)wynik_erozji);



	// Dilate 3x3 square "max"
	// const int n = 3;
	
	// for (int i = 0; i < n; i++)
	// {
	// 	for (int j = 0; j < n; j++)
	// 	{
	// 		float4 pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x + i - (n / 2), coord.y + j - (n / 2))));
	// 		if (pixel.y > wynik_dylatacji)
	// 		{
	// 			wynik_dylatacji = step(prog_binaryzacji, pixel).y;
	// 		}
	// 	}
	// }
	
	// write_imageui(outputImage, coord, 255*(uint)wynik_dylatacji);


	// Dilate 33x33 square "max"
	// const int n = 33;
	
	// for (int i = 0; i < n; i++)
	// {
	// 	for (int j = 0; j < n; j++)
	// 	{
	// 		float4 pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x + i - (n / 2), coord.y + j - (n / 2))));
	// 		if (pixel.y > wynik_dylatacji)
	// 		{
	// 			wynik_dylatacji = step(prog_binaryzacji, pixel).y;
	// 		}
	// 	}
	// }
	
	// write_imageui(outputImage, coord, 255*(uint)wynik_dylatacji);


	// Dilate 5x5 circle "max"
	// const int n = 5;
	
	// for (int i = 0; i < n; i++)
	// {
	// 	for (int j = 0; j < n; j++)
	// 	{
	// 		if (abs(i - (n / 2)) != (n / 2) && abs(j - (n / 2)) != (n / 2))
	// 		{
	// 			float4 pixel = convert_float4(read_imageui(inputImage, imageSampler, (int2)(coord.x + i - (n / 2), coord.y + j - (n / 2))));
	// 			if (pixel.y > wynik_dylatacji)
	// 			{
	// 				wynik_dylatacji = step(prog_binaryzacji, pixel).y;
	// 			}
	// 		}
	// 	}
	// }
	
	// write_imageui(outputImage, coord, 255*(uint)wynik_dylatacji);	
}
	
