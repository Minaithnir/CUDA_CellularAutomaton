#include "CellularAutomaton.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <time.h>

void e(std::string string)
{
	std::cout << string << std::endl;
}

CellularAutomaton::CellularAutomaton(void)
{
	world = NULL;
	d_w = NULL;
	d_nW = NULL;
	d_pixels = NULL;
	texture = NULL;

	createWorld(WORLD_W, WORLD_H);
}

CellularAutomaton::CellularAutomaton(int w, int h)
{
	world = NULL;
	d_w = NULL;
	d_nW = NULL;
	d_pixels = NULL;
	texture = NULL;

	createWorld(w, h);
}

CellularAutomaton::~CellularAutomaton(void)
{
	cudaFree(d_w);
	cudaFree(d_nW);
	cudaFree(d_pixels);
}

void CellularAutomaton::resize(int w, int h)
{
	createWorld(w, h);
}

void CellularAutomaton::createWorld(int w, int h)
{
    cudaError_t cudaStatus;

	width = w;
	height = h;
	
	if(world != NULL)
		delete[] world;

	if(d_w != NULL)
		cudaFree(d_w);

	if(d_nW != NULL)
		cudaFree(d_nW);

	if(d_pixels != NULL)
		cudaFree(d_pixels);
		
	world = new bool[cellCount()];

	cudaStatus = cudaMalloc((void**)&d_w, cellCount() * sizeof(bool));
	cudaStatus = cudaMalloc((void**)&d_nW, cellCount() * sizeof(bool));
	cudaStatus = cudaMalloc((void**)&d_pixels, pixelsCount() * sizeof(sf::Uint8));

	if(texture != NULL)
		delete texture;

	texture = new sf::Texture;
    texture->create(width, height);
    texture->setSmooth(false);
    sprite.setTexture(*texture);

	reset();
}

int CellularAutomaton::getGeneration()
{
	return currentGen;
}

void CellularAutomaton::reset()
{
	int cIndex;
    for(int i=0; i<height+2; i++)
    {
        for(int j=0; j<width+2; j++)
        {
			cIndex = i*(width+2)+j;

            if(i==0 || j==0 || i==height+1 || j==width+1)
				world[cIndex] = false;
            else
            {
                if(rand()%100 <= COVER_PERCENT)
                    world[cIndex] = false;
                else
                    world[cIndex] = true;
            }
        }
    }
    currentGen = 0;

	updateDevice();
}

void CellularAutomaton::setCell(int x, int y, bool state)
{
	updateHost();
	if(x>0 && y>0 && x<height+1 && y<width+1)
		world[(y+1)*(width+2)+x+1] = state;

	updateDevice();
}

void CellularAutomaton::clear(bool state)
{
	int cIndex;
    for(int i=0; i<height+2; i++)
        for(int j=0; j<width+2; j++)
		{
			cIndex = i*(width+2)+j;
            world[cIndex] = state;
		}
		
    currentGen = 0;
	
	updateDevice();
}

void CellularAutomaton::draw(sf::RenderWindow &window)
{
	updatePixels();
    texture->update(pixels);
	sprite.setTextureRect(sf::IntRect(0,0, width, height));
    window.draw(sprite);
}

__global__ void computeCells(bool* world, bool* nextWorld, int w, int h) 
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int x = i/(w+2), y=i%(w+2);
	if(i>=(w+2)*(h+2)){}
	else if(x<=0 || y<=0 || x>=h+1 || y>=w+1)
		nextWorld[i] = false;
	else
	{
		// Somme des cellules voisines
		int nSum = 0;

		//neighbourgSum = neighbourgSum << world[i][j] + world[i][j];
		nSum = world[i-(w+2)-1] ? 1 : 0;
		nSum = world[i-(w+2)] ? (nSum? nSum<<1:1) : nSum;
		nSum = world[i-(w+2)+1] ? (nSum? nSum<<1:1) : nSum;
		nSum = world[i-1] ? (nSum? nSum<<1:1) : nSum;
		nSum = world[i+1] ? (nSum? nSum<<1:1) : nSum;
		nSum = world[i+(w+2)-1] ? (nSum? nSum<<1:1) : nSum;
		nSum = world[i+(w+2)] ? (nSum? nSum<<1:1) : nSum;
		nSum = world[i+(w+2)+1] ? (nSum? nSum<<1:1) : nSum;

		// Regle du "Jeu de Vie"
		//Born if 3 neighbours
		if(nSum & RULE_B && !world[i])
			nextWorld[i] = true;
		else if(nSum & RULE_S)
			nextWorld[i] = world[i];
		else
			nextWorld[i] = false;
	}
}

__global__ void swapCell(bool* world, bool* nextWorld, sf::Uint8* pixels, int w, int h)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i<(w+2)*(h+2))
	{
		world[i] = nextWorld[i];
	}
}

__global__ void worldToPixels(bool* world, sf::Uint8* pixels, int w, int h)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int x = i/(w+2), y=i%(w+2);

	if(x>0 && y>0 && x<h+1 && y<w+1)
	{
		x--; y--;
		int color = 255 * !world[i];

		pixels[(x * w + y) * 4]     = color; // R?
		pixels[(x * w + y) * 4 + 1] = color; // G?
		pixels[(x * w + y) * 4 + 2] = color; // B?
		pixels[(x * w + y) * 4 + 3] = 255; // A?
	}
}

void CellularAutomaton::nextStep()
{
	int blockSize = CUDA_BLOCK_SIZE;
	int nbBlock = cellCount()/blockSize;    // The actual grid size needed, based on input size

	computeCells<<<nbBlock, blockSize>>>(d_w, d_nW, width, height);
	cudaDeviceSynchronize();

	swapCell<<<nbBlock, blockSize>>>(d_w, d_nW, d_pixels, width, height);
	cudaDeviceSynchronize();

	currentGen++;
}

void CellularAutomaton::updateHost()
{
	cudaMemcpy(world, d_w, cellCount() * sizeof(bool), cudaMemcpyDeviceToHost);
}

void CellularAutomaton::updateDevice()
{
	cudaMemcpy(d_w, world, cellCount() * sizeof(bool), cudaMemcpyHostToDevice);
}

void CellularAutomaton::updatePixels()
{
	int blockSize = CUDA_BLOCK_SIZE;
	int nbBlock = cellCount()/blockSize;

	worldToPixels<<<nbBlock, blockSize>>>(d_w, d_pixels, width, height);
	cudaDeviceSynchronize();

	cudaMemcpy(pixels, d_pixels, pixelsCount() * sizeof(bool), cudaMemcpyDeviceToHost);
}

int CellularAutomaton::cellCount()
{
	return (width+2)*(height+2);
}

int CellularAutomaton::pixelsCount()
{
	return width*height*4;
}