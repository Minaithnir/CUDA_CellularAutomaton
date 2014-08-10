#include "CellularAutomaton.hpp"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <time.h>


CellularAutomaton::CellularAutomaton(void)
{
	cudaMalloc((void**)&d_w, CELL_COUNT * sizeof(int));
	cudaMalloc((void**)&d_nW, CELL_COUNT * sizeof(int));
	cudaMalloc((void**)&d_pixels, PIXELS_SIZE * sizeof(sf::Uint8));

    texture.create(WORLD_W, WORLD_H);
    texture.setSmooth(false);
    sprite.setTexture(texture);

	reset();
}


CellularAutomaton::~CellularAutomaton(void)
{
	cudaFree(d_w);
	cudaFree(d_nW);
}

int CellularAutomaton::getGeneration()
{
	return currentGen;
}

void CellularAutomaton::reset()
{
	int cIndex;
    for(unsigned int i=0; i<WORLD_H+2; i++)
    {
        for(unsigned int j=0; j<WORLD_W+2; j++)
        {
			cIndex = i*(WORLD_W+2)+j;

            if(i==0 || j==0 || i==WORLD_H+1 || j==WORLD_W+1)
				world[cIndex] = 0;
            else
            {
                if(rand()%100 <= COVER_PERCENT)
                    world[cIndex] = 0;
                else
                    world[cIndex] = 1;
            }
        }
    }
    currentGen = 0;

	updateDevice();
}

void CellularAutomaton::setCell(unsigned int x, unsigned int y, bool state)
{
	//updateHost();
	if(x>0 && y>0 && x<WORLD_H+1 && y<WORLD_W+1)
		world[(y+1)*(WORLD_W+2)+x+1] = state?1:0;

	updateDevice();
}

void CellularAutomaton::clear(bool state)
{
	int cIndex;
    for(unsigned int i=0; i<WORLD_H+2; i++)
        for(unsigned int j=0; j<WORLD_W+2; j++)
		{
			cIndex = i*(WORLD_W+2)+j;
            world[cIndex] = state?1:0;
		}
		
    currentGen = 0;
	
	updateDevice();
}

void CellularAutomaton::draw(sf::RenderWindow &window)
{
	updatePixels();
    texture.update(pixels);

    window.draw(sprite);
}

__global__ void computeCell(int* world, int* nextWorld) 
{
	int i = threadIdx.x + blockIdx.x * blockDim.x; 
	int x = i/(WORLD_W+2), y=i%(WORLD_W+2);
	if(i>=CELL_COUNT){}
	else if(x<=0 || y<=0 || x>=WORLD_H+1 || y>=WORLD_W+1)
		nextWorld[i] = 0;
	else
	{
		// Somme des cellules voisines
		int nSum = 0;

		//neighbourgSum = neighbourgSum << world[i][j] + world[i][j];
		nSum = world[i-(WORLD_W+2)-1] ? 1 : 0;
		nSum = world[i-(WORLD_W+2)] ? (nSum? nSum<<1:1) : nSum;
		nSum = world[i-(WORLD_W+2)+1] ? (nSum? nSum<<1:1) : nSum;
		nSum = world[i-1] ? (nSum? nSum<<1:1) : nSum;
		nSum = world[i+1] ? (nSum? nSum<<1:1) : nSum;
		nSum = world[i+(WORLD_W+2)-1] ? (nSum? nSum<<1:1) : nSum;
		nSum = world[i+(WORLD_W+2)] ? (nSum? nSum<<1:1) : nSum;
		nSum = world[i+(WORLD_W+2)+1] ? (nSum? nSum<<1:1) : nSum;

		// Regle du "Jeu de Vie"
		//Born if 3 neighbours
		if(world[i])
			world[i] = world[i]<255?world[i]+1:255;

		if(nSum & RULE_B)
			nextWorld[i] = world[i]?world[i]:1;
		else if(nSum & RULE_S)
			nextWorld[i] = world[i];
		else
			nextWorld[i] = false;
	}
}

__global__ void swapCells(int* world, int* nextWorld, sf::Uint8* pixels)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i<CELL_COUNT)
	{
		world[i] = nextWorld[i];
	}
}

__global__ void pixelsToHost(int* world, sf::Uint8* pixels)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int x = i/(WORLD_W+2), y=i%(WORLD_W+2);

	if(x>0 && y>0 && x<WORLD_H+1 && y<WORLD_W+1)
	{
		x--; y--;

		if(world[i])
		{			
			pixels[(x * WORLD_W + y) * 4]     = world[i]; // R
			pixels[(x * WORLD_W + y) * 4 + 1] = 255-world[i]; // G
			pixels[(x * WORLD_W + y) * 4 + 2] = 0; // B
		}
		else
		{
			pixels[(x * WORLD_W + y) * 4]     = 255; // R
			pixels[(x * WORLD_W + y) * 4 + 1] = 255; // G
			pixels[(x * WORLD_W + y) * 4 + 2] = 255; // B
		}
		pixels[(x * WORLD_W + y) * 4 + 3] = 255; // A
	}
}

void CellularAutomaton::nextStep()
{
	int blockSize = CUDA_BLOCK_SIZE;
	int nbBlock = CELL_COUNT/blockSize;    // The actual grid size needed, based on input size

	computeCell<<<nbBlock, blockSize>>>(d_w, d_nW);
	cudaDeviceSynchronize();

	swapCells<<<nbBlock, blockSize>>>(d_w, d_nW, d_pixels);
	cudaDeviceSynchronize();
	
	updateHost();

	currentGen++;
}

void CellularAutomaton::updateHost()
{
	cudaMemcpy(world, d_nW, CELL_COUNT * sizeof(int), cudaMemcpyDeviceToHost);
}

void CellularAutomaton::updateDevice()
{
	cudaMemcpy(d_w, world, CELL_COUNT * sizeof(int), cudaMemcpyHostToDevice);
}

void CellularAutomaton::updatePixels()
{
	int blockSize = CUDA_BLOCK_SIZE;
	int nbBlock = CELL_COUNT/blockSize;

	pixelsToHost<<<nbBlock, blockSize>>>(d_w, d_pixels);
	cudaDeviceSynchronize();

	cudaMemcpy(pixels, d_pixels, PIXELS_SIZE * sizeof(sf::Uint8), cudaMemcpyDeviceToHost);
}