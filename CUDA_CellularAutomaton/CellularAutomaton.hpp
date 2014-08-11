#pragma once

#include "Pattern.hpp"

#include <SFML/Graphics.hpp>

#define WORLD_W 1000
#define WORLD_H 1000
#define CELL_COUNT (WORLD_H+2)*(WORLD_W+2)

#define PIXELS_SIZE WORLD_W*WORLD_H*4

#define COVER_PERCENT 50
#define RULE_S 6
#define RULE_B 4

#define CUDA_BLOCK_SIZE 1024

class CellularAutomaton
{
public:
	CellularAutomaton(void);
	virtual ~CellularAutomaton(void);

	void reset();
	void nextStep();
	void setCell(unsigned int x, unsigned int y, bool state);
	void setGrid(Pattern &pattern, sf::Vector2i position);
	void clear(bool state);

	void draw(sf::RenderWindow &window);

	int getGeneration();

protected:
	void updateHost();
	void updateDevice();
	void updatePixels();

	bool world[CELL_COUNT];

	bool* d_w;
	bool* d_nW;
	sf::Uint8* d_pixels;
	
	unsigned int currentGen;

    sf::Sprite sprite;
    sf::Texture texture;
    sf::Uint8 pixels[PIXELS_SIZE];
};

