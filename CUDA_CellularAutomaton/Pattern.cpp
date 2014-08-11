#include "Pattern.hpp"

#include <fstream>

Pattern::Pattern(void)
{
	bool bg[] = {0,1,0,0,0,1,1,1,1};
	grid = NULL;
	texture.setSmooth(false);
	texture.setRepeated(false);
	setGrid(bg , 3, 3);
}

Pattern::~Pattern(void)
{
	if(grid != NULL)
		delete[] grid;
}

void Pattern::setGrid(bool* g,unsigned int width,unsigned int height)
{
	if(grid != NULL)
		delete[] grid;
	
	grid = new bool[width*height];
	sf::Uint8* pixels = new sf::Uint8[width*height];
	for(unsigned int i=0; i<width*height; i++)
	{
		grid[i] = g[i];
		
		pixels[i*4]     = g[i]?0:255; // R?
		pixels[i*4 + 1] = 255; // G?
		pixels[i*4 + 2] = g[i]?0:255; // B?
		pixels[i*4 + 3] = 255; // A?
	}
	
	texture.create(width, height);
	texture.update(pixels);
	sprite.setTexture(texture);
}

bool* Pattern::getGrid(void)
{
	return grid;
}

sf::Vector2i Pattern::getSize(void)
{
	return sf::Vector2i(texture.getSize().x, texture.getSize().y);
}

void Pattern::loadFromFile(std::string filename)
{
	std::ifstream file;
	file.open(filename, std::ios::in);
	if(file.is_open())
	{
		std::cout << file.get() << std::endl;
		file.close();
	}
}

void Pattern::draw(sf::RenderTarget &target, sf::Vector2i position)
{
	sprite.setPosition((float)position.x, (float)position.y);
	target.draw(sprite);
}