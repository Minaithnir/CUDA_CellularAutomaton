#pragma once

#include <SFML\Graphics.hpp>
#include <iostream>


class Pattern
{
public:
	Pattern(void);
	virtual ~Pattern(void);

	void setGrid(bool* grid,unsigned int width,unsigned int height);
	bool* getGrid(void);
	sf::Vector2i getSize(void);

	void loadFromFile(std::string filename);

	void draw(sf::RenderTarget &target, sf::Vector2i position);
protected:
	bool* grid;

	sf::Texture texture;
	sf::Sprite sprite;
};

