#include "CellularAutomaton.hpp"

#include <SFML/Graphics.hpp>

#include <iostream>
#include <math.h>

#define ZOOM_FRACTION 10
#define SCROLL_WIDTH 50
#define SCROLL_SPEED 50

#define FRAME_TIME (1.f/30)

int main()
{
    srand(static_cast<unsigned int> (time(NULL)));

    // Create the main window
    sf::RenderWindow App(sf::VideoMode(WORLD_W, WORLD_H), "Game of Life");
	App.setFramerateLimit(60);
    sf::Clock perfClock, clock;
    float elapsedTime=0, perfTime=0;

    //Game of life;
    CellularAutomaton gameOfLife;
	int gen=0;
    bool pausedGame = false;
	
	
	Pattern glider;
	bool patternMode = false;
	

    //View (pour zoom)
    sf::View zoomView;
    zoomView.setSize(WORLD_W / ZOOM_FRACTION, WORLD_H / ZOOM_FRACTION);
    bool zoom = false;

    // Start the game loop
    while (App.isOpen())
    {
        // Process events
        sf::Event event;
        while (App.pollEvent(event))
        {
            // Close window : exit
            if (event.type == sf::Event::Closed)
                App.close();
            // Simulation controls
            if (event.type == sf::Event::KeyPressed)
            {
                // Pause simulation
                if(event.key.code == sf::Keyboard::Space)
                {
                    if(pausedGame)
                        pausedGame = false;
                    else
                        pausedGame = true;
                }

                // Toggle zoom
                if(event.key.code == sf::Keyboard::Z)
                {
                    if(zoom)
                    {
                        zoom = false;
                        App.setView(App.getDefaultView());
                    }
                    else
                    {
                        zoom = true;
                        App.setView(zoomView);
                    }
                }

				if(event.key.code == sf::Keyboard::Num1)
                {
                    if(patternMode)
                        patternMode = false;
                    else
                        patternMode = true;
                }

                // Randomize each cell state
                if(event.key.code == sf::Keyboard::R)
                    gameOfLife.reset();

                // Kill all cells
                if(event.key.code == sf::Keyboard::C)
                    gameOfLife.clear(false);

                // Step by step (only when paused)
                if(event.key.code == sf::Keyboard::Right && pausedGame == true)
                {
                    if(event.key.control)
                        for(unsigned int i=0; i<10; i++)
                            gameOfLife.nextStep();
                    else
                        gameOfLife.nextStep();
                }
            }
            if(event.type == sf::Event::MouseMoved)
            {
                if(!zoom)
                {
                    zoomView.setCenter((float)event.mouseMove.x, (float)event.mouseMove.y);
                }
            }

            // Change cell state according to mouse button
            if(event.type == sf::Event::MouseButtonReleased)
            {
                sf::Vector2f mousePos =  App.mapPixelToCoords(sf::Mouse::getPosition(App));

                mousePos.x = floor(mousePos.x);
                mousePos.y = floor(mousePos.y);
				if(patternMode && event.mouseButton.button == sf::Mouse::Left)
				{
					gameOfLife.setGrid(glider, sf::Vector2i((int)mousePos.x, (int)mousePos.y));
				}
				else
				{
					if(event.mouseButton.button == sf::Mouse::Left)
						gameOfLife.setCell((int) mousePos.x, (int) mousePos.y, true);
					if(event.mouseButton.button == sf::Mouse::Right)
						gameOfLife.setCell((int) mousePos.x, (int) mousePos.y, false);
				}
                
            }
        }
		
        if(zoom)
        {
            sf::Vector2i mousePos =  sf::Mouse::getPosition(App);

            if(mousePos.x < SCROLL_WIDTH)
                zoomView.move(elapsedTime*(-SCROLL_SPEED), 0);
            else if(mousePos.x > (int)(App.getSize().x - SCROLL_WIDTH))
                zoomView.move(elapsedTime*(SCROLL_SPEED), 0);

            if(mousePos.y < SCROLL_WIDTH)
                zoomView.move(0, elapsedTime*(-SCROLL_SPEED));
            else if(mousePos.y > (int)(App.getSize().y - SCROLL_WIDTH))
                zoomView.move(0, elapsedTime*(SCROLL_SPEED));

            App.setView(zoomView);
        }

		
        if(!pausedGame)
            gameOfLife.nextStep();

        #define GREY 200
        App.clear(sf::Color(GREY,GREY,GREY));

		gen = gameOfLife.getGeneration();
		if(gen>0 && gen%1000 == 0)
		{
			perfTime = perfClock.getElapsedTime().asSeconds();
			perfClock.restart();
			std::cout << "generation : " << gen << " en " << perfTime << " soit " << perfTime/1000 << " par generation" << std::endl;
		}


		gameOfLife.draw(App);

		if(patternMode)
		{
            sf::Vector2f mousePos =  App.mapPixelToCoords(sf::Mouse::getPosition(App));
			glider.draw(App, sf::Vector2i((int)mousePos.x, (int)mousePos.y));
		}

		// Update the window
		App.display();

		elapsedTime = clock.getElapsedTime().asSeconds();
		clock.restart();
    }

    return EXIT_SUCCESS;
}