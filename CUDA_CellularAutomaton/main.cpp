#include "CellularAutomaton.hpp"

#include <SFML/Graphics.hpp>

#include <iostream>
#include <math.h>

#define WINDOW_W 800
#define WINDOW_H 600

#define ZOOM_FRACTION 10
#define SCROLL_WIDTH 50
#define SCROLL_SPEED 50

#define FRAME_TIME (1.f/30)
#define GREY 200

int main()
{
    srand(static_cast<unsigned int> (time(NULL)));

    // Create the main window
    sf::RenderWindow App(sf::VideoMode(WINDOW_W, WINDOW_H), "Game of Life");
	App.setFramerateLimit(60);
    sf::Clock clock;
    float elapsedTime=0;

    //Game of life;
    CellularAutomaton gameOfLife(WINDOW_W, WINDOW_H);
	int gen=0;
    bool pausedGame = true;

    //View (pour zoom)
	sf::View mainView = App.getDefaultView();
    sf::View zoomView;
    zoomView.setSize(WINDOW_W / ZOOM_FRACTION, WINDOW_H / ZOOM_FRACTION);
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
				switch(event.key.code)
				{
				case sf::Keyboard::Space : // Pause simulation
					if(event.key.code == sf::Keyboard::Space)
					{
						if(pausedGame)
							pausedGame = false;
						else
							pausedGame = true;
					}
					break;
				case sf::Keyboard::Z : // Toggle zoom
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
					break;
				case sf::Keyboard::R : // reset grid with random cell states
                    gameOfLife.reset();
					break;
				case sf::Keyboard::C: // clear all living cells
                    gameOfLife.clear(false);
					break;
				case sf::Keyboard::Right :
					if(pausedGame)
					{
						if(event.key.control)
							for(unsigned int i=0; i<10; i++)
								gameOfLife.nextStep();
						else
							gameOfLife.nextStep();
					}
					break;
				default :
					break;
				}
            }
            if(event.type == sf::Event::MouseMoved)
            {
                if(!zoom)
                {
					sf::Vector2f mousePos =  App.mapPixelToCoords(sf::Mouse::getPosition(App));

					mousePos.x = floor(mousePos.x);
					mousePos.y = floor(mousePos.y);
                    zoomView.setCenter(mousePos);
                }
            }

            // Change cell state according to mouse button
            if(event.type == sf::Event::MouseButtonReleased && zoom)
            {
                sf::Vector2f mousePos =  App.mapPixelToCoords(sf::Mouse::getPosition(App));

                mousePos.x = floor(mousePos.x);
                mousePos.y = floor(mousePos.y);

                if(event.mouseButton.button == sf::Mouse::Left)
                    gameOfLife.setCell((int) mousePos.x, (int) mousePos.y, true);
                if(event.mouseButton.button == sf::Mouse::Right)
                    gameOfLife.setCell((int) mousePos.x, (int) mousePos.y, false);
            }

			if(event.type == sf::Event::Resized)
			{
				mainView.setSize((float)App.getSize().x, (float)App.getSize().y);
				mainView.setCenter(App.getSize().x/2.f, App.getSize().y/2.f);
				zoomView.setSize((float)App.getSize().x / ZOOM_FRACTION, (float)App.getSize().y / ZOOM_FRACTION);
				gameOfLife.resize(App.getSize().x, App.getSize().y);
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
		else
			App.setView(mainView);


        if(!pausedGame)
            gameOfLife.nextStep();

        App.clear(sf::Color(GREY,GREY,GREY));

		gameOfLife.draw(App);

		App.display();
		
		elapsedTime = clock.getElapsedTime().asSeconds();
		clock.restart();
    }

    return EXIT_SUCCESS;
}

