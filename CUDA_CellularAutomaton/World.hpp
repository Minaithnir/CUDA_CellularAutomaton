#ifndef WORLD_HPP
#define WORLD_HPP

#define WORLD_W 1000
#define WORLD_H WORLD_W

#define PIXELS_SIZE WORLD_H*WORLD_W*4

#define COVER_PERCENT 40
#define RULE_B 0x04 // B678
#define RULE_S 0x06 // S45678

#include <SFML/Graphics.hpp>

class World
{
    public:
        World();
        virtual ~World();

        void Reset();
        void ComputeNextStep();
        void Draw(sf::RenderWindow &App);

        void SetCell(int x, int y, bool state);

        void Clear(bool alive);

        void ToggleEpileptic();

    protected:

        bool world[WORLD_H+2][WORLD_W+2];
        bool nextWorld[WORLD_H+2][WORLD_W+2];

    private:

        bool ComputeCell(int x, int y);
        int generation;
        bool epileptic;

        sf::Clock clock;

        sf::Sprite sprite;
        sf::Texture texture;
        sf::Uint8 pixels[PIXELS_SIZE];
};

#endif // WORLD_HPP
