CUDA_CellularAutomaton
======================

CUDA_CellularAutomaton est une implementation en C++ du jeu de la vie de Conway.
Le traitement est effectuer sur le GPU grace à CUDA.
La partie graphique utilise la bibliothèque SFML.

Quelques controles ont été implémentés :

Espace : démarre ou met en pause la simulation R : Redemarre la simulation à zero en donnant un etat aléatoire à chaque cellule C : Redemarre la simulation avec plus aucune cellule vivante Z : Active / Desactive le zoom.

Lorsque le zoom est actif sur une zone, on peut utiliser la souris pour activer / desactiver une cellule individuellement.

Lorsque la simulation est en pause, la touche de droite permet de simuler les changements génération par génération.
