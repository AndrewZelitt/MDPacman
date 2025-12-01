To setup you need to in the top bar of vscode type >python and find the "set interpreter setting" from there you set it to the formal methods interpreter and make sure that pygame is also in your environment.



Base Pacman code from pacmancode.com level 6



You will be given the position on the grid where each pellet is, each ghost is, where pacman is, where each fruit is, 
and where each power pellet is. You will also get whether the pacman is in powermode or whatever it is called. With this you can get calculate the ideal next move, I think it makes sense to only calculate the reward maps and states individually based on the current state of the automata rather than calculating the full product MDP as that will save on compute as it needs to happen on each time step. 

positions will be rounded to the nearest whole to make it easier.


Pacman movement options, are: stop, up, down, left, right

You will get the list of ghosts that are not in EYE mode which is when they arent a threat, otherwise it just gives there position at all times.