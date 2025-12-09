import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
import random
from nodes import NodeGroup
from pellets import PelletGroup
from ghosts import GhostGroup
from fruit import Fruit
from pauser import Pause
from text import TextGroup
from sprites import LifeSprites
from sprites import MazeSprites
import pygame_widgets
from pygame_widgets.slider import Slider
from pygame_widgets.textbox import TextBox
from simulator import run_pacman_simulation
import time

class GameController(object):
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode(SCREENSIZE, 0, 32)
        self.background = None
        self.clock = pygame.time.Clock()
        self.fruit = None
        self.pause = Pause(True)
        self.level = 0
        self.lives = 1
        self.score = 0
        self.sco = 0
        self.textgroup = TextGroup()
        self.lifesprites = LifeSprites(self.lives)
        # MDP step-mode controls (used by the `S` key to step simulation one move at a time)
        self._mdp_step_active = False
        self._mdp_nodes = None
        self._mdp_pacman = None
        self._mdp_pellets = None
        self._mdp_ghosts = None
        self._mdp_mdp = None
        self._mdp_bg = None
        self._mdp_max_moves = 50
        self._mdp_moves_done = 0
        self._mdp_score = 0

    def restartGame(self):
        self.lives = 1
        self.level = 0
        self.pause.paused = True
        self.fruit = None
        self.startGame()
        self.score = 0
        self.textgroup.updateScore(self.score)
        self.textgroup.updateLevel(self.level)
        self.textgroup.showText(READYTXT)
        self.lifesprites.resetLives(self.lives)

    def resetLevel(self):
        self.pause.paused = True
        self.pacman.reset()
        self.ghosts.reset()
        self.fruit = None
        self.textgroup.showText(READYTXT)

    def nextLevel(self):
        self.showEntities()
        self.level += 1
        self.pause.paused = True
        self.startGame()
        self.textgroup.updateLevel(self.level)

    def setBackground(self):
        self.background = pygame.surface.Surface(SCREENSIZE).convert()
        self.background.fill(BLACK)

    def _mdp_init_step_mode(self):
        """Initialize a fresh local game state for step-by-step MDP simulation."""
        from mdp_value_iteration import ValueIterationPacmanMDP
        from simulator import next_step

        nodes = NodeGroup("maze1.txt")
        nodes.setPortalPair((0, 17), (27, 17))
        pacman = Pacman(nodes.getNodeFromTiles(15, 26))
        pellets = PelletGroup("maze1.txt")
        ghosts = GhostGroup(nodes.getStartTempNode(), pacman)
        ghosts.randomizeSpawn(nodes, exclude_coords={(15.0, 26.0)})

        mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        bg = pygame.surface.Surface(SCREENSIZE).convert()
        bg = mazesprites.constructBackground(bg, self.level % 5)

        mdp = ValueIterationPacmanMDP(nodes)

        # store in controller state
        self._mdp_step_active = True
        self._mdp_nodes = nodes
        self._mdp_pacman = pacman
        self._mdp_pellets = pellets
        self._mdp_ghosts = ghosts
        self._mdp_mdp = mdp
        self._mdp_bg = bg
        self._mdp_moves_done = 0
        # reset local step-mode score
        self._mdp_score = 0

        # Render initial state
        self.screen.blit(self._mdp_bg, (0, 0))
        self._mdp_pellets.render(self.screen)
        self._mdp_pacman.render(self.screen)
        self._mdp_ghosts.render(self.screen)
        self.textgroup.render(self.screen)
        pygame.display.update()
        print("MDP step-mode initialized. Press S to advance one move.")

    def _mdp_step(self):
        """Advance one MDP move in the step-mode simulation."""
        from simulator import next_step

        if not self._mdp_step_active:
            return

        nodes = self._mdp_nodes
        pacman = self._mdp_pacman
        pellets = self._mdp_pellets
        ghosts = self._mdp_ghosts
        mdp = self._mdp_mdp

        if len(pellets.pelletList) == 0:
            print("No pellets left — ending MDP step-mode.")
            self._mdp_step_active = False
            return

        # Compute best action and Q-values
        try:
            best_action, q_list = mdp.compute_best_action_with_qs(pacman.node, pellets.pelletList, ghosts)
        except Exception:
            best_action = mdp.compute_best_action(pacman.node, pellets.pelletList, ghosts)
            q_list = []

        if q_list:
            qstr = ", ".join(f"{a}:{q:.1f}" for (a, q, _, _) in q_list)
        else:
            qstr = ""
        msg = f"Step {self._mdp_moves_done}: Pacman {pacman.node.coords} -> action={best_action}"
        print(msg + (f" | Qs: {qstr}" if qstr else ""))

        # Collect pellets at current node
        pellets_to_remove = []
        for pellet in pellets.pelletList:
            if pacman.node.coords == pellet.coord:
                pellets_to_remove.append(pellet)
                if pellet.name == POWERPELLET:
                    ghosts.startFreight()

        # update local score for pellets
        for pellet in pellets_to_remove:
            if pellet.name == POWERPELLET:
                self._mdp_score += getattr(pellet, 'points', 50)
            else:
                self._mdp_score += getattr(pellet, 'points', 10)

        for pellet in pellets_to_remove:
            pellets.pelletList.remove(pellet)

        # Check collisions (local step-mode handling)
        for ghost in ghosts:
            try:
                collided = pacman.collideGhost(ghost)
            except Exception:
                # fallback to node-equality
                collided = (pacman.node.coords == getattr(ghost.node, 'coords', None))

            if collided:
                if ghost.mode.current is FREIGHT:
                    # Pacman eats ghost in freight mode — mirror real game behavior
                    points = getattr(ghost, 'points', 200)
                    self._mdp_score += points
                    print(f"MDP step-mode: ate ghost for {points} points. total={self._mdp_score}")
                    # Hide ghost and mark it spawning (don't teleport it directly)
                    ghost.visible = False
                    ghosts.updatePoints()
                    try:
                        ghost.startSpawn()
                        nodes.allowHomeAccess(ghost)
                    except Exception:
                        # As a fallback, move ghost to a safe node away from Pacman
                        try:
                            ghost.node = nodes.getNodeFromTiles(2+11.5, 3+14)
                            ghost.setPosition()
                        except Exception:
                            pass
                else:
                    # Pacman dies in local simulation
                    print(f"MDP step-mode: Pacman died at {pacman.node.coords}. score={self._mdp_score}")
                    # print the final score for this move
                    print(f"Score: {self._mdp_score}")
                    self._mdp_step_active = False
                    return

        # Apply pacman move (node-step)
        next_node = pacman.node.neighbors.get(best_action)
        if next_node is not None:
            pacman.node = next_node
            pacman.setPosition()

        # Move ghosts one step
        for ghost in ghosts:
            if getattr(ghost, 'node', None) is None:
                continue
            next_ghost_node, _ = next_step(ghost.node, pacman.node, nodes)
            ghost.node = next_ghost_node
            ghost.setPosition()

        # Render frame
        self.screen.blit(self._mdp_bg, (0, 0))
        pellets.render(self.screen)
        pacman.render(self.screen)
        ghosts.render(self.screen)
        self.textgroup.render(self.screen)
        pygame.display.update()

        self._mdp_moves_done += 1
        if self._mdp_moves_done >= self._mdp_max_moves:
            print("Reached max step count — ending MDP step-mode.")
            self._mdp_step_active = False
        # print current score after the move
        print(f"Score: {self._mdp_score}")

    def startGame(self):
        self.setBackground()
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.background = self.mazesprites.constructBackground(self.background, self.level%5)
        self.nodes = NodeGroup("maze1.txt")
        self.nodes.setPortalPair((0,17), (27,17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12,14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15,14), RIGHT)
        # Randomize Pacman's start position among available nodes
        try:
            candidates = list(self.nodes.nodesLUT.values())
            start_node = random.choice(candidates)
        except Exception:
            start_node = self.nodes.getNodeFromTiles(15, 26)
        self.pacman = Pacman(start_node)
        self.pellets = PelletGroup("maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        # Randomize ghost spawn positions so they don't always start at the same spots
        self.ghosts.randomizeSpawn(self.nodes, exclude_coords={(15.0, 26.0)})
        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
        #self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        #self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
        self.nodes.denyAccessList(12, 14, UP, self.ghosts)
        self.nodes.denyAccessList(15, 14, UP, self.ghosts)
        self.nodes.denyAccessList(12, 26, UP, self.ghosts)
        self.nodes.denyAccessList(15, 26, UP, self.ghosts)
        self.slider = Slider(pygame.display.get_surface(), SCREENWIDTH/2 + 10, 5, 100, 10, min=1000, max=100000, step=1000)
        self.slidertxt = self.textgroup.addText("1000", WHITE, SCREENWIDTH/2 - 100, 0, 50)
        self.sco_id = self.textgroup.addText("Score: ", WHITE, SCREENWIDTH/2 - 100 , 20, 50)
        
        #output = TextBox(pygame.display, 0, 0, 50 , 50, fontSize=50, fontColor=WHITE)
        #output.setText("TextBox")
        #pygame.display.update()
        
        #output.disable()  # Act as label instead of textbox

    #this is the main loop of the program this is where a lot of code will go that will give information.
    def update(self):
        dt = self.clock.tick(30) / 1000.0
        #dt = self.clock.tick(1000) / 1000.0
        self.textgroup.update(dt)
        self.pellets.update(dt)

        # Update Pacman first so pellet pickup and mode changes happen before ghosts move.
        if self.pacman.alive:
            if not self.pause.paused:
                self.pacman.update(dt)
        else:
            self.pacman.update(dt)

        if not self.pause.paused:
            if self.fruit is not None:
                self.fruit.update(dt)
            # Handle pellet pickup immediately after Pacman moved; this can enable freight mode
            # before ghosts take their step (so Pacman can eat nearby ghosts).
            self.checkPelletEvents()

            # Now update ghosts (they will react to freight if started above)
            self.ghosts.update(dt)

            # Check ghost collisions after ghosts moved
            self.checkGhostEvents()
            self.checkFruitEvents()

        afterPauseMethod = self.pause.update(dt)
        if afterPauseMethod is not None:
            afterPauseMethod()
        self.checkEvents()
        self.render()
        """
        print("this is the powerpellet list")
        for pellet in self.pellets.pelletList:
            if pellet.name == POWERPELLET:
                print(pellet.coord)
        print("this is the pellet list")
        for pellet in self.pellets.pelletList:
            if pellet.name != POWERPELLET:
                print(pellet.coord)
        print("Ghost Positions:")
        for ghost in self.ghosts:
            print({round(ghost.position.x/TILEWIDTH)} , {round(ghost.position.y/TILEHEIGHT)})
        if self.fruit != None:
            print("Fruit Position:")
            print({self.fruit.position.x/TILEWIDTH} , {self.fruit.position.y/TILEHEIGHT})
       
        # powerpellets will have a higher reward normally than non power pellets, this will need to update each time.
        # use row and col to do calculations, you can just divide the values by the tilewidth and height to get the closest coordinate
        print(f"Pacman Position:\n {round(self.pacman.position.x/TILEWIDTH)} , {round(self.pacman.position.y/TILEHEIGHT)}")
        print("\n")
        """

    def updateScore(self, points):
        self.score += points
        self.textgroup.updateScore(self.score)

    def checkEvents(self):
        
        for event in pygame.event.get():
            pygame_widgets.update(event)
            self.textgroup.updateText(self.slidertxt, self.slider.getValue())
            if event.type == QUIT:
                exit()
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    if self.pacman.alive:
                        self.pause.setPause(playerPaused=True)
                        if not self.pause.paused:
                            self.textgroup.hideText()
                            self.showEntities()
                        else:
                            self.textgroup.showText(PAUSETXT)
                            self.hideEntities()
                elif event.key == K_s:
                    # Step-mode: initialize on first press, then each subsequent press advances one MDP move
                    if not self._mdp_step_active:
                        self._mdp_init_step_mode()
                    else:
                        self._mdp_step()
                elif event.key == K_k:
                    # Print some node info to console and show a short on-screen message
                    try:
                        total = len(self.nodes.nodesLUT)
                        sample = []
                        for i, nod in enumerate(self.nodes.nodesLUT):
                            if i >= 4:
                                break
                            sample.append(self.nodes.nodesLUT[nod].coords)
                        print(f"Nodes: {total}, sample coords: {sample}")
                        self.textgroup.addText(f"Nodes: {total}", WHITE, 0, 0, 18, time=3)
                    except Exception:
                        # Fallback to plain prints
                        for nod in self.nodes.nodesLUT:
                            print(nod, self.nodes.nodesLUT[nod].coords)
            elif event.type == MOUSEBUTTONDOWN:
                # If step-mode active, advance one step on left-click inside the game window
                if event.button == 1 and self._mdp_step_active:
                    self._mdp_step()

    def checkGhostEvents(self):
        for ghost in self.ghosts:
            if self.pacman.collideGhost(ghost):
                if ghost.mode.current is FREIGHT:
                    self.pacman.visible = False
                    ghost.visible = False
                    self.updateScore(ghost.points)
                    self.textgroup.addText(str(ghost.points), WHITE, ghost.position.x, ghost.position.y, 8, time=1)
                    self.ghosts.updatePoints()
                    self.pause.setPause(pauseTime=1, func=self.showEntities)
                    ghost.startSpawn()
                    self.nodes.allowHomeAccess(ghost)
                elif ghost.mode.current is not SPAWN:
                     if self.pacman.alive:
                         self.lives -=  1
                         self.lifesprites.removeImage()
                         self.pacman.die()
                         self.ghosts.hide()
                         if self.lives <= 0:
                             self.textgroup.showText(GAMEOVERTXT)
                             self.pause.setPause(pauseTime=3, func=self.restartGame)
                         else:
                             self.pause.setPause(pauseTime=3, func=self.resetLevel)

    def checkFruitEvents(self):
        if self.pellets.numEaten == 50 or self.pellets.numEaten == 140:
            if self.fruit is None:
                self.fruit = Fruit(self.nodes.getNodeFromTiles(9, 20))
        if self.fruit is not None:
            if self.pacman.collideCheck(self.fruit):
                self.updateScore(self.fruit.points)
                self.textgroup.addText(str(self.fruit.points), WHITE, self.fruit.position.x, self.fruit.position.y, 8, time=1)
                self.fruit = None
            elif self.fruit.destroy:
                self.fruit = None

    def checkPelletEvents(self):
        pellet = self.pacman.eatPellets(self.pellets.pelletList)
        if pellet:
            self.pellets.numEaten += 1
            self.updateScore(pellet.points)
            
            #self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky) 
            #self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
            self.pellets.pelletList.remove(pellet)
            if pellet.name == POWERPELLET:
               self.ghosts.startFreight()
            if self.pellets.isEmpty():
                self.hideEntities()
                self.pause.setPause(pauseTime=3, func=self.nextLevel)

    def showEntities(self):
        self.pacman.visible = True
        self.ghosts.show()

    def hideEntities(self):
        self.pacman.visible = False
        self.ghosts.hide()

    def render(self):
        # If MDP step-mode is active, render the step-mode frame instead of the live game
        if getattr(self, '_mdp_step_active', False):
            try:
                self.screen.blit(self._mdp_bg, (0, 0))
                if self._mdp_pellets is not None:
                    self._mdp_pellets.render(self.screen)
                if self._mdp_pacman is not None:
                    self._mdp_pacman.render(self.screen)
                if self._mdp_ghosts is not None:
                    self._mdp_ghosts.render(self.screen)
                self.textgroup.render(self.screen)
                pygame.display.update()
                return
            except Exception:
                # Fall back to normal render if anything goes wrong
                pass

        self.screen.blit(self.background, (0, 0))
        self.pellets.render(self.screen)
        if self.fruit is not None:
            self.fruit.render(self.screen)
        self.pacman.render(self.screen)
        self.ghosts.render(self.screen)
        self.textgroup.render(self.screen)
        for i in range(len(self.lifesprites.images)):
            x = self.lifesprites.images[i].get_width() * i
            y = SCREENHEIGHT - self.lifesprites.images[i].get_height()
            self.screen.blit(self.lifesprites.images[i], (x, y))
        events = pygame.event.get()
        pygame_widgets.update(events)
        pygame.display.update()
def largecalculations():
        start = time.perf_counter()
        for m in range(30):
            x = 0
            for i in range(1, 100000):
                x *= i
                x /= i
        end = time.perf_counter()

        print(f"overall time = {(end - start)}")
        return

    
if __name__ == "__main__":
    game = GameController()
    game.startGame()
    while True:
        game.update()