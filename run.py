import pygame
from pygame.locals import *
from constants import *
from pacman import Pacman
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

    def startGame(self):
        self.setBackground()
        self.mazesprites = MazeSprites("maze1.txt", "maze1_rotation.txt")
        self.background = self.mazesprites.constructBackground(self.background, self.level%5)
        self.nodes = NodeGroup("maze1.txt")
        self.nodes.setPortalPair((0,17), (27,17))
        homekey = self.nodes.createHomeNodes(11.5, 14)
        self.nodes.connectHomeNodes(homekey, (12,14), LEFT)
        self.nodes.connectHomeNodes(homekey, (15,14), RIGHT)
        self.pacman = Pacman(self.nodes.getNodeFromTiles(15, 26))
        self.pellets = PelletGroup("maze1.txt")
        self.ghosts = GhostGroup(self.nodes.getStartTempNode(), self.pacman)
        self.ghosts.blinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 0+14))
        self.ghosts.pinky.setStartNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.ghosts.inky.setStartNode(self.nodes.getNodeFromTiles(0+11.5, 3+14))
        self.ghosts.clyde.setStartNode(self.nodes.getNodeFromTiles(4+11.5, 3+14))
        self.ghosts.setSpawnNode(self.nodes.getNodeFromTiles(2+11.5, 3+14))
        self.nodes.denyHomeAccess(self.pacman)
        self.nodes.denyHomeAccessList(self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, LEFT, self.ghosts)
        self.nodes.denyAccessList(2+11.5, 3+14, RIGHT, self.ghosts)
        self.ghosts.inky.startNode.denyAccess(RIGHT, self.ghosts.inky)
        self.ghosts.clyde.startNode.denyAccess(LEFT, self.ghosts.clyde)
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
        if not self.pause.paused:
            self.ghosts.update(dt)
            if self.fruit is not None:
                self.fruit.update(dt)
            self.checkPelletEvents()
            self.checkGhostEvents()
            self.checkFruitEvents()

        if self.pacman.alive:
            if not self.pause.paused:
                #self.pause.setPause()
                #emulating needing to do the calculations every time.
                #need to keep compute time to sub 100 ms (shouldnt be too bad)
                #largecalculations()
                #self.pause.setPause()
                self.pacman.update(dt)
        else:
            self.pacman.update(dt)

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
                    # do simulator instead of other thing.
                    
                    self.textgroup.showText(self.sco_id)
                    #sco = run_pacman_simulation()
                    
                    self.textgroup.updateText(self.sco_id, "Score: " + str(self.sco))
                    self.sco += 100
                    

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
            
            self.ghosts.inky.startNode.allowAccess(RIGHT, self.ghosts.inky) 
            self.ghosts.clyde.startNode.allowAccess(LEFT, self.ghosts.clyde)
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