import pygame
import MDPstuff.MDP as MDP
import tkinter as tk

# Create a Tkinter root window 
root = tk.Tk()

# Get the screen height in pixels
screen_height = root.winfo_screenheight()
screen_height = screen_height - 150
screen_width = screen_height


root.destroy()

pygame.init()
 
# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
 
PI = 3.141592653
 
# Set the height and width of the screen
size = (screen_width, screen_height)
screen = pygame.display.set_mode(size)
 
pygame.display.set_caption("MDPacman")
 
# Loop until the user clicks the close button.
done = False
clock = pygame.time.Clock()

score = 10000
 
# Loop as long as done == False
while not done:
 
    for event in pygame.event.get():  # User did something
        if event.type == pygame.QUIT:  # If user clicked close
            done = True  # Flag that we are done so we exit this loop
 
    # All drawing code happens after the for loop and but
    # inside the main while not done loop.
 
    # Clear the screen and set the screen background
    screen.fill(WHITE)
    background_image = pygame.image.load("Originalpacmaze.webp").convert()
    background_image = pygame.transform.scale(background_image, (screen_height, screen_height))
    screen.blit(background_image, (0, 0))
    
    # Select the font to use, size, bold, italics
    font = pygame.font.SysFont('Calibri', 25, True, False)
 
    # Render the text. "True" means anti-aliased text.
    # Black is the color. This creates an image of the
    # letters, but does not put it on the screen
    text = font.render("MDPacman", True, WHITE)
 
    
    screen.blit(text, [screen_height/2 - text.get_width()/2, 0])

    instruct = font.render("Press Space to Start", True, WHITE)

    screen.blit(instruct, [0, 0])

    score = score + 1


    tex = "Score: " + str(score)


    score_text = font.render(tex, True, WHITE)
    
        
    screen.blit(score_text, [3*screen_height/4 - score_text.get_width()/2, 0])

    
 
    # Go ahead and update the screen with what we've drawn.
    # This MUST happen after all the other drawing commands.
    pygame.display.flip()
 
    # This limits the while loop to a max of 60 times per second.
    # Leave this out and we will use all CPU we can.
    clock.tick(1)
 
# Be IDLE friendly
pygame.quit()