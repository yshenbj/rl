# Example file showing a circle moving on screen
import pygame

# pygame setup
pygame.init()
screen = pygame.display.set_mode((512, 512))
clock = pygame.time.Clock()
running = True
dt = 0

player_pos = pygame.Vector2(screen.get_width() / 2, screen.get_height() / 2)
pix_square_size = 512 / 3
while running:
    # poll for events
    # pygame.QUIT event means the user clicked X to close your window
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # fill the screen with a color to wipe away anything from last frame
    screen.fill((0, 0, 0))

    pygame.draw.aaline(
        screen, 
        (255, 255, 255),
        (pix_square_size * 1,  pix_square_size * 0),
        (pix_square_size * 1,  pix_square_size * 3)
    )
    pygame.draw.aaline(
        screen, 
        (255, 255, 255),
        (pix_square_size * 2,  pix_square_size * 0),
        (pix_square_size * 2,  pix_square_size * 3)
    )
    pygame.draw.aaline(
        screen, 
        (255, 255, 255),
        (pix_square_size * 0,  pix_square_size * 1),
        (pix_square_size * 3,  pix_square_size * 1)
    )
    pygame.draw.aaline(
        screen, 
        (255, 255, 255),
        (pix_square_size * 0,  pix_square_size * 2),
        (pix_square_size * 3,  pix_square_size * 2)
    )
    
    pygame.draw.circle(
        screen, 
        (255, 255, 255), 
        (pix_square_size * 0.5,  pix_square_size * 0.5),
        pix_square_size / 2 - 5
    )
    pygame.draw.circle(
        screen, 
        (0, 0, 0), 
        (pix_square_size * 0.5,  pix_square_size * 0.5),
        pix_square_size / 2 - 10
    )
    
    pygame.draw.line(
        screen, 
        (255, 255, 255),
        (pix_square_size * 1 + 5,  pix_square_size * 1 + 5),
        (pix_square_size * 2 - 5,  pix_square_size * 2 - 5),
        7
    )
    pygame.draw.line(
        screen, 
        (255, 255, 255),
        (pix_square_size * 2 - 5,  pix_square_size * 1 + 5),
        (pix_square_size * 1 + 5,  pix_square_size * 2 - 5),
        7
    )
    
    pygame.draw.circle(screen, "red", player_pos, 5)
    player_pos = pygame.mouse.get_pos()

    # flip() the display to put your work on screen
    pygame.display.flip()

    # limits FPS to 60
    # dt is delta time in seconds since last frame, used for framerate-
    # independent physics.
    clock.tick(300)

pygame.quit()