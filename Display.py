import cv2
import sdl2
import sdl2.ext
import sys
import time

class Display(object):
    def __init__(self, w, h):
        self.W, self.H = w, h
        sdl2.ext.init()
        self.window = sdl2.ext.Window("SLAM", size=(self.W, self.H), position=(sdl2.SDL_WINDOWPOS_CENTERED, sdl2.SDL_WINDOWPOS_CENTERED))
        self.window.show()

    def draw(self, img):
        surf = sdl2.ext.pixels3d(self.window.get_surface())
        surf[:, :, 0:3] = img.swapaxes(0, 1)
        self.window.refresh()

        for event in sdl2.ext.get_events():
            if event.type == sdl2.SDL_QUIT:
                exit(0)

        return