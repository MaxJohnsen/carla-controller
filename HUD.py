import pygame


class InfoBox:
    def __init__(self, size, bg_color=(255, 255, 255, 100), text_color=(0, 0, 0)):
        pygame.font.init()
        self.size = size
        self._bg_color = bg_color
        self._text_color = text_color
        self._content = {}
        self._label_font = pygame.font.SysFont("Tahoma", 12, bold=False)
        self._value_font = pygame.font.SysFont("Tahoma", 12, bold=True)
        self._text_gap = 10

    def update_content(self, items):
        for label, value in items:
            self._content[label] = value

    def render_surface(self):
        rel_y = (self.size[1] - (16 * len(self._content))) / 2
        surface = pygame.Surface(self.size, pygame.SRCALPHA)
        surface.fill(self._bg_color)
        for label, value in self._content.items():
            label = self._label_font.render(label, False, self._text_color)
            value = self._value_font.render(str(value), False, self._text_color)
            label_x = (self.size[0] / 2) - self._text_gap - label.get_width()
            value_x = (self.size[0] / 2) + self._text_gap
            surface.blit(label, (label_x, rel_y))
            surface.blit(value, (value_x, rel_y))
            rel_y += label.get_height()

        return surface
