import pygame as pg


class QuadTree():

    def __init__(self, items, rect, depth=1,color=(200,50,100)):
        rect = pg.Rect(rect)
        self.bounding_rect = rect
        self.items = []
        self.color = color
        self.rebuild(items, rect, depth)

    def rebuild(self, items, rect, depth):
        rect = pg.Rect(rect)
        self.items = []

        self.ne = None
        self.se = None

        self.nw = None
        self.sw = None

        cx, cy = rect.center

        depth -= 1
        if depth == 0:
            self.items = items
            return

        items_ne = []
        items_se = []
        items_nw = []
        items_sw = []

        for item in items:
            in_ne = item.rect.x >= cx and item.rect.y < cy
            in_se = item.rect.x >= cx and item.rect.y >= cy
            in_nw = item.rect.x <= cx and item.rect.y < cy
            in_sw = item.rect.x <= cx and item.rect.y >= cy

            in_self = in_ne and in_se and in_nw and in_sw

            if in_self:
                self.items.append(item)
            else:
                if in_ne: items_ne.append(item)
                if in_se: items_se.append(item)
                if in_nw: items_nw.append(item)
                if in_sw: items_sw.append(item)

        width = rect.width // 2
        height = rect.height // 2

        if items_ne:
            self.ne = QuadTree(items_ne, (cx, rect.top, width, height), depth, color=(50, 100, 200))
        if items_se:
            self.se = QuadTree(items_se, (cx, cy, width, height), depth, color=(50, 200, 100))
        if items_nw:
            self.nw = QuadTree(items_nw, (rect.left, rect.top, width, height), depth, color=(200, 100, 200))
        if items_sw:
            self.sw = QuadTree(items_sw, (rect.left, cy, width, height), depth, color=(200, 200, 100))

    def debug_draw(self, surface):
        if self.ne:
            self.ne.debug_draw(surface)
        if self.se:
            self.se.debug_draw(surface)
        if self.nw:
            self.nw.debug_draw(surface)
        if self.sw:
            self.sw.debug_draw(surface)

        pg.draw.rect(surface, self.color, self.bounding_rect, 2)
