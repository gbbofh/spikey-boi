import pygame as pg


class QuadTree():

    def __init__(self, items, rect, depth=1,color=(200,50,100)):
        rect = pg.Rect(rect)
        self.bounding_rect = rect

        self.items = []
        self.spanning_items = []

        self.color = color
        self.parent = None

        self.ne = None
        self.se = None
        self.nw = None
        self.sw = None

        if depth > 0:
            self.rebuild(items, rect, depth)

    def rebuild(self, items, rect, depth):
        rect = pg.Rect(rect)

        self.items = []
        self.spanning_items = []

        self.ne = None
        self.se = None

        self.nw = None
        self.sw = None

        cx, cy = rect.center

        self.cx = cx
        self.cy = cy

        if depth <= 0 or not items:
            self.items = items
            return

        depth -= 1

        items_ne = []
        items_se = []
        items_nw = []
        items_sw = []

        for item in items:
            fits_in_ne = item.rect.x >= cx and item.rect.y < cy and item.rect.right <= rect.right and item.rect.bottom <= cy
            fits_in_se = item.rect.x >= cx and item.rect.y >= cy and item.rect.right <= rect.right and item.rect.bottom <= rect.bottom
            fits_in_nw = item.rect.x < cx and item.rect.y < cy and item.rect.right <= cx and item.rect.bottom <= cy
            fits_in_sw = item.rect.x < cx and item.rect.y >= cy and item.rect.right <= cx and item.rect.bottom <= rect.bottom

            # If the item fits fully within a single child node, add it to the corresponding list
            if fits_in_ne:
                items_ne.append(item)
            elif fits_in_se:
                items_se.append(item)
            elif fits_in_nw:
                items_nw.append(item)
            elif fits_in_sw:
                items_sw.append(item)
            else:
                # Item spans multiple quadrants or the entire screen
                self.spanning_items.append(item)

            # in_ne = item.rect.x >= cx and item.rect.y < cy
            # in_se = item.rect.x >= cx and item.rect.y >= cy
            # in_nw = item.rect.x <= cx and item.rect.y < cy
            # in_sw = item.rect.x <= cx and item.rect.y >= cy

            # in_self = in_ne and in_se and in_nw and in_sw

            # if in_self:
            #     self.items.append(item)
            # else:
            #     if in_ne: items_ne.append(item)
            #     if in_se: items_se.append(item)
            #     if in_nw: items_nw.append(item)
            #     if in_sw: items_sw.append(item)

        width = rect.width // 2
        height = rect.height // 2

        if items_ne:
            self.ne = QuadTree(items_ne, (cx, rect.top, width, height), depth, color=(50, 100, 200))
            self.ne.parent = self
        if items_se:
            self.se = QuadTree(items_se, (cx, cy, width, height), depth, color=(50, 200, 100))
            self.se.parent = self
        if items_nw:
            self.nw = QuadTree(items_nw, (rect.left, rect.top, width, height), depth, color=(200, 100, 200))
            self.nw.parent = self
        if items_sw:
            self.sw = QuadTree(items_sw, (rect.left, cy, width, height), depth, color=(200, 200, 100))
            self.sw.parent = self

    def get_container_of(self, item):
        """
        Returns the QuadTree node that contains the given item.
        
        :param item: The item to search for (expected to have a `rect` attribute).
        :return: The QuadTree node containing the item, or None if not found.
        """
        # If this is a leaf node, check if the item exists here
        if not any([self.ne, self.se, self.nw, self.sw]):
            return self if item in self.items else None

        # Determine which child node might contain the item
        cx, cy = self.bounding_rect.center
        if hasattr(item, "rect"):
            if item.rect.x >= cx and item.rect.y < cy and self.ne:
                return self.ne.get_container_of(item)
            if item.rect.x >= cx and item.rect.y >= cy and self.se:
                return self.se.get_container_of(item)
            if item.rect.x < cx and item.rect.y < cy and self.nw:
                return self.nw.get_container_of(item)
            if item.rect.x < cx and item.rect.y >= cy and self.sw:
                return self.sw.get_container_of(item)

        # If the item doesn't fit in any child node, it might be here
        return self if item in self.items else None

        # if item in self.items:
        #     return self

        # in_ne = self.ne.get_container_of(item) if self.ne else None
        # if in_ne:
        #     return in_ne

        # in_se = self.ne.get_container_of(item) if self.ne else None
        # if in_se:
        #     return in_se

        # in_nw = self.ne.get_container_of(item) if self.ne else None
        # if in_nw:
        #     return in_nw

        # in_sw = self.ne.get_container_of(item) if self.ne else None
        # if in_sw:
        #     return in_sw

    def hit(self, rect, exclude=None):
        # Find the hits at the current level.
        rect = pg.Rect(rect)
        hits = set([self.items[n] for n in rect.collidelistall(self.items)])
        hits |= set([self.spanning_items[n] for n in rect.collidelistall(self.spanning_items)])
        
        # Recursively check the lower quadrants.
        if self.nw and rect.left < self.cx and rect.top < self.cy:
            hits |= self.nw.hit(rect)
        if self.sw and rect.left < self.cx and rect.bottom < self.cy:
            hits |= self.sw.hit(rect)
        if self.ne and rect.right >= self.cx and rect.top >= self.cy:
            hits |= self.ne.hit(rect)
        if self.se and rect.right >= self.cx and rect.bottom >= self.cy:
            hits |= self.se.hit(rect)

        if exclude:
            hits.discard(exclude)

        return hits

    def debug_draw(self, surface):
        #if self.ne:
        #    self.ne.debug_draw(surface)
        #if self.se:
        #    self.se.debug_draw(surface)
        #if self.nw:
        #    self.nw.debug_draw(surface)
        #if self.sw:
        #    self.sw.debug_draw(surface)

        #pg.draw.rect(surface, self.color, self.bounding_rect, 2)
        pass

