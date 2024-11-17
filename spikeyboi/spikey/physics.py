import pygame as pg


import spikeyboi.spikey


class Physics():

    def __init__(self, physics_group: pg.sprite.Group):
        self.quadtree = spikeyboi.spikey.sim_instance.quadtree
        self.objects = physics_group

    def fixed_update(self, fixed_delta):
        """
        Perform collision detection and resolution.
        """
        # Rebuild the quadtree
        self.quadtree.rebuild(list(self.objects), self.quadtree.bounding_rect, 2)

        # Resolve collisions iteratively
        unresolved = True
        max_iterations = 10
        iterations = 0

        while unresolved and iterations < max_iterations:
            unresolved = False
            iterations += 1

            for obj in self.objects:
                if not hasattr(obj, "rect") or getattr(obj, "is_static", False):
                    continue  # Skip static objects and invalid ones

                # Find potential collisions
                candidates = self.quadtree.hit(obj.rect)
                print(obj,candidates)

                for other in candidates:
                    if obj == other or not hasattr(other, "rect"):
                        continue  # Skip self and invalid objects

                    if obj.rect.colliderect(other.rect):
                        unresolved = True  # There are still collisions to resolve

                        # Notify objects of the collision
                        if hasattr(obj, "on_collision"):
                            obj.on_collision(other)
                        if hasattr(other, "on_collision"):
                            other.on_collision(obj)

                        # Resolve collision
                        self.resolve_collision(obj, other)

    def resolve_collision(self, obj, other):
        """
        Resolve a collision by repositioning objects to prevent overlap.
        """
        is_static = getattr(obj, 'is_static', False)

        # Calculate overlap
        overlap_x = min(obj.rect.right - other.rect.left, other.rect.right - obj.rect.left)
        overlap_y = min(obj.rect.bottom - other.rect.top, other.rect.bottom - obj.rect.top)

        # Resolve smaller overlap
        if abs(overlap_x) < abs(overlap_y):
            # Resolve horizontally
            if obj.rect.centerx < other.rect.centerx:
                obj.rect.right -= overlap_x if not is_static else 0
                if hasattr(obj, 'x'):
                    obj.x -= overlap_x if not is_static else 0
            else:
                obj.rect.left += overlap_x if not is_static else 0
                if hasattr(obj, 'x'):
                    obj.x += overlap_x if not is_static else 0
        else:
            # Resolve vertically
            if obj.rect.centery < other.rect.centery:
                obj.rect.bottom -= overlap_y if not is_static else 0
                if hasattr(obj, 'y'):
                    obj.y -= overlap_y if not is_static else 0
            else:
                obj.rect.top += overlap_y if not is_static else 0
                if hasattr(obj, 'y'):
                    obj.y += overlap_y if not is_static else 0
