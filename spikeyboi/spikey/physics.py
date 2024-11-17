import numpy as np
import pygame as pg


import spikeyboi.spikey


# TODO: Update physics to make use of collision masks?
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

    def cast_ray(self, origin, direction, max_distance, exclude=None):
        """
        Cast a ray from origin in the given direction and return the first collision.
        
        Args:
            origin: Tuple or numpy array (x, y) of ray origin
            direction: Normalized direction vector
            max_distance: Maximum distance to check
            exclude: Set of objects to exclude from check
            
        Returns:
            Tuple of (hit_object, hit_point, distance) or None if no hit
        """
        exclude = exclude or set()
        origin = np.array(origin)
        direction = np.array(direction) 
        direction = direction / np.linalg.norm(direction)  # Ensure normalized
        
        end = origin + direction * max_distance
        
        # Create bounding rectangle for broad phase
        min_x = min(origin[0], end[0])
        min_y = min(origin[1], end[1])
        width = abs(end[0] - origin[0])
        height = abs(end[1] - origin[1])
        
        broad_rect = pg.Rect(min_x, min_y, width + 1, height + 1)
        candidates = self.quadtree.hit(broad_rect)
        
        closest_hit = None
        closest_distance = max_distance
        closest_point = None

        for obj in candidates:
            if obj in exclude or not hasattr(obj, 'rect') or not hasattr(obj, 'mask'):
                continue

            # First do a quick line-rect test
            if not self.line_intersects_rect((origin, end), obj.rect):
                continue
                
            # Create ray mask in object's local space
            local_origin = (origin[0] - obj.rect.x, origin[1] - obj.rect.y)
            local_end = (end[0] - obj.rect.x, end[1] - obj.rect.y)
            
            # Create a surface exactly the size of the object
            ray_surface = pg.Surface(obj.rect.size, pg.SRCALPHA)
            
            # Draw the line in object space
            pg.draw.line(ray_surface, (255, 255, 255, 255), local_origin, local_end)
            ray_mask = pg.mask.from_surface(ray_surface)
            pg.image.save(ray_mask.to_surface(), "ray_mask_debug.png")
            
            # Check for collision with object's mask
            overlap = obj.mask.overlap_mask(ray_mask, (0, 0))
            if overlap.count() > 0:
                # Get the first collision point
                overlap_points = overlap.outline()
                if overlap_points:
                    # Convert local hit point back to world space
                    local_hit = overlap_points[0]
                    hit_point = np.array((
                        local_hit[0] + obj.rect.x,
                        local_hit[1] + obj.rect.y
                    ))
                    
                    # Calculate distance
                    dist = np.linalg.norm(hit_point - origin)
                    
                    if dist < closest_distance:
                        closest_distance = dist
                        closest_hit = obj
                        closest_point = hit_point

        if closest_hit is None:
            return None
            
        return (closest_hit, closest_point, closest_distance)

    def line_intersects_rect(self, line, rect):
        """
        Test if a line segment intersects with a rectangle.
        Uses Cohen-Sutherland algorithm for efficiency.
        """
        start, end = line
        
        def get_outcode(x, y):
            code = 0
            if x < rect.left: code |= 1
            if x > rect.right: code |= 2
            if y < rect.top: code |= 4
            if y > rect.bottom: code |= 8
            return code
            
        outcode1 = get_outcode(start[0], start[1])
        outcode2 = get_outcode(end[0], end[1])
        
        while True:
            if not (outcode1 | outcode2):  # Both points inside
                return True
            if (outcode1 & outcode2):  # Both points on same side
                return False
                
            # Pick an outside point
            outcode = outcode1 if outcode1 else outcode2
            
            # Find intersection point
            if outcode & 8:  # Above
                x = start[0] + (end[0] - start[0]) * (rect.bottom - start[1]) / (end[1] - start[1])
                y = rect.bottom
            elif outcode & 4:  # Below
                x = start[0] + (end[0] - start[0]) * (rect.top - start[1]) / (end[1] - start[1])
                y = rect.top
            elif outcode & 2:  # Right
                y = start[1] + (end[1] - start[1]) * (rect.right - start[0]) / (end[0] - start[0])
                x = rect.right
            else:  # Left
                y = start[1] + (end[1] - start[1]) * (rect.left - start[0]) / (end[0] - start[0])
                x = rect.left
                
            if outcode == outcode1:
                start = (x, y)
                outcode1 = get_outcode(x, y)
            else:
                end = (x, y)
                outcode2 = get_outcode(x, y)

