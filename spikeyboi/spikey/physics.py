import numpy as np
import pygame as pg


import spikeyboi.spikey


# TODO: Update physics to make use of collision masks?
class Physics():

    def __init__(self, physics_group: pg.sprite.Group):
        self.quadtree = spikeyboi.spikey.sim_instance.quadtree
        self.objects = physics_group
        self._ray_objects = []
        self._ray_objects_next = []

    def fixed_update(self, fixed_delta):
        """
        Perform collision detection and resolution.
        """
        # Rebuild the quadtree
        self.quadtree.rebuild(list(self.objects), self.quadtree.bounding_rect, 5)

        # Resolve collisions iteratively
        unresolved = True
        max_iterations = 20
        iterations = 0

        pairs = set()

        while unresolved and iterations < max_iterations:
            unresolved = False
            iterations += 1

            for obj in self.objects:
                # if not hasattr(obj, "rect") or getattr(obj, "is_static", False):
                if not hasattr(obj, 'rect'):
                    # print(hasattr(obj, 'rect'), getattr(obj, 'is_static', False))
                    continue  # Skip static objects and invalid ones

                # Find potential collisions
                candidates = self.quadtree.hit(obj.rect, exclude=obj)

                # if type(obj) == spikeyboi.spikey.food.Food:
                #     print(f'{candidates=}')

                for other in candidates:
                    # # if obj == other or not hasattr(other, 'is_static', False):
                    # if obj == other:
                    #     continue  # Skip self and invalid objects

                    pair = (min(id(obj), id(other)), max(id(obj), id(other)))

                    if obj.rect.colliderect(other.rect):
                        unresolved = True  # There are still collisions to resolve

                        # Resolve collision
                        rel_pos = self.resolve_collision(obj, other)

                        if pair in pairs:
                            continue

                        pairs.add(pair)

                        # Notify objects of the collision
                        if hasattr(obj, "on_collision"):
                            obj.on_collision(other, rel_pos)
                        if hasattr(other, "on_collision"):
                            other.on_collision(obj, rel_pos)

        self._ray_objects = self._ray_objects_next
        self._ray_objects_next = []

    def collision_normal(self, left, right):
        """
        Calculate the approximate collision normal from overlapping area
        
        Args:
            left: A Sprite object with rect, mask, x, and y properties
            right: A Sprite object with rect, mask, x, and y properties
            
        Returns:
            A tuple of (dx, dy) or None if no collision normal can be calculated
        """
        xoff = right.rect[0] - left.rect[0]
        yoff = right.rect[1] - left.rect[1]

        offset = (xoff, yoff)

        left_mask = left.mask
        right_mask = right.mask

        does_overlap = left_mask.overlap(right_mask, offset)
        overlap = left_mask.overlap_area(right_mask, offset)

        if overlap == 0:
            return

        x_plus = left_mask.overlap_area(right_mask, (xoff + 1, yoff))
        x_minus = left_mask.overlap_area(right_mask, (xoff - 1, yoff))

        y_plus = left_mask.overlap_area(right_mask, (xoff, yoff + 1))
        y_minus = left_mask.overlap_area(right_mask, (xoff, yoff - 1))

        dx = x_plus - x_minus
        dy = y_plus - y_minus

        if dx == 0 and dy == 0:
            return

        magnitude = np.linalg.norm((dx, dy))
        step = min(overlap, 5)

        dx = int(dx / magnitude * step)
        dy = int(dy / magnitude * step)

        return (dx, dy)


    def resolve_collision(self, obj, other):
        if pg.sprite.spritecollide(obj, (other,), False, pg.sprite.collide_mask):
            norm = self.collision_normal(obj, other)
            if norm is None:
                return
            dx, dy = norm
            if not getattr(obj, 'is_static', False):
                name = obj.__class__.__name__
                rect = obj.rect.copy()
                obj.rect.x += dx
                obj.rect.y += dy
                obj.x, obj.y = obj.rect.x, obj.rect.y
            if not getattr(other, 'is_static', False):
                name = other.__class__.__name__
                rect = other.rect.copy()
                other.rect.x -= dx
                other.rect.y -= dy
                other.x, other.y = other.rect.x, other.rect.y
            return (-dx, -dy)

    def cast_cone(self, origin, direction, angle, max_distance, exclude=None):
        """
        Cast a cone (isosceles triangle) from origin in the given direction and return objects within the cone.
        
        Args:
            origin: Tuple or numpy array (x, y) of cone origin
            direction: Normalized direction vector
            angle: Cone angle in radians
            max_distance: Maximum distance to check
            exclude: Set of objects to exclude from check
            
        Returns:
            List of tuples (hit_object, hit_point, distance) for all objects within the cone
        """
        exclude = exclude or set()
        origin = np.array(origin)
        direction = np.array(direction)
        direction = direction / np.linalg.norm(direction)  # Ensure normalized

        # Compute left and right directions
        angle_left = angle / 2
        angle_right = -angle / 2

        def rotate_vector(vec, angle):
            """Rotate vector by angle in radians."""
            cos_a = np.cos(angle)
            sin_a = np.sin(angle)
            return np.array([vec[0] * cos_a - vec[1] * sin_a,
                             vec[0] * sin_a + vec[1] * cos_a])
        
        left_dir = rotate_vector(direction, angle_left)
        right_dir = rotate_vector(direction, angle_right)
        
        left_point = origin + left_dir * max_distance
        right_point = origin + right_dir * max_distance

        # Compute the bounding rectangle of the triangle
        xs = [origin[0], left_point[0], right_point[0]]
        ys = [origin[1], left_point[1], right_point[1]]
        min_x = min(xs)
        min_y = min(ys)
        max_x = max(xs)
        max_y = max(ys)
        width = max_x - min_x
        height = max_y - min_y

        broad_rect = pg.Rect(min_x, min_y, width + 1, height + 1)
        candidates = self.quadtree.hit(broad_rect)

        hits = []

        for obj in candidates:
            if obj in exclude or not hasattr(obj, 'rect') or not hasattr(obj, 'mask'):
                continue

            if not obj.rect.colliderect(broad_rect):
                continue

            # Create cone mask in object's local space
            local_origin = (origin[0] - obj.rect.x, origin[1] - obj.rect.y)
            local_left_point = (left_point[0] - obj.rect.x, left_point[1] - obj.rect.y)
            local_right_point = (right_point[0] - obj.rect.x, right_point[1] - obj.rect.y)

            # Create a surface exactly the size of the object
            cone_surface = pg.Surface(obj.rect.size, pg.SRCALPHA)

            # Draw the triangle in object space
            pg.draw.polygon(cone_surface, (255, 255, 255, 255), [local_origin, local_left_point, local_right_point])
            cone_mask = pg.mask.from_surface(cone_surface)
            pg.image.save(cone_mask.to_surface(), "cone_mask_debug.png")

            # Check for collision with object's mask
            overlap = obj.mask.overlap_mask(cone_mask, (0, 0))
            if overlap.count() > 0:
                # Get the collision points
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

                    hits.append((obj, hit_point, dist))

        return hits

    # def cast_cone(self, origin, direction, angle, max_distance, exclude=None):
    #     """
    #     Cast a cone from an origin and detect objects within it.
    #     
    #     Args:
    #         origin (tuple): Starting point of the cone as (x, y).
    #         direction (tuple): Direction vector for the cone.
    #         angle (float): Full angle of the cone in degrees.
    #         max_distance (float): Maximum distance the cone extends.
    #         exclude (set): Optional set of objects to exclude from detection.
    #     
    #     Returns:
    #         list: A list of objects detected within the cone.
    #     """
    #     exclude = exclude or set()
    #     origin = np.array(origin, dtype=float)
    #     direction = np.array(direction, dtype=float)
    #     direction /= np.linalg.norm(direction)  # Normalize direction vector
    #     
    #     # Calculate the boundaries of the cone
    #     half_angle = np.radians(angle / 2)
    #     cos_half_angle = np.cos(half_angle)
    #     
    #     # Convert max_distance to a circle boundary for broad phase
    #     cone_rect = pg.Rect(
    #         origin[0] - max_distance,
    #         origin[1] - max_distance,
    #         2 * max_distance,
    #         2 * max_distance
    #     )
    #     
    #     # Broad phase collision detection (using a quadtree or all objects)
    #     candidates = self.quadtree.hit(cone_rect) if hasattr(self, "quadtree") else self.objects
    #     
    #     detected = []
    #     
    #     for obj in candidates:
    #         if obj in exclude or not hasattr(obj, 'rect') or not hasattr(obj, 'mask'):
    #             continue

    #         # Calculate vector from origin to object's center
    #         obj_center = np.array(obj.rect.center, dtype=float)
    #         to_obj = obj_center - origin
    #         dist = np.linalg.norm(to_obj)
    #         
    #         if dist > max_distance:
    #             continue  # Skip objects outside the max distance
    #         
    #         # Check angle
    #         if dist > 0:
    #             to_obj /= dist  # Normalize vector to object
    #         dot_product = np.dot(direction, to_obj)
    #         if dot_product < cos_half_angle:
    #             continue  # Skip objects outside the cone angle
    #         
    #         # Fine phase: Check precise collision with the mask
    #         local_origin = (int(origin[0] - obj.rect.x), int(origin[1] - obj.rect.y))
    #         overlap = obj.mask.overlap_area(pg.mask.Mask((1, 1), fill=True), local_origin)
    #         if overlap > 0:
    #             detected.append(obj)
    # 
    #     return detected

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
            # Testing with thicker lines
            pg.draw.line(ray_surface, (255, 255, 255, 255), local_origin, local_end, 3)
            ray_mask = pg.mask.from_surface(ray_surface)
            # pg.image.save(ray_mask.to_surface(), "ray_mask_debug.png")
            
            # Check for collision with object's mask
            overlap = obj.mask.overlap_mask(ray_mask, (0, 0))
            if overlap.count() > 0:
                # Get the first collision point
                overlap_points = overlap.outline()
                if overlap_points:
                    # Convert local hit point back to world space
                    # local_hit = overlap_points[0]
                    pts = np.array(overlap_points)
                    pts[:] += np.array((obj.rect.x, obj.rect.y))
                    dist = np.linalg.norm(pts - origin, axis=1)
                    ind = np.argmin(dist)

                    local_hit = pts[ind]

                    if dist[ind] < closest_distance:
                        closest_distance = dist[ind]
                        closest_hit = obj
                        closest_point = local_hit

                    # hit_point = np.array((
                    #     local_hit[0] + obj.rect.x,
                    #     local_hit[1] + obj.rect.y
                    # ))
                    # 
                    # # Calculate distance
                    # dist = np.linalg.norm(hit_point - origin)
                    
                    # if dist < closest_distance:
                    #     closest_distance = dist
                    #     closest_hit = obj
                    #     closest_point = hit_point

        if closest_hit is None:
            return None

        self._ray_objects_next.append((origin, closest_point))
            
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

    def debug_draw(self, surf):
        color = (250,100,100)
        for obj in self.objects:
            pg.draw.rect(surf, color, obj.rect, 1)

        for obj in self._ray_objects:
            pg.draw.line(surf, (100,100,250), obj[0], obj[1], 3)

        # This causes flickering because raycasting happens on a fixed tick -- need to determine
        # a way to prevent this flickering from occuring.
        # self._ray_objects.clear()

