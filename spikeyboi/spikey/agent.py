import numpy as np
import pygame as pg


import spikeyboi.spikey
import spikeyboi.spikey.brain


class Agent(pg.sprite.Sprite):

    def __init__(self, *group: pg.sprite.Group):
        super().__init__(group)

        self.vertices = np.array([
            (-5, -5),
            (-5, 5),
            (10, 0)
        ])

        self.image = pg.Surface((20,20), pg.SRCALPHA)
        self.rect = self.image.get_rect()

        self.angle = 0
        self.x = self.rect.x
        self.y = self.rect.y

        pg.draw.polygon(self.image, (255,255,255), self.vertices)

        self.mask = pg.mask.from_surface(self.image)

        self.brain = spikeyboi.spikey.brain.Brain(50, 5, 2)
        self.on_agent_moved_event = []

    def _make_rotation_matrix(self):
        m = [

            [np.cos(self.angle), -np.sin(self.angle)],
            [np.sin(self.angle), np.cos(self.angle)]
        ]
        return np.array(m)

    def update(self, delta_time):
        # x,y = self.rect.center
        self.brain.inputs[:] = spikeyboi.spikey.random.uniform(0, 1)
        self.brain.update(delta_time)

        # self.angle += self.brain.outputs[0] * delta_time
        # self.angle -= self.brain.outputs[1] * delta_time

        # fwd_speed = self.brain.outputs.sum() / self.brain.outputs.size

        # # Testing movement
        # forward = self.get_forward()
        # self.x += 10 * forward[0] * delta_time * fwd_speed
        # self.y -= 10 * forward[1] * delta_time * fwd_speed

        # m = self._make_rotation_matrix()
        # self.image.fill((0,0,0,0))
        # v = self.vertices @ m
        # v[:,0] += self.image.get_size()[0] / 2
        # v[:,1] += self.image.get_size()[1] / 2
        # self.rect.x = self.x
        # self.rect.y = self.y

        # dx = np.abs(self.rect.centerx - x)
        # dy = np.abs(self.rect.centery - y)

        # # if dx > 3 or dy > 3:
        # #     for e in self.on_agent_moved_event:
        # #         e(self)

        # pg.draw.polygon(self.image, (255,255,255), v)

    def fixed_update(self, fixed_delta):
        # qt = spikeyboi.spikey.sim_instance.quadtree
        # cx, cy = self.rect.center
        # size = 100
        # rect = pg.Rect(cx - size // 2, cy - size // 2, size, size)
        # items = qt.hit(rect, exclude=self)

        x,y = self.rect.center
        self.angle += 0.2 * self.brain.outputs[0] * fixed_delta
        self.angle -= 0.2 * self.brain.outputs[1] * fixed_delta

        fwd_speed = self.brain.outputs.sum() / self.brain.outputs.size

        # Testing movement
        forward = self.get_forward()
        self.x += 2 * forward[0] * fixed_delta * fwd_speed
        self.y += 2 * forward[1] * fixed_delta * fwd_speed

        m = self._make_rotation_matrix()
        self.image.fill((0,0,0,0))
        v = self.vertices @ m
        v[:,0] += self.image.get_size()[0] / 2
        v[:,1] += self.image.get_size()[1] / 2
        self.rect.x = self.x
        self.rect.y = self.y

        hit = spikeyboi.spikey.sim_instance.physics.cast_ray(self.rect.center, forward, 300)
        if hit:
            obj, pt, dist = hit
            print(f'Ray: {obj}, {pt}, {dist}')

        pg.draw.polygon(self.image, (255,255,255), v)
        self.mask = pg.mask.from_surface(self.image)

    def get_forward(self):
        return np.array([np.cos(self.angle), -np.sin(self.angle)])

    def get_angle(self):
        if not self.target:
            return 0

        u = self.forward
        v = (self.target.x - self.x, self.target.y - self.y)

        norm = np.linalg.norm(v)
        v /= norm if norm != 0 else 1

        d = np.dot(u, v)

        angle = np.arccos(d)

        c = u[0] * v[1] - u[1] * v[0]

        return angle * np.sign(c)

    def debug_draw(self, surface):

        pg.draw.line(surface, (255,255,255), self.rect.center, self.get_forward() * 300)

