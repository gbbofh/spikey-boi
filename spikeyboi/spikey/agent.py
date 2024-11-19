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

        self.brain = spikeyboi.spikey.brain.Brain(50, 7, 2)
        self.on_agent_moved_event = []
        self.ray_angles = np.array([np.pi / 3, np.pi / 6, np.pi / 12, 0, -np.pi / 12, -np.pi / 6, -np.pi / 3])

        # For reward modulation
        self.prev_x = self.x
        self.prev_y = self.y

        self.distances = np.zeros_like(self.ray_angles)
        self.prev_distances = np.zeros_like(self.ray_angles)

    def _make_rotation_matrix(self):
        m = [

            [np.cos(self.angle), -np.sin(self.angle)],
            [np.sin(self.angle), np.cos(self.angle)]
        ]
        return np.array(m)

    def fixed_update(self, fixed_delta):
        sim = spikeyboi.spikey.sim_instance

        hits = []
        self.distances[:] = 0

        for i,a in enumerate(self.ray_angles):
            angle_plus = self.angle + a
            dir_plus = np.array((np.cos(angle_plus), np.sin(angle_plus)))

            center = self.rect.center

            hit = sim.physics.cast_ray(center, dir_plus, 200, exclude={self})
            hits.append(hit)

            if hit:
                obj, point, dist = hit
                self.distances[i] = dist
                # weights = {
                #     spikeyboi.spikey.wall.Wall: 0.5,
                #     spikeyboi.spikey.food.Food: 1.0,
                # }
                # rewards = {
                #     spikeyboi.spikey.wall.Wall: 0.2,
                #     spikeyboi.spikey.food.Food: 0.7,
                # }
                # r = rewards[type(obj)]
                # w = weights[type(obj)]
                # input = w / (1 + np.exp(-dist))
                # self.brain.inputs[i] = input
            else:
                self.brain.inputs[i] *= np.exp(-1 / 20)

        deltas = self.distances - self.prev_distances
        self.prev_distances = self.distances

        self.brain.inputs[:] = 1 / (1 + np.exp(deltas))
        self.brain.update(fixed_delta)

        x,y = self.rect.center
        self.angle += 0.2 * self.brain.outputs[0] * fixed_delta
        self.angle -= 0.2 * self.brain.outputs[1] * fixed_delta

        self.angle = np.mod(self.angle, 2 * np.pi)

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

    def on_collision(self, other):
        if type(other) == spikeyboi.spikey.food.Food:
            self.brain.rewards[:,self.brain.output_first:self.brain.output_last + 1] += 0.5 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.brain.rewards[self.brain.input_first:self.brain.input_last + 1,:] += 0.3 * spikeyboi.spikey.sim_instance.fixed_delta_time
        elif type(other) == spikeyboi.spikey.wall.Wall:
            ltd = self.brain.output_first if self.brain.outputs[0] > self.brain.outputs[1] else self.brain.output_last
            ltp = int(not ltd)
            mask = self.distances > 0
            r = self.brain.rewards[:len(self.distances)]

            r[mask, :] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time
            # self.brain.rewards[self.distances > 0,:] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.brain.rewards[:,ltp] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.brain.rewards[:,ltd] -= 0.2 * spikeyboi.spikey.sim_instance.fixed_delta_time
            # self.brain.rewards[:,self.brain.output_first:self.brain.output_last + 1] += 0.5
            # self.brain.rewards[:,self.brain.input_first:self.brain.input_last + 1] += 0.3

    def debug_draw(self, surface):

        pg.draw.line(surface, (255,255,255), self.rect.center, self.rect.center + self.get_forward() * 300)

