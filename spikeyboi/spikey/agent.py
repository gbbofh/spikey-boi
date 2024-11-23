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
        input_scale = np.zeros_like(self.brain.inputs)

        self.distances[:] = 0

        for i,a in enumerate(self.ray_angles):
            angle_plus = self.angle + a
            dir_plus = np.array((np.cos(angle_plus), -np.sin(angle_plus)))

            center = self.rect.center

            hit = sim.physics.cast_ray(center, dir_plus, 500, exclude={self})
            hits.append(hit)

            if hit:
                obj, point, dist = hit
                max_dist = np.linalg.norm(spikeyboi.spikey.sim_instance.size)
                # self.distances[i] = dist
                self.distances[i] = dist / max_dist
                if type(obj) is spikeyboi.spikey.food.Food:
                    # input = 1 / (1 + np.exp(5 * -dist))

                    # self.brain.inputs[i] += input
                    input_scale[i] = 5
                if type(obj) is spikeyboi.spikey.food.Food:
                    # input = 1 / (1 + np.exp(5 * -dist))

                    # self.brain.inputs[i] += input
                    input_scale[i] = 5
                    self.brain.rewards[i,:] += 0.05 * fixed_delta

                    presyn = self.brain.net.P_pre > 0.2
                    self.brain.rewards[presyn] += 0.02 * fixed_delta

                    postsyn = self.brain.net.P_post < 0.01
                    self.brain.rewards[postsyn] -= 0.01 * fixed_delta
                elif type(obj) is spikeyboi.spikey.wall.Wall:
                    # input = 1 / (1 + np.exp(3 * -dist))

                    # self.brain.inputs[i] += input
                    input_scale[i] = 3
                    self.brain.rewards[i,:] += 0.02 * fixed_delta

                    presyn = self.brain.net.P_pre > 0.2
                    presyn = presyn[i]
                    self.brain.rewards[presyn,i] -= 0.02 * fixed_delta

                    postsyn = self.brain.net.P_post < 0.01
                    postsyn = postsyn[i]
                    self.brain.rewards[postsyn,i] += 0.02 * fixed_delta
            # else:
            #     self.brain.inputs[i] *= np.exp(-1 / 20)

        deltas = self.distances - self.prev_distances
        self.prev_distances = self.distances

        # self.brain.inputs[:] += 0.5 / (1 + np.exp(input_scale * -deltas)) * fixed_delta
        # S = lambda x,s: 0.5 / (1 + np.exp(-s * (x - 0.8)))
        S = lambda x,s: 0.5 / (1 + np.exp(s * (x - 0.8)))
        # self.brain.inputs[:] = S(deltas, input_scale)
        self.brain.inputs[:] = S(self.distances, input_scale)
        self.brain.inputs[:] *= np.exp(-1/20)

        inds = np.where(deltas > 0)
        for i in np.nditer(inds, ('zerosize_ok',)):
            self.brain.rewards[:,i] += 0.08 * fixed_delta
        self.brain.update(fixed_delta)

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
        # print(f'{self} collided with {other}')
        if type(other) == spikeyboi.spikey.food.Food:
            # print('omnomnomnom')
            self.brain.rewards[:,self.brain.output_first:self.brain.output_last + 1] += 0.5 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.brain.rewards[self.brain.input_first:self.brain.input_last + 1,:] += 0.3 * spikeyboi.spikey.sim_instance.fixed_delta_time

            presyn = self.brain.net.P_pre > 0.2
            self.brain.rewards[presyn] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time

            postsyn = self.brain.net.P_post < 0.1
            self.brain.rewards[postsyn] -= 0.05 * spikeyboi.spikey.sim_instance.fixed_delta_time

        elif type(other) == spikeyboi.spikey.wall.Wall:
            # print('AAAaaaAAAaa')
            ltd = self.brain.output_first if self.brain.outputs[0] > self.brain.outputs[1] else self.brain.output_last
            ltp = self.brain.output_last if self.brain.outputs[0] > self.brain.outputs[1] else self.brain.output_first

            mask = self.distances > 0
            r = self.brain.rewards[:len(self.distances)]

            r[mask, :] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time

            self.brain.rewards[:self.brain.net.num_exc,ltp] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.brain.rewards[self.brain.net.num_exc:,ltp] -= 0.2 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.brain.rewards[:self.brain.net.num_exc,ltd] -= 0.2 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.brain.rewards[self.brain.net.num_exc:,ltd] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time

            presyn = self.brain.net.P_pre > 0.1
            self.brain.rewards[presyn] -= 0.02 * spikeyboi.spikey.sim_instance.fixed_delta_time

            postsyn = self.brain.net.P_post < 0.2
            self.brain.rewards[postsyn] -= 0.02 * spikeyboi.spikey.sim_instance.fixed_delta_time

    def debug_draw(self, surface):
        # pg.draw.line(surface, (255,255,255), self.rect.center, self.rect.center + self.get_forward() * 300)
        pass

