import numpy as np
import pygame as pg


import spikeyboi.spikey
import spikeyboi.spikey.brain


class Agent(pg.sprite.Sprite):

    WALL_HIT_L_ID = 7
    WALL_HIT_R_ID = 8
    FOUND_FOOD_ID = 9

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

        # self.brain = spikeyboi.spikey.brain.Brain(50, 7, 2)
        self.brain = spikeyboi.spikey.brain.Brain(30, 10, 2)
        self.on_agent_moved_event = []
        # self.ray_angles = np.array([
        #     np.pi / 3, 
        #     np.pi / 6,
        #     np.pi / 12,
        #     0,
        #     -np.pi / 12,
        #     -np.pi / 6,
        #     -np.pi / 3
        # ])

        # self.ray_angles[:] /= 2
        # self.ray_angles[1:-1] /= 2
        self.angles = np.linspace(-np.pi / 3, np.pi / 3, 20)
        self.angles[:] /= 4

        self.inputs = np.zeros_like(self.angles)

        # For reward modulation
        # self.prev_x = self.x
        # self.prev_y = self.y

        # self.distances = np.zeros_like(self.ray_angles)
        # self.prev_distances = np.zeros_like(self.ray_angles)

        self.distances = np.zeros_like(self.angles)
        self.prev_distances = np.zeros_like(self.angles)
        self.deltas = np.zeros_like(self.brain.inputs[:7])

        self.mean_distances = np.zeros_like(self.brain.inputs[:7])

        # self.hits = []
        # self.input_scale = np.zeros_like(self.brain.inputs)
        # self.sigmoid_scale = np.zeros_like(self.brain.inputs)

        # self.S = lambda x,s: 0.5 / (1 + np.exp(s * (x - 0.8)))
        self.S = lambda x,s: 0.5 / (1 + np.exp(s * (x - 1.0)))
        self.id = 0
        self.food_consumed = 0

        self.hit_food = False
        self.hit_agent = False
        self.hit_wall = False
        self.collision_normal = None

    def _make_rotation_matrix(self):
        m = [

            [np.cos(self.angle), -np.sin(self.angle)],
            [np.sin(self.angle), np.cos(self.angle)]
        ]
        return np.array(m)

    def fixed_update(self, fixed_delta):
        sim = spikeyboi.spikey.sim_instance

        # self.distances[:] = 0
        # self.hits.clear()

        # angles = np.linspace(-np.pi / 3, np.pi / 3, 20)
        # angles[:] /= 4

        # inputs = np.zeros_like(angles)

        for i, a in enumerate(self.angles):
            angle = self.angle + a
            dir = np.array((np.cos(angle), -np.sin(angle)))

            origin = self.rect.center
            dist = np.max(spikeyboi.spikey.sim_instance.size)
            hit = sim.physics.cast_ray(origin, dir, dist, exclude={self})
            if hit:
                obj, point, hit_dist = hit

                m = {
                    spikeyboi.spikey.food.Food: 5,
                    spikeyboi.spikey.agent.Agent: 4,
                    spikeyboi.spikey.wall.Wall: 3,
                }

                n = {
                    spikeyboi.spikey.food.Food: 0.8,
                    spikeyboi.spikey.agent.Agent: 0.6,
                    spikeyboi.spikey.wall.Wall: 0.4,
                }

                scale = n[type(obj)]
                input = m[type(obj)]
                self.inputs[i] = scale * self.S(hit_dist / dist, input)

        left = self.inputs[:5].mean()
        right = self.inputs[-5:].mean()

        center = self.inputs[5:-5].reshape(5,2).mean(axis=1)

        vision = self.brain.inputs[:7]

        vision[0] = left
        vision[1:-1] = center
        vision[-1] = right

        R = self.brain.rewards[:7]
        R[vision > 0.03, :] += 0.02 * fixed_delta
        R[vision > 0.08,:] += 0.02 * fixed_delta
        R[vision > 0.15,:] += 0.02 * fixed_delta

        presyn = self.brain.net.P_pre > 0.3
        postsyn = self.brain.net.P_pre < 0.2

        self.brain.net.reward[presyn, :] += 0.02 * fixed_delta
        self.brain.net.reward[postsyn, :] -= 0.01 * fixed_delta

        # for i,a in enumerate(self.ray_angles):
        #     angle_plus = self.angle + a
        #     dir_plus = np.array((np.cos(angle_plus), -np.sin(angle_plus)))

        #     center = self.rect.center

        #     ray_length = np.max(spikeyboi.spikey.sim_instance.size)

        #     # TODO: Eventually rework this to use cast cone?
        #     # hit = sim.physics.cast_cone(center, dir_plus, np.pi/6, ray_length, exclude={self})
        #     hit = sim.physics.cast_ray(center, dir_plus, ray_length, exclude={self})
        #     self.hits.append(hit)

        #     if hit:
        #         obj, point, dist = hit
        #         max_dist = np.linalg.norm(spikeyboi.spikey.sim_instance.size)
        #         # self.distances[i] = dist
        #         self.distances[i] = dist / max_dist
        #         if type(obj) is spikeyboi.spikey.food.Food:
        #             self.input_scale[i] = 5
        #             self.sigmoid_scale[i] = 0.8
        #             self.brain.rewards[i,:] += 0.06 * fixed_delta

        #             presyn = self.brain.net.P_pre > 0.2
        #             self.brain.rewards[presyn] += 0.02 * fixed_delta

        #             postsyn = self.brain.net.P_post < 0.01
        #             self.brain.rewards[postsyn] -= 0.01 * fixed_delta

        #         elif type(obj) is spikeyboi.spikey.agent.Agent:
        #             self.input_scale[i] = 4
        #             self.sigmoid_scale[i] = 0.6
        #             self.brain.rewards[i,:] += 0.04 * fixed_delta

        #             presyn = self.brain.net.P_pre > 0.2
        #             self.brain.rewards[presyn] += 0.02 * fixed_delta

        #             postsyn = self.brain.net.P_post < 0.01
        #             self.brain.rewards[postsyn] -= 0.01 * fixed_delta

        #         elif type(obj) is spikeyboi.spikey.wall.Wall:
        #             self.input_scale[i] = 3
        #             self.sigmoid_scale[i] = 0.4
        #             self.brain.rewards[i,:] -= 0.02 * fixed_delta

        #             presyn = self.brain.net.P_pre > 0.2
        #             presyn = presyn[i]
        #             self.brain.rewards[presyn,i] -= 0.02 * fixed_delta

        #             postsyn = self.brain.net.P_post < 0.01
        #             postsyn = postsyn[i]
        #             self.brain.rewards[postsyn,i] -= 0.02 * fixed_delta

        # self.brain.inputs[:7] = self.sigmoid_scale[:7] * self.S(self.distances, self.input_scale[:7])
        if self.hit_wall:
            self.hit_wall = False

            ltd = self.brain.output_first if self.brain.outputs[0] > self.brain.outputs[1] else self.brain.output_last
            ltp = self.brain.output_last if self.brain.outputs[0] > self.brain.outputs[1] else self.brain.output_first

            if self.collision_normal is not None:

                normal = np.array(self.collision_normal).astype(np.float64)

                mag = np.linalg.norm(normal)
                normal /= mag

                angle = self.angle - np.pi / 2
                right = np.array((np.cos(angle), -np.sin(angle)))

                dot = np.dot(right, normal)
                c = right[0] * normal[1] - right[1] * normal[0]

                dir = dot * np.sign(c)

                if dir < 0:
                    self.brain.inputs[Agent.WALL_HIT_L_ID] = 1.0
                    self.brain.rewards[Agent.WALL_HIT_L_ID] += 0.002
                else:
                    self.brain.inputs[Agent.WALL_HIT_R_ID] = 1.0
                    self.brain.rewards[Agent.WALL_HIT_R_ID] += 0.002

            # mask = self.distances > 0
            # r = self.brain.rewards[:len(self.distances)]

            # r[mask, :] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time

            self.brain.rewards[:self.brain.net.num_exc,ltp] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.brain.rewards[self.brain.net.num_exc:,ltp] -= 0.2 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.brain.rewards[:self.brain.net.num_exc,ltd] -= 0.2 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.brain.rewards[self.brain.net.num_exc:,ltd] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time

        if self.hit_food:
            self.hit_food = False

            self.food_consumed += 1
            self.brain.inputs[Agent.FOUND_FOOD_ID] = 1.0
            self.brain.rewards[:,self.brain.output_first:self.brain.output_last + 1] += 0.5 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.brain.rewards[self.brain.input_first:self.brain.input_last + 1,:] += 0.3 * spikeyboi.spikey.sim_instance.fixed_delta_time

            presyn = self.brain.net.P_pre > 0.2
            self.brain.rewards[presyn] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time

            postsyn = self.brain.net.P_post < 0.1
            self.brain.rewards[postsyn] -= 0.05 * spikeyboi.spikey.sim_instance.fixed_delta_time

        self.brain.inputs[:] *= np.exp(-1/20)

        deltas = self.distances - self.prev_distances
        self.prev_distances = self.distances

        self.mean_distances[0] = self.distances[:5].mean()
        self.mean_distances[1:-1] = self.distances[5:-5].mean()
        self.mean_distances[-1] = self.distances[-5:].mean()

        left = deltas[:5].mean()
        center = deltas[5:-5].mean()
        right = deltas[-5:].mean()

        self.deltas[0] = left
        self.deltas[1:-1] = center
        self.deltas[-1] = right

        # self.brain.rewards[:, self.deltas > 0] += 0.06 * fixed_delta
        inds = np.where(self.deltas > 0)
        for i in np.nditer(inds, ('zerosize_ok',)):
            self.brain.rewards[:,i] += 0.08 * fixed_delta
        self.brain.update(fixed_delta)

        # inds = np.where(deltas > 0)
        # for i in np.nditer(inds, ('zerosize_ok',)):
        #     self.brain.rewards[:,i] += 0.08 * fixed_delta
        # self.brain.update(fixed_delta)

        self.angle += 0.2 * self.brain.outputs[0] * fixed_delta
        self.angle -= 0.2 * self.brain.outputs[1] * fixed_delta

        self.angle = np.mod(self.angle, 2 * np.pi)

        fwd_speed = self.brain.outputs.sum() / self.brain.outputs.size

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

    def on_collision(self, other, normal):
        self.collision_normal = normal
        if type(other) == spikeyboi.spikey.food.Food:
            # self.food_consumed += 1
            # self.brain.inputs[Agent.FOUND_FOOD_ID] += 0.2
            # self.brain.rewards[:,self.brain.output_first:self.brain.output_last + 1] += 0.5 * spikeyboi.spikey.sim_instance.fixed_delta_time
            # self.brain.rewards[self.brain.input_first:self.brain.input_last + 1,:] += 0.3 * spikeyboi.spikey.sim_instance.fixed_delta_time

            # presyn = self.brain.net.P_pre > 0.2
            # self.brain.rewards[presyn] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time

            # postsyn = self.brain.net.P_post < 0.1
            # self.brain.rewards[postsyn] -= 0.05 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.hit_food = True

        elif type(other) == spikeyboi.spikey.wall.Wall:
            # ltd = self.brain.output_first if self.brain.outputs[0] > self.brain.outputs[1] else self.brain.output_last
            # ltp = self.brain.output_last if self.brain.outputs[0] > self.brain.outputs[1] else self.brain.output_first

            # if normal is not None:

            #     # negative of the collision normal should be a vector from
            #     # the agent, to the collision point
            #     normal = np.array(normal).astype(np.float64)

            #     mag = np.linalg.norm(normal)
            #     normal /= mag

            #     angle = self.angle - np.pi / 2
            #     right = np.array((np.cos(angle), -np.sin(angle)))

            #     dot = np.dot(right, normal)
            #     c = right[0] * normal[1] - right[1] * normal[0]

            #     dir = dot * np.sign(c)

            #     if dir < 0:
            #         self.brain.inputs[Agent.WALL_HIT_L_ID] += 0.3
            #         self.brain.rewards[Agent.WALL_HIT_L_ID] += 0.002
            #     else:
            #         self.brain.inputs[Agent.WALL_HIT_R_ID] += 0.3
            #         self.brain.rewards[Agent.WALL_HIT_R_ID] += 0.002

            # mask = self.distances > 0
            # r = self.brain.rewards[:len(self.distances)]

            # r[mask, :] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time

            # self.brain.rewards[:self.brain.net.num_exc,ltp] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time
            # self.brain.rewards[self.brain.net.num_exc:,ltp] -= 0.2 * spikeyboi.spikey.sim_instance.fixed_delta_time
            # self.brain.rewards[:self.brain.net.num_exc,ltd] -= 0.2 * spikeyboi.spikey.sim_instance.fixed_delta_time
            # self.brain.rewards[self.brain.net.num_exc:,ltd] += 0.1 * spikeyboi.spikey.sim_instance.fixed_delta_time

            # presyn = self.brain.net.P_pre > 0.1
            # self.brain.rewards[presyn] -= 0.02 * spikeyboi.spikey.sim_instance.fixed_delta_time

            # postsyn = self.brain.net.P_post < 0.2
            # self.brain.rewards[postsyn] -= 0.02 * spikeyboi.spikey.sim_instance.fixed_delta_time
            self.hit_wall = True

    def debug_draw(self, surface):
        angle = self.angle - np.pi / 2
        r = np.array((50 * np.cos(angle), -50 * np.sin(angle)))

        rx, ry = r.astype(np.int32)

        x, y = self.rect.center
        pg.draw.line(surface, (100,100,255,100), self.rect.center, (x + rx, y + ry))

        forward = 50 * np.array(self.get_forward())
        pg.draw.line(surface, (100,255,100,100), self.rect.center, forward + self.rect.center)

