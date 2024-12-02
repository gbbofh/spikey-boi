import numpy as np
import pygame as pg


import spikeyboi.spikey
import spikeyboi.spikey.brain


class Agent(pg.sprite.Sprite):

    WALL_HIT_L_ID = 7
    WALL_HIT_R_ID = 8
    FOUND_FOOD_ID = 9

    # VISION_UPDATE_TICKS = 5
    # BRAIN_UPDATE_TICKS = 2
    UPDATE_TICKS_MAX = 4
    VISION_UPDATE_TICKS = 4
    BRAIN_UPDATE_TICKS = 2

    FOOD_IN_SCALE = 5
    AGENT_IN_SCALE = 4
    WALL_IN_SCALE = 3

    FOOD_SIGMOID_SCALE = 0.8 * 2
    AGENT_SIGMOID_SCALE = 0.6 * 2
    WALL_SIGMOID_SCALE = 0.4 * 2

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

        self.brain = spikeyboi.spikey.brain.Brain(30, 10, 2)
        self.on_agent_moved_event = []
        self.angles = np.linspace(np.pi / 3, -np.pi / 3, 20)
        self.angles[:] /= 3

        self.inputs = np.zeros_like(self.angles)

        self.distances = np.zeros_like(self.angles)
        self.prev_distances = np.zeros_like(self.angles)
        self.deltas = np.zeros_like(self.brain.inputs[:7])

        self.mean_distances = np.zeros_like(self.brain.inputs[:7])

        self.S = lambda x,s: 0.5 / (1 + np.exp(s * (x - 1.0)))
        self.id = 0
        self.food_consumed = 0

        self.hit_food = False
        self.hit_agent = False
        self.hit_wall = False
        self.collision_normal = None

        self.tick_count = 0

    def _make_rotation_matrix(self):
        m = [

            [np.cos(self.angle), -np.sin(self.angle)],
            [np.sin(self.angle), np.cos(self.angle)]
        ]
        return np.array(m)

    def fixed_update(self, fixed_delta):
        sim = spikeyboi.spikey.sim_instance

        max_dist = np.linalg.norm(sim.size)

        if self.tick_count % Agent.VISION_UPDATE_TICKS == 0:

            for i, a in enumerate(self.angles):
                angle = self.angle + a
                dir = np.array((np.cos(angle), -np.sin(angle)))

                origin = self.rect.center
                # dist = np.max(spikeyboi.spikey.sim_instance.size)
                hit = sim.physics.cast_ray(origin, dir, max_dist, exclude={self})
                if hit:
                    obj, point, hit_dist = hit

                    m = {
                        spikeyboi.spikey.food.Food: Agent.FOOD_IN_SCALE,
                        spikeyboi.spikey.agent.Agent: Agent.AGENT_IN_SCALE,
                        spikeyboi.spikey.wall.Wall: Agent.WALL_IN_SCALE,
                    }

                    n = {
                        spikeyboi.spikey.food.Food: Agent.FOOD_SIGMOID_SCALE,
                        spikeyboi.spikey.agent.Agent: Agent.AGENT_SIGMOID_SCALE,
                        spikeyboi.spikey.wall.Wall: Agent.WALL_SIGMOID_SCALE,
                    }

                    scale = n[type(obj)]
                    input = m[type(obj)]
                    self.inputs[i] = scale * self.S(hit_dist / max_dist, input)
                    self.distances[i] = hit_dist / max_dist

            left = self.inputs[:5].mean()
            right = self.inputs[-5:].mean()

            center = self.inputs[5:-5].reshape(5,2).mean(axis=1)

            vision = self.brain.inputs[:7]

            vision[0] = left
            vision[1:-1] = center
            vision[-1] = right

            R = self.brain.rewards[:7]

            scaled_vision = vision / 0.8

            mask_walls = scaled_vision > 0.2
            mask_agents = scaled_vision > 0.4
            mask_food = scaled_vision > 0.5

            R[mask_walls, :] += 0.01 * fixed_delta
            R[mask_agents, :] += 0.02 * fixed_delta
            R[mask_food, :] += 0.03 * fixed_delta

        # self.brain.rewards[:, 11:13] += 0.01 * fixed_delta
        self.brain.rewards[:, self.brain.output_first:self.brain.output_last + 1] += 0.01 * fixed_delta

        presyn = self.brain.net.P_pre > 0.2
        postsyn = self.brain.net.P_pre < 0.1

        self.brain.net.reward[presyn, :] += 0.02 * fixed_delta
        self.brain.net.reward[postsyn, :] -= 0.01 * fixed_delta

        if self.hit_wall:
            self.hit_wall = False

            ltd = self.brain.output_first if self.brain.outputs[0] > self.brain.outputs[1] else self.brain.output_last
            ltp = self.brain.output_last if self.brain.outputs[0] > self.brain.outputs[1] else self.brain.output_first

            if self.collision_normal is not None:

                normal = np.array(self.collision_normal).astype(np.float64)

                mag = np.linalg.norm(normal)
                normal = normal / mag if mag != 0 else normal

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

            self.brain.rewards[:self.brain.net.num_exc,ltp] += 0.1 * fixed_delta
            self.brain.rewards[self.brain.net.num_exc:,ltp] -= 0.2 * fixed_delta
            self.brain.rewards[:self.brain.net.num_exc,ltd] -= 0.2 * fixed_delta
            self.brain.rewards[self.brain.net.num_exc:,ltd] += 0.1 * fixed_delta

        if self.hit_food:
            self.hit_food = False

            self.food_consumed += 1
            self.brain.inputs[Agent.FOUND_FOOD_ID] = 1.0
            self.brain.rewards[:,self.brain.output_first:self.brain.output_last + 1] += 0.5 * fixed_delta
            self.brain.rewards[self.brain.input_first:self.brain.input_last + 1,:] += 0.3 * fixed_delta

            presyn = self.brain.net.P_pre > 0.2
            self.brain.rewards[presyn] += 0.1 * fixed_delta

            postsyn = self.brain.net.P_post < 0.1
            self.brain.rewards[postsyn] -= 0.05 * fixed_delta

        if self.hit_agent:
            self.hit_agent = False
            self.brain.rewards[:, self.brain.output_first:self.brain.output_last + 1] -= 0.2 * fixed_delta
            self.brain.rewards[self.brain.input_first:self.brain.input_last + 1,:] += 0.1 * fixed_delta

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

        if np.all(self.brain.outputs > 0):
            inds = np.where(self.deltas > 0)
            for i in np.nditer(inds, ('zerosize_ok',)):
                self.brain.rewards[:,i] += 0.08 * fixed_delta

        if self.tick_count % Agent.BRAIN_UPDATE_TICKS == 0:
            self.brain.update(fixed_delta)

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

        self.tick_count = (self.tick_count + 1) % Agent.UPDATE_TICKS_MAX

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
            self.hit_food = True

        elif type(other) == spikeyboi.spikey.wall.Wall:
            self.hit_wall = True

    def debug_draw(self, surface):
        angle = self.angle - np.pi / 2
        r = np.array((50 * np.cos(angle), -50 * np.sin(angle)))

        rx, ry = r.astype(np.int32)

        x, y = self.rect.center
        pg.draw.line(surface, (100,100,255,100), self.rect.center, (x + rx, y + ry))

        forward = 50 * np.array(self.get_forward())
        pg.draw.line(surface, (100,255,100,100), self.rect.center, forward + self.rect.center)

