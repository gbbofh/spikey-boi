import numpy as np


import pygame


import util
import network


class Agent():

    NUM_INPUTS = 16
    IN_DXP = 0
    IN_DYP = 1
    IN_DXN = 2
    IN_DYN = 3
    IN_DST = 4
    IN_WALL_L = 5
    IN_WALL_R = 6
    IN_TARG = 7
    IN_LOOK = 8
    IN_NEAR = 9
    IN_COSA = 10
    IN_SINA = 11
    IN_PX = 12
    IN_PY = 13
    IN_NX = 14
    IN_NY = 15
    IN_LAST = IN_NY

    NUM_MOTORS = 1
    MOTOR_START = IN_LAST + 1
    MOTOR_LEFT = MOTOR_START
    MOTOR_RIGHT = MOTOR_LEFT + NUM_MOTORS
    # MOTOR_FORWARD = MOTOR_RIGHT + NUM_MOTORS
    # MOTOR_LAST = MOTOR_FORWARD
    MOTOR_LAST = MOTOR_RIGHT + NUM_MOTORS - 1
    MOTOR_WINDOW = 1000

    # ROTATION_CONST = 500
    # MOVE_CONST = 50
    # ROTATION_CONST = 800
    # MOVE_CONST = 100
    # MOVE_SPEED = 0.05
    # ROTATION_SPEED = 0.005
    MOVE_SPEED = 0.2
    ROTATION_SPEED = 0.02

    MAX_TIME = 5000

    MAX_SPEED = 1
    MAX_ROTATION = 0.2

    def __init__(self, net=None):
        self.net = net or network.Network()
        self.screen = pygame.display.get_surface()

        self.x = self.screen.get_width() // 2
        self.y = self.screen.get_height() // 2

        self.angle = 0
        self.forward = (0, 0)
        self.pos = (self.x, self.y)

        self.time_accum = 0

        self.I_scale = 1.0

        self.vertices = np.array([
            (-5, 5),
            (-5, -5),
            (10, 0)
        ])

        self.bounds = pygame.Rect(self.x - 5, self.y - 5, 10, 10)
        self.color = (255, 255, 255)
        self.inputs = self.net.I_inj[:]

        self.s_trace = self.net.spike_trace
        self.firing_rates = self.net.firing_rates

        L_start = Agent.MOTOR_LEFT
        L_end = Agent.MOTOR_LEFT + Agent.NUM_MOTORS
        R_start = Agent.MOTOR_RIGHT
        R_end = Agent.MOTOR_RIGHT + Agent.NUM_MOTORS
        # F_start = Agent.MOTOR_FORWARD
        # F_end = Agent.MOTOR_FORWARD + Agent.NUM_MOTORS

        self.motor_left = self.s_trace[L_start:L_end, :]
        self.motor_right = self.s_trace[R_start:R_end, :]
        # self.motor_forward = self.s_trace[F_start:F_end, :]

        self.L_fr = 0
        self.R_fr = 0
        self.F_fr = 0

        self.target = None
        self.collision_count = 0

        self.last_time = util.Time.cur_time

        self.on_target_found = set()
        self.look_time = 0

        self.inputs_enabled = True
        self.prev_dist = 0
        self.angle_to_target = np.pi
        self.last_angle_to_target = np.pi

    def draw(self):
        theta = self.angle

        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        pos = np.array(self.pos)
        
        verts = []

        for v in self.vertices:
            verts.append(rot @ v + pos)

        self.bounds = pygame.draw.polygon(self.screen, self.color, verts, 0)

        forward = np.array(self.forward)

        start = (pos[0] + 10*forward[0], pos[1] + 10*forward[1])
        end = (pos[0] + 20*forward[0], pos[1]+ 20*forward[1])
        end_l = (pos[0] + 20 * np.cos(self.angle + np.pi / 4), pos[1] + 20 * np.sin(self.angle + np.pi / 4))
        end_r = (pos[0] + 20 * np.cos(self.angle - np.pi / 4), pos[1] + 20 * np.sin(self.angle - np.pi / 4))

        pygame.draw.line(self.screen, (255,0,255,255), start, end)
        pygame.draw.line(self.screen, (255,0,255,255), start, end_l)
        pygame.draw.line(self.screen, (255,0,255,255), start, end_r)

    def update(self):
        delta_time = util.Time.delta_time / 1000
        self.time_accum += util.Time.delta_time

        if self.time_accum >= self.MAX_TIME:
            self.time_accum = 0

        if self.check_wall_ahead():
            wall_angle = self.get_closest_wall_angle()

            if wall_angle != np.pi:
                distance = self.get_wall_distance(self.angle + wall_angle)

                if wall_angle > 0:
                    ltp = np.s_[:self.net.num_exc, Agent.MOTOR_LEFT:Agent.MOTOR_RIGHT]
                    iltd = np.s_[self.net.num_inh:, Agent.MOTOR_LEFT:Agent.MOTOR_RIGHT]
                    ltd = np.s_[:self.net.num_exc, Agent.MOTOR_RIGHT:Agent.MOTOR_LAST + 1]
                    iltp = np.s_[self.net.num_inh:, Agent.MOTOR_RIGHT:Agent.MOTOR_LAST + 1]

                    wall_ltd = np.s_[Agent.IN_WALL_R, :]
                    wall_ltp = np.s_[:0]

                    self.inputs[Agent.IN_WALL_R] += 1 - distance / 40 * delta_time
                    # print('Should Turn Left:', wall_angle, distance / 40)
                else:
                    ltp = np.s_[:self.net.num_exc, Agent.MOTOR_RIGHT:Agent.MOTOR_LAST + 1]
                    iltd = np.s_[self.net.num_inh:, Agent.MOTOR_RIGHT:Agent.MOTOR_LAST + 1]
                    ltd = np.s_[:self.net.num_exc, Agent.MOTOR_LEFT:Agent.MOTOR_RIGHT]
                    iltp = np.s_[self.net.num_inh:, Agent.MOTOR_LEFT:Agent.MOTOR_RIGHT]

                    wall_ltd = np.s_[Agent.IN_WALL_L, :]
                    wall_ltp = np.s_[:0]

                    self.inputs[Agent.IN_WALL_L] += 1 - distance / 40 * delta_time
                    # print('Should Turn Right:', wall_angle, distance / 40)

                x = self.x / self.screen.get_width() - 0.5
                y = self.y / self.screen.get_height() - 0.5

                place_x = Agent.IN_PX if x > 0 else Agent.IN_NX
                place_y = Agent.IN_PY if y > 0 else Agent.IN_NY

                x_ltd = np.s_[:0]
                x_ltp = np.s_[place_x, :]

                y_ltd = np.s_[:0]
                y_ltp = np.s_[place_y, :]

                reward = 1 * delta_time
                self.reward(0.5 * reward, ltp)
                self.reward(0.5 * reward, iltp)
                self.reward(reward, wall_ltp)
                self.reward(reward / 10, x_ltp)
                self.reward(reward / 10, y_ltp)

                self.reward(-0.5 * reward, ltd)
                self.reward(-0.5 * reward, iltd)
                self.reward(-reward, wall_ltd)
                self.reward(-reward / 10, x_ltd)
                self.reward(-reward / 10, y_ltd)

                pre_ltp = self.net.P_pre > 0.1
                post_ltp = self.net.P_post > 0.2

                pre_ltd = self.net.P_pre <= 0.2
                post_ltd = self.net.P_post <= 0.2

                self.net.reward[pre_ltp, :] += 0.002 * delta_time
                self.net.reward[:, post_ltp] += 0.002 * delta_time

                self.net.reward[pre_ltd, :] -= 0.002 * delta_time
                self.net.reward[:, post_ltd] -= 0.002 * delta_time

        if self.check_target_collision():
            self.target.on_collision()
            self.collision_count += 1
            self.time_accum = 0
            self.last_time = util.Time.cur_time

            x = self.x / self.screen.get_width() - 0.5
            y = self.y / self.screen.get_height() - 0.5

            place_x = Agent.IN_PX if x > 0 else Agent.IN_NX
            place_y = Agent.IN_PY if y > 0 else Agent.IN_NY

            x_ltp = np.s_[place_x, :]
            x_ltd = np.s_[:0]

            y_ltp = np.s_[place_y, :]
            y_ltd = np.s_[:0]

            self.inputs[Agent.IN_TARG] = self.I_scale

            self.reward(self.I_scale, np.s_[:self.net.num_exc, Agent.MOTOR_START:Agent.MOTOR_LAST + 1])
            self.reward(self.I_scale / 2, x_ltp)
            self.reward(self.I_scale / 2, y_ltp)

            # Spike trace masks
            pre_ltp = self.net.P_pre > 0.1
            post_ltp = self.net.P_post > 0.1

            pre_ltd = self.net.P_pre <= 0.15
            post_ltd = self.net.P_post <= 0.15

            # Recurrent "hidden" neurons
            pre_ltp = self.net.P_pre > 0.1
            post_ltp = self.net.P_post > 0.1

            pre_ltd = self.net.P_pre <= 0.15
            post_ltd = self.net.P_post <= 0.15

            self.net.reward[pre_ltp, :] += 0.001 * delta_time
            self.net.reward[:, post_ltp] += 0.001 * delta_time

            self.net.reward[pre_ltd, :] -= 0.001 * delta_time
            self.net.reward[:, post_ltd] -= 0.001 * delta_time

            for f in self.on_target_found:
                f()

        angle = self.get_angle()

        self.last_angle_to_target = self.angle_to_target
        self.angle_to_target = angle

        if np.abs(angle) <= np.abs(self.last_angle_to_target):

            reward = 0.2 * np.abs(angle) * delta_time

            self.inputs[Agent.IN_LOOK] += reward
            self.reward(reward, np.s_[Agent.IN_LOOK, :])

            # self.reward(reward, np.s_[:self.net.num_exc, Agent.MOTOR_START:Agent.MOTOR_LAST + 1])
            # self.reward(-reward, np.s_[self.net.num_exc:, Agent.MOTOR_START:Agent.MOTOR_LAST + 1])

            pre_ltp = self.net.P_pre > 0.15
            post_ltp = self.net.P_post > 0.15

            pre_ltd = self.net.P_pre <= 0.1
            post_ltd = self.net.P_post <= 0.1

            self.net.reward[pre_ltp, :] += 0.01 * delta_time
            self.net.reward[:, post_ltp] += 0.01 * delta_time

            self.net.reward[pre_ltd, :] -= 0.005 * delta_time
            self.net.reward[:, post_ltd] -= 0.005 * delta_time

        if np.abs(angle) < np.pi / 4:
            self.look_time += 0.01 * delta_time

            reward = 0.5 * self.I_scale * (np.pi / 4 - np.abs(angle)) / (np.pi / 4) * self.look_time * delta_time

            self.inputs[Agent.IN_LOOK] += reward
            self.reward(reward, np.s_[Agent.IN_LOOK, :])
            # self.reward(reward, np.s_[Agent.IN_COSA, :])
            # self.reward(reward, np.s_[Agent.IN_SINA :])

            # inds = self.get_forward_presynaptics()
            # inds = self.get_motor_presynaptics()
            # self.reward((reward, -reward), where=inds)

            self.reward(reward, np.s_[:self.net.num_exc, Agent.MOTOR_START:Agent.MOTOR_LAST + 1])
            self.reward(-reward, np.s_[self.net.num_exc:, Agent.MOTOR_START:Agent.MOTOR_LAST + 1])

            pre_ltp = self.net.P_pre >= 0.2
            post_ltp = self.net.P_post >= 0.2

            pre_ltd = self.net.P_pre <= 0.1
            post_ltd = self.net.P_post <= 0.1

            self.net.reward[pre_ltp, :] += 0.1 * delta_time
            self.net.reward[:, post_ltp] += 0.1 * delta_time

            self.net.reward[pre_ltd, :] -= 0.005 * delta_time
            self.net.reward[:, post_ltd] -= 0.005 * delta_time

            # self.net.reward[Agent.IN_COSA, :] += 0.05
            # self.net.reward[Agent.IN_SINA, :] += 0.05
        else:
            self.look_time = 0

            pre_ltd = self.net.P_pre <= 0.15
            post_ltd = self.net.P_post <= 0.15

            self.net.reward[pre_ltd, :] -= 0.005 * delta_time
            self.net.reward[:, post_ltd] -= 0.005 * delta_time

        target_x = self.target.x
        target_y = self.target.y

        self.x = max(5, self.x)
        self.x = min(self.x, self.screen.get_width() - 5)

        self.y = max(5, self.y)
        self.y = min(self.y, self.screen.get_height() - 5)

        self.pos = (self.x, self.y)

        x = self.x / self.screen.get_width() - 0.5
        y = self.y / self.screen.get_height() - 0.5

        dx = (target_x - self.x) / self.screen.get_width()
        dy = (target_y - self.y) / self.screen.get_height()
        dist = np.linalg.norm((dx, dy))

        sx = np.sign(dx)
        sy = np.sign(dy)

        if self.prev_dist - dist > 0 and np.abs(angle) < np.pi / 4:
            reward = 0.8 * (1 - np.abs(dist)) * delta_time

            self.inputs[Agent.IN_NEAR] += reward

            # post_ltp = self.net.P_post > 0.2

            ltp = np.s_[Agent.IN_NEAR, :]

            self.reward(reward, ltp)

            # self.net.w[post_ltp] += 0.1 * delta_time

        elif self.prev_dist - dist < 0:
            ltd = np.s_[Agent.IN_NEAR, :]

            reward = 0.6 * (1 - np.abs(dist)) * delta_time

            self.reward(-reward, ltd)

        self.prev_dist = dist

        self.inputs[Agent.IN_DXP] = sx * self.I_scale * np.exp(-2 * np.abs(dx))
        self.inputs[Agent.IN_DYP] = sy * self.I_scale * np.exp(-2 * np.abs(dy))
        self.inputs[Agent.IN_DXN] = -sx * self.I_scale * np.exp(-2 * np.abs(dx))
        self.inputs[Agent.IN_DYN] = -sy * self.I_scale * np.exp(-2 * np.abs(dy))
        self.inputs[Agent.IN_DST] = self.I_scale * np.exp(-2 * dist)
        self.inputs[Agent.IN_WALL_L] *= self.I_scale * np.exp(-0.1 / 10)
        self.inputs[Agent.IN_WALL_R] *= self.I_scale * np.exp(-0.1 / 10)
        self.inputs[Agent.IN_TARG] *= self.I_scale * np.exp(-0.1 / 100)
        self.inputs[Agent.IN_LOOK] *= self.I_scale * np.exp(-0.1 / 25)
        self.inputs[Agent.IN_NEAR] *= self.I_scale * np.exp(-0.1 / 25)
        self.inputs[Agent.IN_COSA] = self.I_scale * np.cos(angle)
        self.inputs[Agent.IN_SINA] = self.I_scale * np.sin(angle)
        self.inputs[Agent.IN_PX] = self.I_scale * x
        self.inputs[Agent.IN_PY] = self.I_scale * y
        self.inputs[Agent.IN_NX] = -self.I_scale * x
        self.inputs[Agent.IN_NY] = -self.I_scale * y
        # ltp = np.s_[:self.net.num_exc, Agent.MOTOR_LEFT:Agent.MOTOR_RIGHT]

        # self.net.reward[:self.net.num_exc, Agent.MOTOR_LEFT] = 1
        # self.net.w[:self.net.num_exc, Agent.MOTOR_LEFT] = 1
        # print(self.net.w[self.net.num_exc])

        self.inputs *= int(self.inputs_enabled)

        self.inputs[:] = np.clip(self.inputs, -self.I_scale, self.I_scale)

        self.net.update()

        self.L_fr = self.firing_rates[Agent.MOTOR_LEFT]
        self.R_fr = self.firing_rates[Agent.MOTOR_RIGHT]

        L = self.L_fr / self.net.params.F_t
        R = self.R_fr / self.net.params.F_t

        # self.angle += np.clip(Agent.ROTATION_SPEED * self.L_fr, 0, Agent.MAX_ROTATION)
        # self.angle -= np.clip(Agent.ROTATION_SPEED * self.R_fr, 0, Agent.MAX_ROTATION)

        self.angle += Agent.ROTATION_SPEED * L
        self.angle -= Agent.ROTATION_SPEED * R

        self.angle = np.mod(self.angle, 2 * np.pi)

        self.x += Agent.MOVE_SPEED * np.cos(self.angle) * (L + R) /  2
        self.y += Agent.MOVE_SPEED * np.sin(self.angle) * (L + R) /  2
        # self.x += np.clip(Agent.MOVE_SPEED * np.cos(self.angle) * (L + R) /  2, -Agent.MAX_SPEED, Agent.MAX_SPEED)
        # self.y += np.clip(Agent.MOVE_SPEED * np.sin(self.angle) * (L + R) /  2, -Agent.MAX_SPEED, Agent.MAX_SPEED)
        self.pos = (self.x, self.y)

        self.forward = (np.cos(self.angle), np.sin(self.angle))

        # Use average firing rate of left and right controllers
        self.pos = (self.x, self.y)

        self.forward = (np.cos(self.angle), np.sin(self.angle))

        # Use average firing rate of left and right controllers
        # to direct forward motion
        # self.x += np.cos(self.angle) * (L + R) / (2 * Agent.MOVE_CONST)
        # self.y += np.sin(self.angle) * (L + R) / (2 * Agent.MOVE_CONST)

    def set_target(self, target=None):
        self.target = target

    def get_distance(self):
        if not self.target:
            return 0

        dx = self.target.x - self.x
        dy = self.target.y - self.y

        return np.sqrt(dx**2 + dy**2)

    def get_closest_wall_angle(self):
        num_rays = 10
        rect = self.screen.get_rect()
        half_width = np.pi / 4
        angle_range = np.linspace(0, half_width, num_rays)

        distances = np.ones(2 * num_rays) * np.inf
        angles = np.ones(2 * num_rays) * np.pi

        for i, a in enumerate(angle_range):
            d = self.get_wall_distance(self.angle + a)
            x = self.x + d * np.cos(self.angle + a)
            y = self.y + d * np.sin(self.angle + a)

            if not rect.collidepoint((x, y)):
                distances[i] = d
                angles[i] = a

            d = self.get_wall_distance(self.angle - a)
            x = self.x + d * np.cos(self.angle - a)
            y = self.y + d * np.sin(self.angle - a)

            if not rect.collidepoint((x, y)):
                distances[i + num_rays] = d
                angles[i + num_rays] = -a
        return angles[np.argmin(distances)]

    def get_wall_distance(self, angle):
        rect = self.screen.get_rect()

        distance = 10

        while True:
            x = self.x + distance * np.cos(angle)
            y = self.y + distance * np.sin(angle)

            if not rect.collidepoint((x, y)):
                return distance

            distance += 5

    def get_wall_angle(self):
        rect = self.screen.get_rect()
        angle = self.angle
        half_width = np.pi / 4

        angles = np.linspace(0, half_width, 20)
        for a in angles:
            x_l = self.x + 40 * np.cos(angle + a)
            y_l = self.y + 40 * np.sin(angle + a)

            if not rect.collidepoint((x_l, y_l)):
                return a
            
            x_r = self.x + 40 * np.cos(angle - a)
            y_r = self.y + 40 * np.sin(angle - a)

            if not rect.collidepoint((x_r, y_r)):
                return -a

        return np.pi

    def get_angle(self):
        if not self.target:
            return 0

        # Testing a quick hack for inputs
        u = self.forward
        v = (self.target.x - self.x, self.target.y - self.y)

        norm = np.linalg.norm(v)
        v /= norm if norm != 0 else 1

        d = np.dot(u, v)

        angle = np.arccos(d)

        c = u[0] * v[1] - u[1] * v[0]

        return angle * np.sign(c)

    def check_target_collision(self):
        if self.bounds.colliderect(self.target.bounds):
            return True
        return False

    def check_in_bounds(self):
        rect = self.screen.get_rect().inflate(-10, -10)
        return rect.collidepoint(self.x, self.y)

    def check_wall_ahead(self):
        forward = self.forward
        forward = (forward[0], forward[1])
        points = [
            (self.x + 30 * forward[0], self.y + 30 * forward[1]),
            (self.x + 30 * np.cos(self.angle + np.pi/4), self.y + 30 * np.sin(self.angle + np.pi/4)),
            (self.x + 30 * np.cos(self.angle - np.pi/4), self.y - 30 * np.sin(self.angle + np.pi/4))
        ]

        bounds = self.screen.get_rect()

        return np.any([not bounds.collidepoint(p) for p in points])

    def get_motor_presynaptics(self):
        motor_start = Agent.MOTOR_START
        motor_end = Agent.MOTOR_LAST + 1
        exc = np.s_[:self.net.num_exc, motor_start:motor_end]
        inh = np.s_[self.net.num_exc:, motor_start:motor_end]

        return (exc, inh)

    def get_left_presynaptics(self):
        motor_start = Agent.MOTOR_LEFT
        motor_end = motor_start + Agent.NUM_MOTORS
        exc = np.s_[:self.net.num_exc, motor_start:motor_end]
        inh = np.s_[self.net.num_exc:, motor_start:motor_end]

        return (exc, inh)

    def get_right_presynaptics(self):
        motor_start = Agent.MOTOR_RIGHT
        motor_end = motor_start + Agent.NUM_MOTORS
        exc = np.s_[:self.net.num_exc, motor_start:motor_end]
        inh = np.s_[self.net.num_exc:, motor_start:motor_end]

        return (exc, inh)


    def get_wall_postsynaptics(self):
        # return np.s_[Agent.IN_WALL:Agent.IN_WALL + 1, :]
        # print(self.net.reward[np.s_[Agent.IN_WALL:Agent.IN_WALL + 1, :]].shape)
        return np.s_[Agent.IN_WALL:Agent.IN_WALL + 1, :]

    def reward(self, r=0, where=None, clip=None):
        where = np.s_[:0] if where is None else where
        clip = clip or (self.net.params.r_min, self.net.params.r_max)
        self.net.reward[where] = np.clip(self.net.reward[where] + r, *clip)


    # def reward(self, r=None, r_p=0, r_n=0, where=None, where_exc=None, where_inh=None, reason=None):
    #     if where is not None:
    #         where_exc, where_inh = where
    #     # where_exc = where_exc or np.s_[:self.net.num_exc,:]
    #     # where_inh = where_inh or np.s_[self.net.num_exc:, :]

    #     if r is not None:
    #         r_p, r_n = r

    #     self.net.reward[where_exc] += r_p
    #     self.net.reward[where_inh] += r_n

    def set_pos(self, pos):
        self.x, self.y = pos
        self.pos = pos

    def toggle_inputs(self):
        self.inputs_enabled = not self.inputs_enabled

