import math
import matplotlib.pyplot as plt
import numpy as np
import random
import time

from collections import defaultdict
from euclid import Circle, Point2, Vector2, LineSegment2

from ..utils import svg
from IPython.display import clear_output, display, HTML

import cv2

class GameObject(object):
    def __init__(self, position, velocity, speed, obj_type, settings):
        """Esentially represents circles of different kinds, which have
        position and velocity."""
        self.settings = settings
        self.radius = self.settings["object_radius"]

        self.obj_type = obj_type
        self.position = position
        self.velocity    = velocity
        self.bounciness = 1.0
        self.speed = speed


    def wall_collisions(self):
        """Update velocity upon collision with the wall."""
        
        world_size = self.settings["world_size"]

        for dim in range(2):
            if self.position[dim] - self.radius       <= 0               and self.velocity[dim] < 0:
                self.velocity[dim] = - self.velocity[dim] * self.bounciness
            elif self.position[dim] + self.radius + 1 >= world_size[dim] and self.velocity[dim] > 0:
                self.velocity[dim] = - self.velocity[dim] * self.bounciness
        

    def move(self, dt):
        """Move as if dt seconds passed"""
        self.position += dt * self.velocity
        self.position = Point2(*self.position)
        '''
        if self.position[0] >= self.settings["world_size"][0]:
            self.position[0] -= self.settings["world_size"][0]
        if self.position[1] >= self.settings["world_size"][1]:
            self.position[1] -= self.settings["world_size"][1]
        if self.position[0] < 0.0:
            self.position[0] += self.settings["world_size"][0]
        if self.position[1] < 0.0:
            self.position[1] += self.settings["world_size"][1]
        '''

    def step(self, dt):
        """Move and bounce of walls."""
        self.wall_collisions()
        self.move(dt)

    def as_circle(self):
        return Circle(self.position, float(self.radius))

    def draw(self, vis):
        """Return svg object for this item."""
        color = self.settings["colors"][self.obj_type]
        draw_position = (int(self.position[0] + 10), int(self.position[1] + 10))
        cv2.circle(vis, draw_position, int(self.radius), color, -1, cv2.LINE_AA)


class KarpathyGame(object):
    def __init__(self, settings):
        """Initiallize game simulator with settings"""
        self.settings = settings
        self.size  = self.settings["world_size"]
        self.walls = [LineSegment2(Point2(0,0),                        Point2(0,self.size[1])),
                      LineSegment2(Point2(0,self.size[1]),             Point2(self.size[0], self.size[1])),
                      LineSegment2(Point2(self.size[0], self.size[1]), Point2(self.size[0], 0)),
                      LineSegment2(Point2(self.size[0], 0),            Point2(0,0))]


        self.num_active = [self.settings['num_objects_active']['prey'], self.settings['num_objects_active']['pred']] #number of [prey, pred] controlling movement

        self.objects = []

        for obj_type, number in settings["num_objects"].items():
            for _ in range(number):
                self.objects.append(self.spawn_object(obj_type))


        self.observation_lines = self.generate_observation_lines()

        self.object_reward = np.zeros(self.num_active[0] + self.num_active[1])
        self.collected_rewards = []

        # every observation_line sees one of objects or wall and
        # two numbers representing velocity of the object (if applicable)
        self.eye_observation_size = len(self.settings["objects"]) + 3
        # additionally there are two numbers representing agents own velocity and position.
        self.observation_size = self.eye_observation_size * len(self.observation_lines) + 2 + 2

        #Four possible movement directions plus stationary
        self.directions = [Vector2(*d) for d in [[1,0], [0,1], [-1,0],[0,-1],[0.0,0.0]]]

        self.num_actions = len(self.directions)
        print('num_actions ', self.num_actions)

        #self.objects_eaten = defaultdict(lambda: 0)
        self.objects_eaten = {'prey':{'prey':0, 'pred':0},'pred':{'prey':0, 'pred':0}}
        self.num_acts_so_far = 0
        self.list_to_remove = []

        tests = float(self.settings["maximum_velocity"]['prey'])

        self.max_velocity = max(float(self.settings["maximum_velocity"]['prey']),
                                    float(self.settings["maximum_velocity"]['pred']))

    def perform_action(self, action_id, obj_type):
        """Change velocity to one of the individual's vectors"""
        for ind, active_ind in enumerate(self.get_list(obj_type)):
            assert 0 <= action_id[ind] < self.num_actions
            self.objects[active_ind].velocity *= 0.5
            self.objects[active_ind].velocity += (
            	self.directions[int(action_id[ind])] * self.settings["delta_v"])

	        


    def spawn_object(self, obj_type, ):
        """Spawn object of a given type and add it to the objects array"""
        radius = self.settings["object_radius"]
        position = np.random.uniform([radius, radius], np.array(self.size) - radius)
        position = Point2(float(position[0]), float(position[1]))
        max_velocity = float(self.settings["maximum_velocity"][obj_type])
        velocity = np.random.uniform(-max_velocity, max_velocity, 2).astype(float)
        velocity = Vector2(float(velocity[0]), float(velocity[1]))
        return GameObject(position, velocity, max_velocity, obj_type, self.settings) 
        



    def step(self, dt, num_types):
        """Simulate all the objects for a given ammount of time.
           and resolve collisions with other objects"""
        for obj in self.objects:
            obj.step(dt)
        for i in range(num_types):
            self.resolve_collisions(i)
        #generate new individuals post collision
        for i in self.list_to_remove:
            self.objects[i] = self.spawn_object(self.objects[i].obj_type)
        self.list_to_remove = []


    def squared_distance(self, p1, p2):
        return (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2

    def get_list(self, obj_type):
        gen = False
        #Currently will never be true, skip to else statement for understanding
        if gen:
            num_observations = 0 #how many objects have observed so far
            active_ind = 0 #the object that is currently observing
            members = []
            while num_observations < self.num_active[obj_type]:
            #THIS MIGHT BE A PLACE TO OPTIMIZE
                #find an object of the correct type
                if self.objects[active_ind].obj_type != self.settings['objects'][obj_type]:
                    active_ind += 1
                    continue
                num_observations += 1
                members.append(active_ind)
                active_ind += 1
            return members
        #Get the active prey and predator positions in the object list
        else:
            if obj_type == 1:
                return(range(self.settings['num_objects']['prey'], self.settings['num_objects']['prey'] + self.num_active[obj_type]))
            elif obj_type == 0:
                return(range(self.num_active[obj_type]))


    def resolve_collisions(self, obj_type):
        """If hero touches, hero eats. Also reward gets updated."""
        #assumes all individuals are the same size
        collision_distance = 2 * self.settings["object_radius"]
        collision_distance2 = collision_distance ** 2
        for ind, active_ind in enumerate(self.get_list(obj_type)):
 
            to_remove = []

            for i, obj in enumerate(self.objects): 
                if i == active_ind:
                    continue 
                if self.squared_distance(self.objects[active_ind].position, obj.position) < collision_distance2:
                    to_remove.append(i)
            for i in to_remove:

                self.objects_eaten[self.objects[active_ind].obj_type][self.objects[i].obj_type] += 1
                self.object_reward[obj_type * self.num_active[0]  + ind] += self.settings["object_reward"][self.objects[active_ind].obj_type][self.objects[i].obj_type]
                if self.objects[active_ind].obj_type == self.objects[i].obj_type:
                    self.list_to_remove.append(i)  #comment out this line to allow prey to hit each other
                    continue
                else:
                    self.list_to_remove.append(i)
                #if obj_type != 0: #if predator ind = 1, other must be prey (ind == 0) NEEDS TO BE CHANGED FOR >2 OBJECT TYPES  
                #    #print('here')
                #    self.list_to_remove.append(i)

                
               

    def inside_walls(self, point):
        """Check if the point is inside the walls"""
        EPS = 1e-4
        return (EPS <= point[0] < self.size[0] - EPS and
                EPS <= point[1] < self.size[1] - EPS)

    def observe(self, obj_type):
        """Return observation vector. For all the observation directions it returns representation
        of the closest object to the hero - might be nothing, another object or a wall.
        Representation of observation for all the directions will be concatenated.
        """
        num_obj_types = len(self.settings["objects"]) + 1 # and wall
        #max_velocity_x, max_velocity_y = self.settings["maximum_velocity"]

        observable_distance = self.settings["observation_line_length"]

        observation = np.zeros((self.num_active[obj_type], self.observation_size), dtype = float)

        for ind, active_ind in enumerate(self.get_list(obj_type)):
            #test = self.objects[active_ind].position[0] #to delete

            relevant_objects = [obj for obj in self.objects[:active_ind] + self.objects[active_ind + 1:] 
                                if obj.position.distance(self.objects[active_ind].position) < observable_distance]


            # objects sorted from closest to furthest
            relevant_objects.sort(key=lambda x: x.position.distance(self.objects[active_ind].position))

            observation_offset = 0
            for i, observation_line in enumerate(self.generate_observation_lines()):
                # shift to hero position
                observation_line = LineSegment2(self.objects[active_ind].position + Vector2(*observation_line.p1),
                                                self.objects[active_ind].position + Vector2(*observation_line.p2))
                observed_object = None
                # if end of observation line is outside of walls, we see the wall.
                if not self.inside_walls(observation_line.p2):
                    observed_object = "**wall**"
                for obj in relevant_objects:
                    if observation_line.distance(obj.position) < self.settings["object_radius"]:
                        observed_object = obj
                        break
                object_type_id = None
                velocity_x, velocity_y = 0, 0
                proximity = 0
                if observed_object == "**wall**": # wall seen
                    object_type_id = num_obj_types - 1
                    # a wall has no velocity
                    velocity_x, velocity_y = 0, 0
                    # best candidate is intersection between
                    # observation_line and a wall, that's
                    # closest to the individual
                    best_candidate = None
                    for wall in self.walls:
                        candidate = observation_line.intersect(wall)
                        if candidate is not None:
                            if (best_candidate is None or
                                    best_candidate.distance(self.objects[active_ind].position) >
                                    candidate.distance(self.objects[active_ind].position)):
                                best_candidate = candidate
                    if best_candidate is None:
                        # assume it is due to rounding errors
                        # and wall is barely touching observation line
                        proximity = observable_distance
                    else:
                        proximity = best_candidate.distance(self.objects[active_ind].position)
                elif observed_object is not None: # agent seen
                    object_type_id = self.settings["objects"].index(observed_object.obj_type)
                    velocity_x, velocity_y = tuple(observed_object.velocity)
                    intersection_segment = obj.as_circle().intersect(observation_line)
                    assert intersection_segment is not None
                    try:
                        proximity = min(intersection_segment.p1.distance(self.objects[active_ind].position),
                                        intersection_segment.p2.distance(self.objects[active_ind].position))
                    except AttributeError:
                        proximity = observable_distance
                for object_type_idx_loop in range(num_obj_types):
                    observation[ind, observation_offset + object_type_idx_loop] = 1.0
                if object_type_id is not None:
                    observation[ind, observation_offset + object_type_id] = proximity / observable_distance
          
                observation[ind, observation_offset + num_obj_types] =     velocity_x   / self.max_velocity  #constant velocities now
                observation[ind, observation_offset + num_obj_types + 1] = velocity_y   / self.max_velocity
                assert num_obj_types + 2 == self.eye_observation_size
                observation_offset += self.eye_observation_size

            #add velocity information about the focal individual
            observation[ind, observation_offset]     = self.objects[active_ind].velocity[0] / self.max_velocity
            observation[ind, observation_offset + 1] = self.objects[active_ind].velocity[1] / self.max_velocity
            observation_offset += 2
            
            # add normalized locaiton of the focal individual in environment        
            observation[ind, observation_offset]     = self.objects[active_ind].position[0] / (self.size[0] / 2) - 1.0  #originally was / 350.0
            observation[ind, observation_offset + 1] = self.objects[active_ind].position[1] / (self.size[1] / 2) - 1.0  # orginally was / 250.0

            assert observation_offset + 2 == self.observation_size

        return observation

    def distance_to_walls(self, obj):
        """Returns distance of a hero to walls"""
        res = float('inf')
        for wall in self.walls:
            res = min(res, obj.position.distance(wall))
        return res - self.settings["object_radius"]

    def collect_reward(self, obj_type):
        """Return accumulated object eating score + current distance to walls score"""
        
        total_reward = np.zeros(self.num_active[obj_type])

        for ind, active_ind in enumerate(self.get_list(obj_type)):

            if self.settings["wall_distance_penalty"] > 0: 
                wall_reward =  (self.settings["wall_distance_penalty"] * 
                               np.exp(-self.distance_to_walls(self.objects[active_ind]) / self.settings["tolerable_distance_to_wall"]))
                assert wall_reward < 1e-3, "You are rewarding hero for being close to the wall!"
            else:
                wall_reward = 0
            
            total_reward[ind] = wall_reward + self.object_reward[obj_type * self.num_active[0]  + ind]
            self.object_reward[obj_type * self.num_active[0]  + ind] = 0
            self.collected_rewards.append(total_reward[ind])
        return total_reward

    def update_num_actions(self, num_acts):
        self.num_acts_so_far = num_acts

    def plot_reward(self, smoothing = 30):
        """Plot evolution of reward over time."""
        plottable = self.collected_rewards[:]
        while len(plottable) > 1000:
            for i in range(0, len(plottable) - 1, 2):
                plottable[i//2] = (plottable[i] + plottable[i+1]) / 2
            plottable = plottable[:(len(plottable) // 2)]
        x = []
        for  i in range(smoothing, len(plottable)):
            chunk = plottable[i-smoothing:i]
            x.append(sum(chunk) / len(chunk))
        plt.plot(list(range(len(x))), x)

    def generate_observation_lines(self):
        """Generate observation segments in settings["num_observation_lines"] directions"""
        result = []

        start = Point2(0.0, 0.0)
        end   = Point2(self.settings["observation_line_length"],
                       self.settings["observation_line_length"])

        for angle in np.linspace(0, 2*np.pi, self.settings["num_observation_lines"], endpoint=False):
            rotation = Point2(math.cos(angle), math.sin(angle))
            current_start = Point2(start[0] * rotation[0], start[1] * rotation[1])
            current_end   = Point2(end[0]   * rotation[0], end[1]   * rotation[1])
            result.append( LineSegment2(current_start, current_end))
        return result

    def _repr_html_(self):
        return self.to_html()

    def to_html(self, stats=[]):
        """Return svg representation of the simulator"""

        stats = stats[:]
        recent_reward = self.collected_rewards[-500:] + [0]
        objects_eaten_str = ', '.join(["%s: %s" % (o,c) for o,c in self.objects_eaten.items()])
        stats.extend([
            #"nearest wall = %.1f" % (self.distance_to_walls(self.objects[0]),),
            "reward       = %.7f" % (sum(recent_reward)/float(len(recent_reward))*100.0,),
            "Objects Eaten => %s" % (objects_eaten_str,),
            "Number of actions so far => %.1f" % (self.num_acts_so_far,),

        ])

        scene = svg.Scene((self.size[0] + 20, self.size[1] + 20 + 20 * len(stats)))
        scene.add(svg.Rectangle((10, 10), self.size))

        '''
        for line in self.generate_observation_lines(self.objects[0].angle):
            scene.add(svg.Line(line.p1 + self.objects[0].position + Point2(10,10),
                               line.p2 + self.objects[0].position + Point2(10,10)))
        '''
        for obj in self.objects:
            scene.add(obj.draw())

        offset = self.size[1] + 15
        for txt in stats:
            scene.add(svg.Text((10, offset + 20), txt, 15))
            offset += 20

        return scene

    def setup_draw(self):
        """
        An optional method to be triggered in simulate(...) to initialise
        the figure handles for rendering.
        simulate(...) will run with/without this method declared in the simulation class
        As we are using SVG strings in KarpathyGame, it is not curently used.
        """
        pass

    def draw(self, stats=[]):
        """
        An optional method to be triggered in simulate(...) to render the simulated environment.
        It is repeatedly called in each simulated iteration.
        simulate(...) will run with/without this method declared in the simulation class.
        """

        stats = stats[:]
        recent_reward = self.collected_rewards[-500:] + [0]
        objects_eaten_str = ', '.join(["%s: %s" % (o,c) for o,c in self.objects_eaten.items()])
        stats.extend([
            #"nearest wall = %.1f" % (self.distance_to_walls(self.objects[0]),),
            "reward = %.7f" % (sum(recent_reward)/float(len(recent_reward))*100.0,),
            "Objects Eaten => %s" % (objects_eaten_str,),
            "Number of actions so far => %.1f" % (self.num_acts_so_far,),

        ])

        visualisation = np.zeros((self.size[1] + 20 + 20 * len(stats), self.size[0] + 20, 3), np.uint8) 
        visualisation[:,:,0] = visualisation[:,:,0] + 160 
        visualisation[:,:,1] = visualisation[:,:,1] + 124
        visualisation[:,:,2] = visualisation[:,:,2] + 110
        

        cv2.rectangle(visualisation, (10, 10), (self.size[0]+10, self.size[1]+10), [255,255,255])

        '''
        for line in self.generate_observation_lines(self.objects[0].angle):
            scene.add(svg.Line(line.p1 + self.objects[0].position + Point2(10,10),
                               line.p2 + self.objects[0].position + Point2(10,10)))
        '''
        
        for obj in self.objects:
            obj.draw(visualisation)

        offset = self.size[1] + 15
        for txt in stats:
            cv2.putText(visualisation, txt, (10, offset + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                        [84, 37, 0], lineType=cv2.LINE_AA)
            offset += 20

        cv2.imshow('rl_schooling', visualisation)
        cv2.waitKey(1)

    def shut_down_graphics(self):
        for indx in range(20):
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)
