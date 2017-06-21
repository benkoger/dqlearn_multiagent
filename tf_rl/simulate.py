from __future__ import division

import math
import time

import matplotlib.pyplot as plt
from itertools import count
from os.path import join, exists
from os import makedirs
from IPython.display import clear_output, display, HTML

def simulate(simulation,
             controller= None,
             fps=60,
             visualize_every=1,
             action_every=1,
             simulation_resolution=None,
             wait=False,
             disable_training=False,
             save_path=None):
    """Start the simulation. Performs three tasks

        - visualizes simulation
        - advances simulator state
        - reports state to controller and chooses actions
          to be performed.

    Parameters
    -------
    simulation: tr_lr.simulation
        simulation that will be simulated ;-)
    controller: tr_lr.controller
        controller used
    fps: int
        frames per seconds
    visualize_every: int
        visualize every `visualize_every`-th frame.
    action_every: int
        take action every `action_every`-th frame
    simulation_resolution: float
        simulate at most 'simulation_resolution' seconds at a time.
        If None, the it is set to 1/FPS (default).
    wait: boolean
        whether to intentionally slow down the simulation
        to appear real time.
    disable_training: bool
        if true training_step is never called.
    save_path: str
        save svg visualization (only tl_rl.utils.svg
        supported for the moment)
    """

    # prepare path to save simulation images
    if save_path is not None:
        if not exists(save_path):
            makedirs(save_path)
    last_image = 0

    # calculate simulation times
    chunks_per_frame = 1
    chunk_length_s   = 1.0 / fps

    if simulation_resolution is not None:
        frame_length_s = 1.0 / fps
        chunks_per_frame = int(math.ceil(frame_length_s / simulation_resolution))
        chunks_per_frame = max(chunks_per_frame, 1)
        chunk_length_s = frame_length_s / chunks_per_frame
        print(chunks_per_frame)

    # state transition bookkeeping
    last_observation = [None] * len(controller) 
    last_action     = [None] * len(controller)

    simulation_started_time = time.time()

    for frame_no in count():
        for _ in range(chunks_per_frame):
            #move all the individuals for one time step
            #deal with any collisions
            simulation.step(chunk_length_s, len(controller))

        if frame_no % action_every == 0:
            #observations of object type 0 and object type 1
            new_observation = [None] * len(controller)
            reward          = [None] * len(controller) 
            new_action = [None] * len(controller)
            
            #go through each network group 
            for i in range(len(controller)):

                new_observation[i] = simulation.observe(i)
                
                #Only Prey are participating, Predators have fixed behavior
                if i == 0:
                    reward[i] = simulation.collect_reward(i)
                    # store last transition
                    #if not None in last_observation:  #This is how it was
                    if last_observation[i] is not None:
                        controller[i].store(last_observation[i], last_action[i], reward[i], new_observation[i])

                    # act
                    action_info = controller[i].action(new_observation[i]) 
                    new_action[i] = action_info[0]
                    num_actions = action_info[1]
                    simulation.update_num_actions(num_actions)
                    simulation.perform_action(new_action[i], i)


                    #train
                    if not disable_training:
                        controller[i].training_step()

                    # update current state as last state.
                    last_action[i] = new_action[i]
                    last_observation[i] = new_observation[i]

                if frame_no % 1000 == 0 and i == 0:
                    controller[i].save('saved_graphs', False, frame_no)

        # adding 1 to make it less likely to happen at the same time as
        # action taking.
        if (frame_no + 1) % visualize_every == 0:
            fps_estimate = frame_no / (time.time() - simulation_started_time)

            # draw simulated environment all the rendering is handled within the simulation object 
            stats = ["fps = %.1f" % (fps_estimate, )]
            if hasattr(simulation, 'draw'): # render with the draw function
                simulation.draw(stats) 
            elif hasattr(simulation, 'to_html'): # in case some class only support svg rendering
                clear_output(wait=True)
                svg_html = simulation.to_html(stats)
                display(svg_html)

            if save_path is not None:
                img_path = join(save_path, "%d.svg" % (last_image,))
                with open(img_path, "w") as f:
                    svg_html.write_svg(f)
                last_image += 1

        time_should_have_passed = frame_no / fps
        time_passed = (time.time() - simulation_started_time)
        if wait and (time_should_have_passed > time_passed):
            time.sleep(time_should_have_passed - time_passed)


