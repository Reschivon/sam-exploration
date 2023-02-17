# Prevent numpy from using up all cpu
from math import fabs
import os
os.environ['MKL_NUM_THREADS'] = '1'  # pylint: disable=wrong-import-position
import argparse
from pathlib import Path
import numpy as np
import utils
import time

import visualize

def _run_eval(cfg, num_episodes=40):

    # Customize env arguments
    cfg['use_gui'] = False
    cfg['show_state_representation'] = False
    cfg['show_occupancy_map'] = False

    env = utils.get_env_from_cfg(cfg, random_seed=9)
    policy = utils.get_policy_from_cfg(cfg, env.get_action_space(), random_seed=9)
    data = [[] for _ in range(num_episodes)]
    episode_count = 0
    # Index [0] to ignore state_info
    states = [env.reset(robot_index)[0] for robot_index in range(env.num_agents)]
    start_time = time.time()
    end_time = time.time()
    done = False
    while True:
        for robot_index in range(env.num_agents):
            action, _ = policy.step(states[robot_index])
            curr_state, _, curr_done, info = env.step(action, robot_index)
            states[robot_index] = curr_state
            done = True if curr_done else done

        end_time = time.time()
        data[episode_count].append({'cube_found': info['cube_found'],\
                                    'cumulative_distance': info['cumulative_distance'],\
                                    'explored_area': info['explored_area'],\
                                    'overlapped_ratio': info['overlapped_ratio'],\
                                    'non_overlapped_ratio': info['non_overlapped_ratio'],\
                                    'ratio_explored': info['ratio_explored'],\
                                    'repetitive_exploration_rate': info['repetive_exploration_rate']})
        
        time_elasped = end_time - start_time

        exploration_ratio_goal = 0.5


        # Decide if and why the episode should end
        end_episode_condition = ''
        if 'theoretical_exploration' in cfg and cfg['theoretical_exploration']:
            # Complete upon reaching exploration_ratio_goal
            if info['ratio_explored'] > exploration_ratio_goal:
                end_episode_condition = 'theoretical_exploration'
        else:
            # Complete upon seeing cube, or done=True
            if done:
                end_episode_condition = 'found_block'
        if time_elasped > 600:
            end_episode_condition = 'timed_out'

        # Reset all agents and log statistics
        if end_episode_condition != '':
            for robot_index in range(env.num_agents):
                env.reset(robot_index)
            done = False

            start_time = time.time()
            episode_count += 1

            env._visualize_state_representation()

            if end_episode_condition == 'theoretical_exploration':
                print('Completed {}/{} episodes (theoretical exploration)'.format(episode_count, num_episodes))
            if end_episode_condition == 'found_block':
                print('Completed {}/{} episodes (found block)'.format(episode_count, num_episodes))
            if end_episode_condition == 'timed_out':
                print('Failed to complete episode:', episode_count)
                
            if episode_count >= num_episodes:
                break

    return data

def main(args):
    config_path = args.config_path
    if config_path is None:
        config_path = utils.select_run()
    if config_path is None:
        print('Please provide a config path')
        return
    cfg = utils.read_config(config_path)

    eval_dir = Path(cfg.logs_dir).parent / 'eval'
    if not eval_dir.exists():
        eval_dir.mkdir(parents=True, exist_ok=True)

    eval_path = eval_dir / '{}.npy'.format(cfg.run_name)
    data = _run_eval(cfg, num_episodes=200)
    np.save(eval_path, data)
    print(eval_path)

    visualize.visualize(eval_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-path')
    main(parser.parse_args())