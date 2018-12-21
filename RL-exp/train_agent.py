from myagent import ExpEnv, ExpAgent
import argparse
import os

"""
Train model-selection agent

Usage:
    python3 train_agent.py model_config_json exec_result_pkl weight_dir

"""
if __name__ == '__main__':
    # argument parsing
    parser = argparse.ArgumentParser(description="Train model-selection agent.")
    parser.add_argument("model_config_json", type=str, help="Input configuration file(.json) of models.")
    parser.add_argument("exec_result_pkl", type=str, help="Input execution result file(.pkl) for training.")
    parser.add_argument("weight_dir", type=str, help="Output directory of saved model weights.")
    args = parser.parse_args()
    if not os.path.isdir(args.weight_dir):
        print("{} does not exist. Now created.".format(args.weight_dir))
        os.mkdir(args.weight_dir)

    expenv = ExpEnv(model_config_json=args.model_config_json,
                    exec_result_pkl=args.exec_result_pkl,)
    expenv.reset()

    expagent = ExpAgent(env=expenv)
    for t_round in range(1,11):
        expenv.open_log(os.path.join(args.weight_dir, "log-round-{}.txt".format(t_round)))
        expagent.dqn.fit(expenv, nb_steps=expenv.action_num*expenv.data_num, nb_max_episode_steps=expenv.action_num, log_interval=900, verbose=1)
        expagent.save_model(path=os.path.join(args.weight_dir, "round-{}.pkl".format(t_round)))
        expenv.log.close()
