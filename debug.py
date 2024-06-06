import argparse

parser = argparse.ArgumentParser()
action_group = parser.add_argument_group('info')
parser.add_argument('--fog_model_input_dim', type=int, default=18, required=False)

args = parser.parse_args()
print(args['info'])
