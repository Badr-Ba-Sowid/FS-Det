import json
import argparse




def parse_config_file(config_file_path: str) -> json:
    with open(config_file_path, 'r') as config_file:
        config_data = json.load(config_file)
        return config_data
    
def init_model(config_file : json):
    pass

def predict():
    pass


def main():
    #Get arguments for config file parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help = "model config file")
    args = parser.parse_args()
    if args.config:
        config_file = parse_config_file(args.config)
        print(config_file)
    else:
        print("No configuration file specified...")
if __name__ == "__main__":
    main()

    