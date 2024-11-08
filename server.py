import flwr as fl
import sys


def get_on_fit_config_fn(cluster = None):
    def fit_config(rnd: int ):
        """Return training configuration dict for each round.
        
        """
        config = {
            "current_round": rnd,  # The current round of federated learning
            "local_epochs": 2 if rnd < 2 else 3,  # 
            "rounds": 5,
            "cluster": cluster if cluster else ""
        }
        return config
        
    return fit_config

def main(argv):
    if (len(argv)==0):
        cluster = None
    else:
        cluster = argv[0]

    strategy = fl.server.strategy.FedAvg(
        on_fit_config_fn=get_on_fit_config_fn(cluster)
        # on_fit_config_fn=fit_config
    )

    ip = "localhost:500{}".format(cluster if cluster else 5)
    print('Running at: {}'.format(ip))
    # fl.server.start_server(ip, config={"num_rounds": 5})
    fl.server.start_server(server_address = ip, config={"num_rounds": 5}, strategy = strategy)


if __name__ == "__main__":
    main(sys.argv[1:])    