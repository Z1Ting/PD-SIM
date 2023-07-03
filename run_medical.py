
def main():
   
    mid = open("config.yaml", "r")

   
    config = yaml.load(mid, Loader=yaml.FullLoader)

    dataset = DataSetWrapper(config['batch_size'], **config['dataset'])
    simclr = SimCLR_medical(dataset, config)
    simclr.train()




if __name__ == "__main__":
    main()
