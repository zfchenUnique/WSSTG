def creatDataloader(opts):
   from data.custom_data_loader import customDataLoader
   data_loader = customDataLoader()
   data_loader.initialize(opt)
   return data_loader
