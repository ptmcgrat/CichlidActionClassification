from Utils.DataPrepare import DP_worker
class ML_model(args):
    def __init__(self, args):
        self.args = args
        #prepare the data is the data is not prepared
        log_dir = arg.Log_directory
        json_file = os.path.join(log_dir,'split.jason')
        #check if data preparation is done
        if not os.path.exists(json_file):
            dp_worker = DP_worker(args)
            dp_worker.work()
        
        
    def work(self):
        
        
    