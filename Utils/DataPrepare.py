class DP_worker(args):
    def __init__(self, args):
        self.args = args
    
    def work(self):
        #convert to jpegs
        videos_temp = self.args.Clips_temp_directory
        for file_name in os.listdir(self.args.ML_videos_directory):
            if not file_name.endswith('.mp4'):
                continue
            print(file_name)
        #calculate means
        
        